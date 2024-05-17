import fitz  # PyMuPDF
import os


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text


def extract_sections(text):
    chief_complaint = ""
    impressions = ""
    icd_codes = []

    lines = text.split('\n')
    in_chief_complaint = False
    in_impressions = False
    in_comments = False

    for line in lines:
        line = line.strip()

        if "Chief Complaint" in line:
            in_chief_complaint = True
            in_impressions = False
            in_comments = False
            chief_complaint = line.replace("Chief Complaint:", "").strip()
        elif "Impression" in line:
            in_impressions = True
            in_chief_complaint = False
            in_comments = False
            impressions = line.replace("Impression:", "").strip()
        elif "Comments" in line:
            in_comments = True
            in_chief_complaint = False
            in_impressions = False
        elif in_chief_complaint:
            chief_complaint += " " + line.strip()
        elif in_impressions:
            impressions += " " + line.strip()
        elif in_comments:
            icd_codes.append(line.strip())

    return chief_complaint, impressions, icd_codes


def extract_data_from_pdfs(pdf_folder):
    data = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_text = extract_text_from_pdf(pdf_path)
            chief_complaint, impressions, icd_codes = extract_sections(pdf_text)
            query = f"Chief complaint of {chief_complaint} and Impression of {impressions}"
            for icd_code in icd_codes:
                data.append((query, icd_code))
    return data


# Path to the folder containing the PDF files
pdf_folder = '/mnt/data/pdf_files'
pdf_data = extract_data_from_pdfs(pdf_folder)


# Load ICD-10 data from Excel file
icd_data = pd.read_excel('/mnt/data/ICDMaster.xlsx', usecols=["Code", "Description"])

# Remove duplicate ICD-10 codes, keeping the first occurrence
icd_data = icd_data.drop_duplicates(subset="Code", keep="first")

# Extract texts and labels
texts = icd_data['Description'].tolist()
codes = icd_data['Code'].tolist()

# Encoding ICD-10 codes as integers
code_to_id = {code: idx for idx, code in enumerate(set(codes))}
id_to_code = {idx: code for code, idx in code_to_id.items()}
labels = [code_to_id[code] for code in codes]

# Combine new data with existing ICD-10 data
for query, icd_code in pdf_data:
    if icd_code in code_to_id:
        texts.append(query)
        labels.append(code_to_id[icd_code])

# Split the combined data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# Initialize the tokenizer and model
checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=len(code_to_id))

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

# Create train and validation datasets
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

# Training arguments
args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    output_dir='./results',
    num_train_epochs=10,  # Increase epochs
    learning_rate=2e-5,  # Adjust learning rate
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()
trainer.evaluate()

# Function to predict top-5 ICD-10 codes
def predict_icd10(text, model, tokenizer, top_k=5):
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding="max_length", max_length=128)
    encoding = encoding.to(model.device)
    outputs = model(**encoding)
    logits = outputs.logits[0]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    top_indices = top_indices.cpu().numpy()
    top_probs = top_probs.cpu().detach().numpy()
    predictions = [(id_to_code[idx], icd_data[icd_data['Code'] == id_to_code[idx]]['Description'].values[0], prob) for idx, prob in zip(top_indices, top_probs)]
    return predictions

# Example inference using a query
query = "Chief complaint of cough and Impression of upper respiratory infection"
top_predictions = predict_icd10(query, model, tokenizer)

# Print results
for code, description, prob in top_predictions:
    print(f"{code}: {description}, prediction accuracy: {prob:.2%}")
