!pip install PyPDF2

import pandas as pd
import PyPDF2
# Load the Excel file
icd_data = pd.read_excel('ICDMaster.xlsx')

# Display the first few rows of the data
icd_data.head()



def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Extract text from the provided PDF files
patient_a_text = extract_text_from_pdf('PATIENTA.pdf')
patient_b_text = extract_text_from_pdf('PATIENTB.pdf')

print("Patient A Text:", patient_a_text[:1000])  # Print first 1000 characters
print("Patient B Text:", patient_b_text[:1000])  # Print first 1000 characters

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import AdamW
from tqdm import tqdm

# Check the number of unique ICD-10 codes
num_labels = len(icd_data['ICD'].unique())
print(f"Number of unique ICD-10 codes: {num_labels}")
# Prepare the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

class IcdDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_len):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        description = str(self.descriptions[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'description_text': description,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Encode labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform([[code] for code in icd_data['ICD']])

# Create dataset and dataloader
dataset = IcdDataset(
    descriptions=icd_data['Description'].tolist(),
    labels=encoded_labels,
    tokenizer=tokenizer,
    max_len=128
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

patient_a_text = 'Cough'
patient_b_text = 'Cough'


def predict_icd10(text, model, tokenizer, mlb, top_k=5):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    top_indices = probabilities.argsort()[-top_k:][::-1]
    top_labels = mlb.classes_[top_indices]
    top_descriptions = icd_data[icd_data['ICD'].isin(top_labels)]['Description'].values

    return list(zip(top_labels, top_descriptions))


# Predict for Patient A
patient_a_icd10_predictions = predict_icd10(patient_a_text, model, tokenizer, mlb)
print("Patient A ICD-10 Predictions:", patient_a_icd10_predictions)

# Predict for Patient B
patient_b_icd10_predictions = predict_icd10(patient_b_text, model, tokenizer, mlb)
print("Patient B ICD-10 Predictions:", patient_b_icd10_predictions)
