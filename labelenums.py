import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Load ICD-10 data from Excel file
icd_data = pd.read_excel('ICDMaster.xlsx', usecols=["ICD", "Description"])

# Remove duplicate ICD-10 codes, keeping the first occurrence
icd_data = icd_data.drop_duplicates(subset="ICD", keep="first")

# Extract ICD-10 codes and descriptions
codes = icd_data['ICD'].tolist()
descriptions = icd_data['Description'].tolist()

# Generate synthetic data
num_copies = 15  # Number of copies for each description
synthetic_descriptions = []
synthetic_codes = []

for desc, code in zip(descriptions, codes):
    for _ in range(num_copies):
        synthetic_descriptions.append(desc)
        synthetic_codes.append(code)

# Get unique ICD-10 codes
unique_codes = sorted(list(set(synthetic_codes)))

# Map ICD-10 codes to binary vectors
code_to_index = {code: idx for idx, code in enumerate(unique_codes)}

def create_binary_vector(codes, code_to_index):
    binary_vector = np.zeros(len(code_to_index), dtype=int)
    for code in codes:
        if code in code_to_index:
            binary_vector[code_to_index[code]] = 1
    return binary_vector

# Create binary vectors for each description
binary_labels = [create_binary_vector([code], code_to_index) for code in synthetic_codes]

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    synthetic_descriptions, binary_labels, test_size=0.2, random_state=42
)

# Initialize the tokenizer and model
checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=len(unique_codes))

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
        label = torch.tensor(self.labels[idx], dtype=torch.float)
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
    num_train_epochs=3,
    learning_rate=2e-5,
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

# Train the model
trainer.train()
trainer.evaluate()

# Function to predict ICD-10 codes
def predict_icd10(text, model, tokenizer, code_to_index, top_k=5):
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding="max_length", max_length=128)
    encoding = encoding.to(model.device)
    outputs = model(**encoding)
    logits = outputs.logits[0]
    probs = torch.sigmoid(logits)
    top_probs, top_indices = torch.topk(probs, top_k)
    top_indices = top_indices.cpu().numpy()
    top_probs = top_probs.cpu().detach().numpy()
    index_to_code = {idx: code for code, idx in code_to_index.items()}
    predictions = [(index_to_code[idx], top_probs[idx]) for idx in top_indices]
    return predictions

# Example inference using a sample query
sample_query = "Patient is experiencing a severe headache with visual disturbances."
top_predictions = predict_icd10(sample_query, model, tokenizer, code_to_index)

# Print results
for code, prob in top_predictions:
    description = icd_data[icd_data['ICD'] == code]['Description'].values[0]
    print(f"{code}: {description}, prediction probability: {prob:.2%}")
