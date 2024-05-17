!pip install accelerate -U


import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import openpyxl

# Load ICD-10 data from Excel file
icd_data = pd.read_excel('ICDMaster.xlsx', usecols=["ICD", "Description"])

# Extract texts and labels
texts = icd_data['Description'].tolist()
codes = icd_data['ICD'].tolist()

# Encoding ICD-10 codes as integers
code_to_id = {code: idx for idx, code in enumerate(set(codes))}
id_to_code = {idx: code for code, idx in code_to_id.items()}
labels = [code_to_id[code] for code in codes]

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir='./results',
    num_train_epochs=1,
    save_steps=1000,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
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
    predictions = [(id_to_code[idx], icd_data[icd_data['ICD'] == id_to_code[idx]]['Description'].values[0], prob) for idx, prob in zip(top_indices, top_probs)]
    return predictions

# Example inference
patient_text = "Chief complaint of cough and Impression of upper respiratory infection"
top_predictions = predict_icd10(patient_text, model, tokenizer)

# Print results
for code, description, prob in top_predictions:
    print(f"{code}: {description}, prediction accuracy: {prob:.2%}")
