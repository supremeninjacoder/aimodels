import pandas as pd
import re

# Load ICD-10 descriptions from an Excel file
df = pd.read_excel('icd10_descriptions.xlsx')

# Function to clean descriptions
def clean_description(description, icd_code):
    icd_code_no_dot = icd_code.replace('.', '')
    pattern = r'^' + re.escape(icd_code_no_dot) + r'\b\s*'
    cleaned_description = re.sub(pattern, '', description).strip()
    return cleaned_description

# Clean descriptions
df['Cleaned_Description'] = df.apply(lambda row: clean_description(row['Description'], row['ICD-10 Code']), axis=1)

# Save cleaned descriptions to a new CSV file
df.to_csv('cleaned_icd10_descriptions.csv', index=False)

# Extract ICD-10 codes and cleaned descriptions
icd_codes = df['ICD-10 Code'].tolist()
descriptions = df['Cleaned_Description'].tolist()

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Load pre-trained ClinicalBERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(icd_codes))

# Tokenize the data
inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt", max_length=128)
labels = torch.tensor([icd_codes.index(code) for code in icd_codes])

# Create a dataset and split into train and validation sets
dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    save_steps=500
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_clinicalbert')
tokenizer.save_pretrained('./fine_tuned_clinicalbert')

from transformers import BertModel
import torch

# Load the fine-tuned model and tokenizer
model = BertModel.from_pretrained('./fine_tuned_clinicalbert')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_clinicalbert')

# Function to get embeddings
def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Generate embeddings for all descriptions
description_embeddings = []
for description in descriptions:
    embedding = get_embeddings(description, model, tokenizer)
    description_embeddings.append(embedding)

description_embeddings = torch.cat(description_embeddings)

# Function to find the nearest neighbor
def find_nearest_neighbor(query, description_embeddings, icd_codes, model, tokenizer):
    query_embedding = get_embeddings(query, model, tokenizer)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, description_embeddings)
    top_k_indices = similarities.topk(5).indices
    top_k_codes = [icd_codes[idx] for idx in top_k_indices]
    return top_k_codes

# Example usage with a new patient file impression or chief complaint
query = "Patient presents with severe abdominal pain and dehydration."
top_5_icd_codes = find_nearest_neighbor(query, description_embeddings, icd_codes, model, tokenizer)
print("Top 5 ICD-10 Codes:", top_5_icd_codes)

Explanation
Data Preparation: Clean the descriptions by removing the ICD codes from the beginning if present.
Fine-Tuning ClinicalBERT: Train the model on the cleaned descriptions and corresponding ICD-10 codes.
Generating Embeddings and Matching: Use the fine-tuned model to generate embeddings for the descriptions. Then, for a given query, compute its embedding and find the top 5 most similar description embeddings using cosine similarity.
    
# Save the model and tokenizer
model.save_pretrained('./fine_tuned_clinicalbert')
tokenizer.save_pretrained('./fine_tuned_clinicalbert')

# Load the model and tokenizer
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('./fine_tuned_clinicalbert')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_clinicalbert')
