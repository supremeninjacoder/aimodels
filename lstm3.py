import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re

# Data Preparation
df = pd.read_excel('icd10_descriptions.xlsx')


def clean_description(description, icd_code):
    icd_code_no_dot = icd_code.replace('.', '')
    pattern = r'^' + re.escape(icd_code_no_dot) + r'\b\s*'
    cleaned_description = re.sub(pattern, '', description).strip()
    return cleaned_description


df['Cleaned_Description'] = df.apply(lambda row: clean_description(row['Description'], row['ICD-10 Code']), axis=1)
df.to_csv('cleaned_icd10_descriptions.csv', index=False)

icd_codes = df['ICD-10 Code'].tolist()
descriptions = df['Cleaned_Description'].tolist()

# Tokenization and Encoding
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

max_length = 128  # Reduce the max length if needed
inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
labels = torch.tensor([icd_codes.index(code) for code in icd_codes])


class ICD10Dataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item


dataset = ICD10Dataset(inputs, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Reduce batch size if needed
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1]
        output = self.fc(hidden)
        return output


embedding_dim = 128  # Adjust embedding dimension
hidden_dim = 64  # Adjust hidden dimension
output_dim = len(set(icd_codes))
vocab_size = tokenizer.vocab_size

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=3):
    model.train()

    for epoch in range(epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_dataloader)}')
        model.train()


train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=3)

# Save Model
model_path = './lstm_icd10_classifier.pt'
torch.save(model.state_dict(), model_path)

# Load Model
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Function to get embeddings
def get_embeddings(text, model, tokenizer, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items() if key != 'token_type_ids'}
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    return outputs

# Generate embeddings for descriptions
description_embeddings = []
for description in descriptions:
    embedding = get_embeddings(description, model, tokenizer)
    description_embeddings.append(embedding)

description_embeddings = torch.cat(description_embeddings)

# Function to find nearest neighbor
def find_nearest_neighbor(query, description_embeddings, icd_codes, model, tokenizer):
    query_embedding = get_embeddings(query, model, tokenizer)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, description_embeddings)
    top_k_indices = similarities.topk(5).indices
    top_k_codes = [icd_codes[idx] for idx in top_k_indices]
    return top_k_codes

# Example query
query = "Patient presents with severe abdominal pain and dehydration."
top_5_icd_codes = find_nearest_neighbor(query, description_embeddings, icd_codes, model, tokenizer)
print("Top 5 ICD-10 Codes:", top_5_icd_codes)
