!pip install -q transformers
!pip install -q datasets
!pip install -q torch
!pip install -q scikit-learn
!pip install -q sentence-transformers

import torch
import pandas as pd
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from datasets import Dataset as HFDataset, load_metric


# Load your dataset
data = pd.read_csv('path_to_your_data.csv')

# Basic preprocessing function
def preprocess_text(text):
    return text.lower().strip()

data['description'] = data['description'].apply(preprocess_text)

# Data augmentation (simple example)
def augment_text(text):
    # You can use various methods like synonyms replacement, random insertion, etc.
    return text

data['augmented_description'] = data['description'].apply(augment_text)

# Combine original and augmented data
augmented_data = pd.concat([data[['description', 'label']], data[['augmented_description', 'label']].rename(columns={'augmented_description': 'description'})])

# Label encoding
label_encoder = LabelEncoder()
augmented_data['label'] = label_encoder.fit_transform(augmented_data['label'])


tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

class ICD10Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.descriptions = data['description'].tolist()
        self.labels = data['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(description, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'label': torch.tensor(label)}

train_dataset = ICD10Dataset(augmented_data, tokenizer, max_length=128)


model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))
model.to('cuda')

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
trainer.save_model("./fine_tuned_model")


# Siamese network for few-shot learning
class SiameseICD10Dataset(Dataset):
    def __init__(self, descriptions1, descriptions2, labels, tokenizer, max_length):
        self.descriptions1 = descriptions1
        self.descriptions2 = descriptions2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions1)

    def __getitem__(self, idx):
        description1 = self.descriptions1[idx]
        description2 = self.descriptions2[idx]
        label = self.labels[idx]

        encoding1 = self.tokenizer(description1, max_length=self.max_length, padding='max_length', truncation=True,
                                   return_tensors='pt')
        encoding2 = self.tokenizer(description2, max_length=self.max_length, padding='max_length', truncation=True,
                                   return_tensors='pt')

        return {'input_ids1': encoding1['input_ids'].squeeze(),
                'attention_mask1': encoding1['attention_mask'].squeeze(),
                'input_ids2': encoding2['input_ids'].squeeze(),
                'attention_mask2': encoding2['attention_mask'].squeeze(),
                'label': torch.tensor(label)}


# Assume descriptions1, descriptions2, and labels are lists of paired descriptions and their labels
siamese_train_dataset = SiameseICD10Dataset(descriptions1, descriptions2, labels, tokenizer, max_length=128)

from transformers import DistilBertModel


class SiameseBert(torch.nn.Module):
    def __init__(self):
        super(SiameseBert, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(768, 1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state[:, 0, :]
        output2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state[:, 0, :]

        combined_output = torch.abs(output1 - output2)
        combined_output = self.dropout(combined_output)
        logits = self.fc(combined_output)

        return logits


siamese_model = SiameseBert()
siamese_model.to('cuda')


from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AdamW

def train_siamese(model, train_dataset, epochs=3, batch_size=16, learning_rate=2e-5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids1 = batch['input_ids1'].to('cuda')
            attention_mask1 = batch['attention_mask1'].to('cuda')
            input_ids2 = batch['input_ids2'].to('cuda')
            attention_mask2 = batch['attention_mask2'].to('cuda')
            labels = batch['label'].to('cuda').float()

            optimizer.zero_grad()
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2).squeeze()
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

train_siamese(siamese_model, siamese_train_dataset, epochs=3, batch_size=16, learning_rate=2e-5)


# Few-shot learning
def few_shot_predict(model, tokenizer, text1, text2, max_length=128):
    model.eval()
    encoding1 = tokenizer(text1, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    encoding2 = tokenizer(text2, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

    input_ids1 = encoding1['input_ids'].to('cuda')
    attention_mask1 = encoding1['attention_mask'].to('cuda')
    input_ids2 = encoding2['input_ids'].to('cuda')
    attention_mask2 = encoding2['attention_mask'].to('cuda')

    with torch.no_grad():
        logits = model(input_ids1, attention_mask1, input_ids2, attention_mask2).squeeze()
        prob = torch.sigmoid(logits).item()

    return prob


# Example of few-shot prediction
example_text1 = "Patient has chest pain and shortness of breath."
example_text2 = "Patient complains of severe chest pain and difficulty in breathing."
probability = few_shot_predict(siamese_model, tokenizer, example_text1, example_text2)
print(f"Similarity Probability: {probability}")

