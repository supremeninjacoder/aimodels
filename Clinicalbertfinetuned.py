!pip install nlpaug pandas transformers sklearn torch

import pandas as pd
import nlpaug.augmenter.word as naw
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_excel('path/to/your/excel_file.xlsx')

icd_codes = df['ICD10_Code'].tolist()
icd_descriptions = df['Description'].tolist()
icd_descriptions = [desc.lower() for desc in icd_descriptions]

# Augment the data
aug = naw.SynonymAug(aug_src='wordnet')
augmented_descriptions = []
labels = []

for desc, label in zip(icd_descriptions, icd_codes):
    for _ in range(5):  # Generate 5 augmentations per description
        augmented_desc = aug.augment(desc)
        augmented_descriptions.append(augmented_desc)
        labels.append(label)

# Create a DataFrame with augmented data
augmented_df = pd.DataFrame({'Description': augmented_descriptions, 'ICD10_Code': labels})
augmented_df.to_csv('augmented_dataset.csv', index=False)


# Reload augmented dataset
df_aug = pd.read_csv('augmented_dataset.csv')

train_desc, val_desc, train_labels, val_labels = train_test_split(df_aug['Description'], df_aug['ICD10_Code'], test_size=0.2, stratify=df_aug['ICD10_Code'])


from torch.utils.data import Dataset

class ICD10Dataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_length):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(description, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'label': torch.tensor(label)}

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(icd_codes)))

# Prepare datasets
train_dataset = ICD10Dataset(train_desc.tolist(), train_labels.tolist(), tokenizer, max_length=128)
val_dataset = ICD10Dataset(val_desc.tolist(), val_labels.tolist(), tokenizer, max_length=128)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_clinicalbert')
tokenizer.save_pretrained('./fine_tuned_clinicalbert')


import torch.nn as nn
from transformers import AutoModel

class SiameseNetwork(nn.Module):
    def __init__(self, model_name):
        super(SiameseNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, 128)

    def forward_one(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

    def forward(self, input1, input2):
        output1 = self.forward_one(**input1)
        output2 = self.forward_one(**input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Create pairs of data for Siamese Network training
def create_pairs(descriptions, labels):
    pairs = []
    labels = []
    for i in range(len(descriptions)):
        for j in range(len(descriptions)):
            pairs.append([descriptions[i], descriptions[j]])
            labels.append(1 if labels[i] == labels[j] else 0)
    return pairs, labels

train_pairs, train_pair_labels = create_pairs(train_desc.tolist(), train_labels.tolist())
val_pairs, val_pair_labels = create_pairs(val_desc.tolist(), val_labels.tolist())

class SiameseDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer, max_length):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        encoding1 = self.tokenizer(pair[0], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding2 = self.tokenizer(pair[1], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids1': encoding1['input_ids'].squeeze(), 'attention_mask1': encoding1['attention_mask'].squeeze(),
                'input_ids2': encoding2['input_ids'].squeeze(), 'attention_mask2': encoding2['attention_mask'].squeeze(), 'label': torch.tensor(label)}

train_dataset = SiameseDataset(train_pairs, train_pair_labels, tokenizer, max_length=128)
val_dataset = SiameseDataset(val_pairs, val_pair_labels, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = SiameseNetwork(model_name)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids1 = batch['input_ids1']
        attention_mask1 = batch['attention_mask1']
        input_ids2 = batch['input_ids2']
        attention_mask2 = batch['attention_mask2']
        labels = batch['label']

        optimizer.zero_grad()

        output1, output2 = model({'input_ids': input_ids1, 'attention_mask': attention_mask1},
                                 {'input_ids': input_ids2, 'attention_mask': attention_mask2})
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Save the model
model_path = './fine_tuned_siamese_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

class PrototypicalNetwork(nn.Module):
    def __init__(self, model_name):
        super(PrototypicalNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def train_prototypical_network(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            protos = outputs.mean(0)
            dists = euclidean_dist(outputs, protos)
            loss = torch.mean((labels * dists).sum())
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Prepare datasets
train_dataset = ICD10Dataset(train_desc.tolist(), train_labels.tolist(), tokenizer, max_length=128)
val_dataset = ICD10Dataset(val_desc.tolist(), val_labels.tolist(), tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = PrototypicalNetwork(model_name)

# Training Loop
optimizer = optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 3

train_prototypical_network(model, train_loader, optimizer, num_epochs)

# Save the model
model_path = './fine_tuned_prototypical_model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)



Changes:
TypeError: new(): invalid
data
type
'str'
Gemini
Hypothesis:
The
error
occurs in the
__getitem__
method
of
the
ICD10Dataset


class . Specifically, the line label = self.labels[idx] is causing the error.The error message TypeError: new(): invalid


data
type
'str'
suggests
that
the
labels
list
contains
strings,
while the torch.tensor() function expects a numeric input.

Suggested
Changes:
Convert
the
labels
to
integers:

from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Fit the LabelEncoder on the unique ICD codes
le.fit(set(icd_codes))

# Transform the labels using the LabelEncoder
train_labels = le.transform(train_labels)
val_labels = le.transform(val_labels)
Use
code
with caution
    Modify
    the
    __getitem__
    method
    to
    handle
    the
    integer
    labels:


def __getitem__(self, idx):
    description = self.descriptions[idx]
    label = self.labels[idx]
    encoding = self.tokenizer(description, max_length=self.max_length, padding='max_length', truncation=True,
                              return_tensors='pt')
    return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)}


Use
code
with caution
    Rerun
    the
    training
    script.
These
changes
should
resolve
the
TypeError and allow
the
training
script
to
run
successfully.

