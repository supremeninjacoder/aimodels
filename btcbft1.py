import pandas as pd
import re

# Load ICD-10 descriptions from an Excel file
df = pd.read_excel('ICDMaster.xlsx', usecols=["ICD", "Description"])
icd_codes = df['ICD'].tolist()
descriptions = df['Description'].tolist()

# Function to clean descriptions
def clean_description(description, icd_code):
    # Remove dot from ICD code for pattern matching
    icd_code_no_dot = icd_code.replace('.', '')
    pattern = r'^' + re.escape(icd_code_no_dot) + r'\b\s*'
    cleaned_description = re.sub(pattern, '', description).strip()
    return cleaned_description

# Remove ICD-10 code from the beginning of each description if present
cleaned_descriptions = []
for desc, icd_code in zip(descriptions, icd_codes):
    cleaned_description = clean_description(desc, icd_code)
    cleaned_descriptions.append(cleaned_description)

# Update the DataFrame with cleaned descriptions
df['Description'] = cleaned_descriptions

# Save the cleaned descriptions to a new CSV file
df.to_csv('cleaned_icd10_descriptions.csv', index=False)

from transformers import MarianMTModel, MarianTokenizer

# Initialize the MarianMT models for back-translation (English to French and back)
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate text
def translate(text, model, tokenizer):
    batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    gen = model.generate(**batch)
    translated = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return translated

# Generate variations using back-translation
augmented_descriptions = []
labels = []

for desc, label in zip(cleaned_descriptions, icd_codes):
    augmented_descriptions.append(desc)
    labels.append(label)
    for _ in range(5):  # Generate 20 variations per description
        # Translate to French and back to English
        translated_desc = translate(translate(desc, model, tokenizer), model, tokenizer)
        augmented_descriptions.append(translated_desc)
        labels.append(label)

# Create a DataFrame for the augmented data
augmented_df = pd.DataFrame({
    'Description': augmented_descriptions,
    'ICD-10 Code': labels
})

# Save to a new CSV file
augmented_df.to_csv('augmented_icd10_descriptions.csv', index=False)

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Get embeddings for the descriptions
descriptions = augmented_df['Description'].tolist()
embeddings = get_embeddings(descriptions)

import torch.nn as nn
import torch.optim as optim


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out


# Initialize the model, loss function, and optimizer
input_dim = embeddings.size(1)
hidden_dim = 128
output_dim = len(set(icd_codes))

model = LSTMClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import DataLoader, TensorDataset

# Convert labels to indices
label_to_idx = {label: idx for idx, label in enumerate(set(labels))}
labels_idx = [label_to_idx[label] for label in labels]

# Create DataLoader
dataset = TensorDataset(embeddings, torch.tensor(labels_idx))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(3):  # Reduced number of epochs for faster training
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'lstm_icd10_model.pth')

# Load the model
model.load_state_dict(torch.load('lstm_icd10_model.pth'))
model.eval()

def classify_text(text):
    with torch.no_grad():
        embedding = get_embeddings([text])
        output = model(embedding)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_label = list(label_to_idx.keys())[predicted_idx]
        return predicted_label

# Example usage
new_text = "Patient complains of severe headache and nausea."
predicted_icd10_code = classify_text(new_text)
print(f'Predicted ICD-10 Code: {predicted_icd10_code}')

Explanation
Data Cleaning: We strip any ICD-10 codes from the beginning of the descriptions.
Data Augmentation: We use back-translation to generate multiple variations of each description to augment our dataset.
Embedding Generation: We use a pre-trained BERT model to convert text descriptions into embeddings.
LSTM Model: We build and train an LSTM model to classify these embeddings into ICD-10 codes.
Inference: We use the trained model to classify new patient texts and find the most similar ICD-10 codes.
This approach leverages data augmentation and pre-trained embeddings to improve the model's performance, making it more robust to the limited training data.