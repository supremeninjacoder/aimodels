import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import nlpaug.augmenter.word as naw
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pandas as pd
import pickle

nltk.download('wordnet')
nltk.download('omw-1.4')
# Assume we have a dataframe with ICD codes and descriptions
df = pd.read_excel('/kaggle/input/icdmaster/ICDMaster.xlsx', usecols=["ICD", "Description"])

icd_codes = df['ICD'].tolist()
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
combined_df = pd.DataFrame({'Description': augmented_descriptions, 'ICD': labels})
combined_df.to_csv('/kaggle/working/augmented.csv', index=False)
# Augment the descriptions
# augmented_descriptions = df['Description'].apply(augment_text)
# augmented_df = pd.DataFrame({'Description': augmented_descriptions, 'ICD': df['ICD']})

import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import nlpaug.augmenter.word as naw
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pandas as pd
import pickle

combined_df = pd.read_csv('/kaggle/input/augmenteddata/augmented_dataset.csv')
# Prepare dataset
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(combined_df['ICD'])

num_labels = len(label_encoder.classes_)  # Define the number of unique labels

data = {"text": combined_df['Description'].tolist(), "label": encoded_labels}
dataset = Dataset.from_dict(data)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModelForSequenceClassification.from_pretrained("medicalai/ClinicalBERT", num_labels=num_labels)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split the dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.15)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Set the model to training mode
model.train()
# Enable quantization-aware training
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_fp32_prepared = torch.quantization.prepare_qat(model)

# Training Arguments
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
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500
)

# Trainer
trainer = Trainer(
    model=model_fp32_prepared,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model with QAT
trainer.train()

# Apply quantization after training
model_int8 = torch.quantization.convert(model_fp32_prepared)
model_int8.save_pretrained("path_to_quantized_model")

# Save tokenizer and label encoder
tokenizer.save_pretrained("path_to_quantized_model")
with open("path_to_quantized_model/label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/working/quantizedout")
model.to('cuda')

# Load tokenizer and label encoder
tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/quantizedout")
with open("/kaggle/working/quantizedout/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

# Generate embeddings and perform cosine similarity calculation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_embeddings(model, tokenizer, texts):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = encoding['input_ids'].to('cuda')
            attention_mask = encoding['attention_mask'].to('cuda')
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings.append(outputs.logits.cpu().numpy())
    return np.vstack(embeddings)

# Remove duplicate ICD-10 codes, keeping the first occurrence
icd_data = df.drop_duplicates(subset="ICD", keep="first")

# Extract texts and labels
icd10_descriptions = icd_data['Description'].tolist()
icd10_codes = icd_data['ICD'].tolist()

# Sample data (Replace with your data)
patient_notes = ["Acute cough", "Acute upper respiratory infection"]

# Generate embeddings
patient_embeddings = generate_embeddings(model, tokenizer, patient_notes)
icd10_embeddings = generate_embeddings(model, tokenizer, icd10_descriptions)

# Compute cosine similarity
similarity_matrix = cosine_similarity(patient_embeddings, icd10_embeddings)

# Find the best match and corresponding ICD-10 code
best_matches = np.argmax(similarity_matrix, axis=1)
for i, match in enumerate(best_matches):
    print(f"Patient note: {patient_notes[i]}")
    print(f"Best matching ICD-10 description: {icd10_descriptions[match]}")
    print(f"ICD-10 code: {icd10_codes[match]}")


To further speed up training and adjust the learning rate, we can utilize the following approaches:

1. **Learning Rate Scheduling**: Implement a learning rate scheduler to adjust the learning rate dynamically during training. This can help converge faster to an optimal solution.
2. **Gradient Accumulation**: Use gradient accumulation to simulate larger batch sizes without increasing memory usage. This can allow training with larger effective batch sizes and potentially faster convergence.
3. **Mixed Precision Training**: Utilize mixed precision training, where some parts of the model are calculated in lower precision (e.g., half-precision floating-point format) to speed up computations while maintaining accuracy.
4. **Parallelism Optimization**: Ensure that your model and data processing are optimized for multi-GPU training. This includes efficient data loading, distributed training strategies, and model parallelism if applicable.

Here's how you can implement these approaches:

```python
from transformers import TrainingArguments, Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

# Define your TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=1,
)

# Define a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3,
    verbose=True,
    threshold=0.0001,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0,
    eps=1e-08
)

# Mixed Precision Training setup
scaler = torch.cuda.amp.GradScaler()

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    fp16=True,  # Enable mixed precision training
)

# Train the model
trainer.train()
```

In this setup:
- We define a `ReduceLROnPlateau` scheduler to reduce the learning rate dynamically based on a validation metric (e.g., loss).
- We enable mixed precision training by setting `fp16=True` and use the `GradScaler` from PyTorch's Automatic Mixed Precision (AMP) library to handle automatic scaling of gradients.
- We accumulate gradients over 4 steps (`gradient_accumulation_steps=4`) to simulate larger batch sizes.
- Ensure that your model and optimizer are compatible with mixed precision training (`model.to(dtype=torch.float16)` and `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)`).

These adjustments should significantly speed up the training process while maintaining or even improving the model's performance. Adjust the parameters as needed based on your specific requirements and available computational resources.