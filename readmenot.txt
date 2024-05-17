To modify the provided code for your use case of predicting top-5 ICD-10 codes from the patient file descriptions, we need to make several adjustments. These include loading the ICD-10 codes and descriptions from the Excel file, preparing the dataset for a single-label classification problem, and updating the inference process to output the top-5 predictions with their corresponding descriptions and prediction accuracies.

Here’s how you can adjust the code:

1. **Loading and Preparing Data:**

   - Load the ICD-10 data from the Excel file.
   - Prepare the dataset to match descriptions to ICD-10 codes.

2. **Modifying the Model and Training Code:**

   - Update the code to handle a single-label classification problem.
   - Train the model using ICD-10 codes and their descriptions.

3. **Inference Code to Output Top-5 Predictions:**

   - Adjust the inference code to output the top-5 ICD-10 codes and their descriptions along with prediction accuracies.

Here's the updated code:

```python
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import openpyxl

# Load ICD-10 data from Excel file
icd_data = pd.read_excel('/mnt/data/ICDMaster.xlsx', usecols=["Code", "Description"])

# Extract texts and labels
texts = icd_data['Description'].tolist()
codes = icd_data['Code'].tolist()

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
    num_train_epochs=5,
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
    predictions = [(id_to_code[idx], icd_data[icd_data['Code'] == id_to_code[idx]]['Description'].values[0], prob) for idx, prob in zip(top_indices, top_probs)]
    return predictions

# Example inference
patient_text = "Chief complaint of cough and Impression of upper respiratory infection"
top_predictions = predict_icd10(patient_text, model, tokenizer)

# Print results
for code, description, prob in top_predictions:
    print(f"{code}: {description}, prediction accuracy: {prob:.2%}")
```

### Explanation:

1. **Loading and Preparing Data:**
   - The ICD-10 data is loaded from the Excel file into a DataFrame.
   - Each ICD-10 code is encoded as an integer.

2. **Model and Training:**
   - The model is initialized with the number of labels equal to the number of unique ICD-10 codes.
   - The custom dataset class is used to handle tokenization and preparation of data for training.
   - The `Trainer` class from Hugging Face's `transformers` library is used to train the model.

3. **Inference:**
   - The `predict_icd10` function takes a text input, tokenizes it, and predicts the top-5 ICD-10 codes.
   - The results include the ICD-10 code, its description, and the prediction accuracy.

You can replace the `patient_text` variable with text extracted from the PDF for actual predictions. The provided code will give you a framework to achieve your specific use case.











To improve the accuracy of your model and ensure it provides relevant predictions, several factors need to be considered and adjustments made. Here's a step-by-step approach to enhance the model:

1. **Data Preprocessing and Balance:**
   - Ensure that the data is balanced and cleaned. Imbalanced datasets can lead to poor model performance.

2. **Model Architecture and Training:**
   - Fine-tuning a pre-trained model like DistilBERT might require more epochs, different learning rates, and possibly data augmentation.

3. **Evaluation Metrics:**
   - Use appropriate evaluation metrics to monitor performance during training.

4. **Inference Process:**
   - Ensure the inference process is properly set up to extract meaningful predictions.

Let's break this down into specific changes:

### Step 1: Data Preprocessing and Balance
Make sure that the ICD-10 code descriptions are well-prepared and the dataset is balanced. Check if there are any very rare codes and handle them accordingly.

### Step 2: Modify the Training and Model Code
Let's enhance the training process by tuning hyperparameters and using a more sophisticated model if necessary.

### Step 3: Ensure Proper Inference
Make sure the inference process uses softmax for multi-class classification and properly handles the top-5 predictions.

Here's the updated code:

```python
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import openpyxl

# Load ICD-10 data from Excel file
icd_data = pd.read_excel('/mnt/data/ICDMaster.xlsx', usecols=["Code", "Description"])

# Extract texts and labels
texts = icd_data['Description'].tolist()
codes = icd_data['Code'].tolist()

# Encoding ICD-10 codes as integers
code_to_id = {code: idx for idx, code in enumerate(set(codes))}
id_to_code = {idx: code for code, idx in code_to_id.items()}
labels = [code_to_id[code] for code in codes]

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

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
    predictions = [(id_to_code[idx], icd_data[icd_data['Code'] == id_to_code[idx]]['Description'].values[0], prob) for idx, prob in zip(top_indices, top_probs)]
    return predictions

# Example inference
patient_text = "Chief complaint of cough and Impression of upper respiratory infection"
top_predictions = predict_icd10(patient_text, model, tokenizer)

# Print results
for code, description, prob in top_predictions:
    print(f"{code}: {description}, prediction accuracy: {prob:.2%}")
```

### Explanation of Changes:
1. **Data Splitting with Stratification:**
   - `stratify=labels` ensures that each split has the same distribution of labels, which helps in handling imbalanced data.

2. **Training Parameters:**
   - Increased the number of epochs to 10 to give the model more time to learn.
   - Adjusted the learning rate for better optimization.
   - Added `evaluation_strategy="epoch"` to evaluate the model after every epoch.

3. **Softmax for Multi-class Classification:**
   - Used `softmax` instead of `sigmoid` for multi-class classification, which is more appropriate for your use case.

### Further Suggestions:
- **Data Augmentation:**
  - You may consider data augmentation techniques to generate more training samples if the dataset is imbalanced.
  
- **Hyperparameter Tuning:**
  - Experiment with different hyperparameters like learning rate, batch size, and number of epochs.

- **Advanced Models:**
  - If DistilBERT does not provide satisfactory results, consider using more advanced models like BERT or RoBERTa.

These changes should improve the model's ability to make accurate predictions for the given queries.













To include classes with only one instance in the dataset while still handling the `ValueError` raised by stratified splitting, you can avoid using stratification and use a simple random split instead. Here's how you can modify the code:

### Step 1: Prepare the Dataset

You will not filter out any codes but will handle the splitting differently.

```python
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import fitz  # PyMuPDF
import numpy as np

# Load ICD-10 data from Excel file
icd_data = pd.read_excel('/mnt/data/ICDMaster.xlsx', usecols=["Code", "Description"])

# Extract texts and labels
texts = icd_data['Description'].tolist()
codes = icd_data['Code'].tolist()

# Encoding ICD-10 codes as integers
code_to_id = {code: idx for idx, code in enumerate(set(codes))}
id_to_code = {idx: code for code, idx in code_to_id.items()}
labels = [code_to_id[code] for code in codes]

# Split the data without stratification
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
    predictions = [(id_to_code[idx], icd_data[icd_data['Code'] == id_to_code[idx]]['Description'].values[0], prob) for idx, prob in zip(top_indices, top_probs)]
    return predictions

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Extract Chief Complaint and Impressions
def extract_sections(text):
    chief_complaint = ""
    impressions = ""
    
    lines = text.split('\n')
    in_chief_complaint = False
    in_impressions = False
    
    for line in lines:
        line = line.strip()
        
        if "Chief Complaint" in line:
            in_chief_complaint = True
            in_impressions = False
            chief_complaint = line.replace("Chief Complaint:", "").strip()
        elif "Impression" in line:
            in_impressions = True
            in_chief_complaint = False
            impressions = line.replace("Impression:", "").strip()
        elif in_chief_complaint:
            chief_complaint += " " + line.strip()
        elif in_impressions:
            impressions += " " + line.strip()
    
    return chief_complaint, impressions

# Extract text from the PDF
pdf_path = '/mnt/data/TEST, PATIENT A.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# Extract Chief Complaint and Impressions
chief_complaint, impressions = extract_sections(pdf_text)

# Concatenate Chief Complaint and Impressions to form the query
query = f"Chief complaint of {chief_complaint} and Impression of {impressions}"

# Example inference using the extracted query
top_predictions = predict_icd10(query, model, tokenizer)

# Print results
for code, description, prob in top_predictions:
    print(f"{code}: {description}, prediction accuracy: {prob:.2%}")
```

### Explanation:

1. **Dataset Preparation:**
   - The dataset is prepared without filtering out any classes.
   - The data is split into training and validation sets without stratification, using a simple random split.

2. **Custom Dataset Class:**
   - The custom dataset class processes the text data and prepares it for the model.

3. **Training Arguments and Trainer:**
   - The training arguments and trainer are defined to train the model on the dataset.

4. **PDF Extraction:**
   - The `extract_text_from_pdf` function extracts text from the PDF file.
   - The `extract_sections` function extracts the chief complaint and impressions from the extracted text.

5. **Prediction:**
   - The `predict_icd10` function takes a text input, processes it, and predicts the top-5 ICD-10 codes using the trained model.
   - The query is formed using the extracted chief complaint and impressions and passed to the model for prediction.

By running this code, you should be able to handle the classes with only one instance and get predictions for the given query.














To handle duplicate ICD-10 code values in the Excel file, you can add a data preprocessing step to check for and remove duplicate entries based on the ICD-10 code. Here's how you can modify the code to include this step:

### Step 1: Preprocess the Data

Add a step to remove duplicate ICD-10 codes from the Excel file data.

```python
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import fitz  # PyMuPDF
import numpy as np
from sklearn.model_selection import train_test_split

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

# Split the data without stratification
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
    predictions = [(id_to_code[idx], icd_data[icd_data['Code'] == id_to_code[idx]]['Description'].values[0], prob) for idx, prob in zip(top_indices, top_probs)]
    return predictions

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Extract Chief Complaint and Impressions
def extract_sections(text):
    chief_complaint = ""
    impressions = ""
    
    lines = text.split('\n')
    in_chief_complaint = False
    in_impressions = False
    
    for line in lines:
        line = line.strip()
        
        if "Chief Complaint" in line:
            in_chief_complaint = True
            in_impressions = False
            chief_complaint = line.replace("Chief Complaint:", "").strip()
        elif "Impression" in line:
            in_impressions = True
            in_chief_complaint = False
            impressions = line.replace("Impression:", "").strip()
        elif in_chief_complaint:
            chief_complaint += " " + line.strip()
        elif in_impressions:
            impressions += " " + line.strip()
    
    return chief_complaint, impressions

# Extract text from the PDF
pdf_path = '/mnt/data/TEST, PATIENT A.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# Extract Chief Complaint and Impressions
chief_complaint, impressions = extract_sections(pdf_text)

# Concatenate Chief Complaint and Impressions to form the query
query = f"Chief complaint of {chief_complaint} and Impression of {impressions}"

# Example inference using the extracted query
top_predictions = predict_icd10(query, model, tokenizer)

# Print results
for code, description, prob in top_predictions:
    print(f"{code}: {description}, prediction accuracy: {prob:.2%}")
```

### Explanation:

1. **Removing Duplicates:**
   - The line `icd_data = icd_data.drop_duplicates(subset="Code", keep="first")` removes duplicate ICD-10 codes, keeping only the first occurrence of each code.

2. **Preprocessing:**
   - After removing duplicates, the code proceeds with extracting texts and labels, encoding the ICD-10 codes, and splitting the data.

3. **Handling the Dataset:**
   - The custom dataset class and the training procedure remain the same.

4. **PDF Extraction and Prediction:**
   - The functions for extracting text from the PDF, extracting specific sections, and predicting the top-5 ICD-10 codes are included and used as before.

By running this modified code, you should be able to preprocess your data to remove duplicates and handle the entire dataset, including classes with only one instance.



















To modify the code such that chief complaint and impressions along with ICD-10 codes assigned to each document present in the comments section of multiple PDF files are extracted and then used for further fine-tuning the model, you can follow these steps:

1. **Extract text, chief complaint, impressions, and ICD-10 codes from multiple PDF files.**
2. **Prepare the data for training by combining the new data with the existing ICD-10 data.**
3. **Fine-tune the model on the combined dataset.**

Here's the modified code:

### Step 1: Extract Data from Multiple PDF Files

First, let's define functions to extract the required sections and ICD-10 codes from the comments section of multiple PDF files.

```python
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
```

### Step 2: Combine Data and Prepare for Training

Next, combine the new data with the existing ICD-10 data and prepare the dataset for training.

```python
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
```

### Explanation:

1. **Data Extraction from Multiple PDFs:**
   - The `extract_data_from_pdfs` function extracts chief complaint, impressions, and ICD-10 codes from the comments section of multiple PDF files.
   - This data is then stored in a list `pdf_data`.

2. **Combining Data:**
   - The new data extracted from PDF files is combined with the existing ICD-10 data.
   - The combined dataset is then split into training and validation sets.

3. **Model Fine-Tuning:**
   - The model is fine-tuned on the combined dataset using the `Trainer` class from Hugging Face's Transformers library.

4. **Prediction:**
   - The `predict_icd10` function predicts the top-5 ICD-10 codes for a given query using the fine-tuned model.
   - An example query is used to demonstrate the prediction process.

By running this code, you should be able to extract data from multiple PDF files, combine it with the existing dataset, and fine-tune the model to improve its performance on the new data.