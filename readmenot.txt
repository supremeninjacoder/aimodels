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

By running this code, you should be able to extract data from multiple PDF files, combine it with the existing dataset, and fine-tune the model to improve its performance on the new data.]]





To implement a system that reads patient files, extracts the chief complaint and impressions, and then queries both an API and a local Excel file to generate relevant ICD-10 codes, you can follow these steps:

1. **Extract Chief Complaint and Impressions from Patient Files:**
   - Use a PDF parser to extract relevant sections from patient files.

2. **Query ICD-10 Codes:**
   - Use the National Library of Medicine’s Clinical Tables API to get ICD-10 codes.
   - Use the local ICD-10 Excel file to find matches and descriptions.

3. **Integrate with an LLM (Language Model):**
   - Use an LLM to generate and validate ICD-10 codes from the extracted text and API results.

### Step-by-Step Implementation

#### Step 1: Extract Chief Complaint and Impressions from PDF

```python
import fitz  # PyMuPDF
import os
import re

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
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        
        if "Chief Complaint" in line:
            chief_complaint = re.sub(r"Chief Complaint\s*[:\-]?", "", line).strip()
        elif "Impression" in line:
            impressions = re.sub(r"Impression\s*[:\-]?", "", line).strip()
    
    return chief_complaint, impressions

def extract_data_from_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    chief_complaint, impressions = extract_sections(pdf_text)
    return chief_complaint, impressions

# Example usage
pdf_path = '/mnt/data/TEST, PATIENT A.pdf'
chief_complaint, impressions = extract_data_from_pdf(pdf_path)
print(f"Chief Complaint: {chief_complaint}")
print(f"Impressions: {impressions}")
```

#### Step 2: Query the ICD-10 API

```python
import requests

def query_icd10_api(query):
    url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
    params = {
        "sf": "code,name",
        "terms": query,
        "maxList": 5
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Example usage
query = "upper respiratory infection"
icd10_results = query_icd10_api(query)
print(icd10_results)
```

#### Step 3: Query the Local Excel File

```python
import pandas as pd

def load_icd10_excel(file_path):
    return pd.read_excel(file_path)

def find_icd10_matches(description, icd10_data):
    matches = icd10_data[icd10_data['Description'].str.contains(description, case=False, na=False)]
    return matches

# Example usage
icd10_data = load_icd10_excel('/mnt/data/ICDMaster.xlsx')
description = "upper respiratory infection"
icd10_matches = find_icd10_matches(description, icd10_data)
print(icd10_matches)
```

#### Step 4: Integrate with an LLM

For this part, we'll use the OpenAI API to generate ICD-10 codes based on the extracted text and results from the API and Excel file. Ensure you have `openai` installed and an API key set up.

```python
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

def generate_icd10_codes(chief_complaint, impressions, api_results, excel_results):
    prompt = (
        f"Chief Complaint: {chief_complaint}\n"
        f"Impressions: {impressions}\n"
        f"API Results: {api_results}\n"
        f"Excel Matches: {excel_results}\n"
        "Based on the above information, generate the most relevant ICD-10 codes:"
    )

    response = openai.Completion.create(
        engine="gpt-4",  # Or "gpt-3.5-turbo" depending on your subscription
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5
    )

    return response.choices[0].text.strip()

# Example usage
chief_complaint, impressions = extract_data_from_pdf(pdf_path)
icd10_api_results = query_icd10_api(chief_complaint + " " + impressions)
icd10_excel_matches = find_icd10_matches(chief_complaint + " " + impressions, icd10_data)
generated_icd10_codes = generate_icd10_codes(chief_complaint, impressions, icd10_api_results, icd10_excel_matches.to_dict(orient='records'))

print("Generated ICD-10 Codes:", generated_icd10_codes)
```

### Putting It All Together

You can create a function to process multiple PDF files, extract the relevant data, query the API and Excel, and generate ICD-10 codes using the LLM:

```python
def process_pdf_folder(pdf_folder, icd10_excel_path, openai_api_key):
    openai.api_key = openai_api_key
    icd10_data = load_icd10_excel(icd10_excel_path)
    results = []

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            chief_complaint, impressions = extract_data_from_pdf(pdf_path)
            if chief_complaint or impressions:
                query = f"{chief_complaint} {impressions}"
                icd10_api_results = query_icd10_api(query)
                icd10_excel_matches = find_icd10_matches(query, icd10_data)
                generated_icd10_codes = generate_icd10_codes(
                    chief_complaint,
                    impressions,
                    icd10_api_results,
                    icd10_excel_matches.to_dict(orient='records')
                )
                results.append({
                    'pdf_file': pdf_file,
                    'chief_complaint': chief_complaint,
                    'impressions': impressions,
                    'generated_icd10_codes': generated_icd10_codes
                })

    return results

# Example usage
pdf_folder = '/mnt/data/pdf_files'
icd10_excel_path = '/mnt/data/ICDMaster.xlsx'
openai_api_key = 'YOUR_OPENAI_API_KEY'
results = process_pdf_folder(pdf_folder, icd10_excel_path, openai_api_key)

for result in results:
    print(f"File: {result['pdf_file']}")
    print(f"Chief Complaint: {result['chief_complaint']}")
    print(f"Impressions: {result['impressions']}")
    print(f"Generated ICD-10 Codes: {result['generated_icd10_codes']}\n")
```

This code will read multiple PDF files, extract the chief complaint and impressions, query both the ICD-10 API and the local Excel file, and use the OpenAI GPT model to generate relevant ICD-10 codes. Ensure you have all necessary libraries installed and replace placeholders with your actual paths and API key.






Comparing the three approaches—using a search API, training a large language model (LLM), and employing an indexed-based approach like Pinecone—each has its own strengths and weaknesses. Let's explore these approaches in detail:

### 1. Search API

#### Pros:
- **Accuracy**: Given that the search API is specifically designed for mapping patient file text to ICD-10 codes, it tends to be highly accurate.
- **Reliability**: Established APIs usually have undergone rigorous testing and validation.
- **Ease of Use**: Minimal setup required; you only need to query the API.
- **Updates and Maintenance**: The service provider handles updates to the ICD-10 codes and improvements in the mapping logic.

#### Cons:
- **Dependency**: Reliance on an external service can be a limitation if the service is down, if there are API rate limits, or if the service is discontinued.
- **Cost**: Using an API might involve subscription fees or per-query charges.
- **Customization**: Limited ability to customize or adapt the mapping logic to specific needs of your organization.

### 2. Training a Large Language Model (LLM)

#### Pros:
- **Customization**: You can train the model on your specific dataset, which might include unique terminologies or specific patterns observed in your patient files.
- **Integration**: Directly integrates into your existing workflow without the need for external dependencies.
- **Continuous Improvement**: The model can continuously learn and improve with more data and feedback over time.

#### Cons:
- **Accuracy and Hallucinations**: LLMs can generate inaccurate outputs or hallucinate information that is not present in the input text.
- **Resource Intensive**: Training and maintaining an LLM requires substantial computational resources and expertise in machine learning.
- **Maintenance**: Continuous maintenance is needed to ensure the model remains up-to-date with the latest ICD-10 codes and medical knowledge.
- **Training Data Quality**: The model's accuracy is highly dependent on the quality and quantity of the training data.

### 3. Indexed-based Approach (e.g., Pinecone)

#### Pros:
- **Scalability**: Vector databases like Pinecone can handle large volumes of data and provide quick lookup times.
- **Customization**: You can tailor the indexing and embedding process to your specific dataset and requirements.
- **Contextual Search**: Using vector embeddings allows for more nuanced and context-aware searches compared to keyword-based searches.

#### Cons:
- **Complexity**: Setting up and maintaining an indexed-based approach can be complex and requires expertise in vector embeddings and database management.
- **Resource Requirements**: Like LLMs, this approach can be resource-intensive in terms of computational power for generating embeddings and indexing.
- **Accuracy**: While generally accurate, the results can be affected by the quality of the embeddings and the indexing strategy.

### Detailed Comparison:

#### Accuracy:
- **Search API**: Typically high, as it is purpose-built for the task.
- **LLM**: Can be high but prone to hallucinations and inaccuracies without continuous training and validation.
- **Indexed-based Approach**: Can be highly accurate if embeddings and indexing are well-implemented.

#### Ease of Use:
- **Search API**: Easiest to use with minimal setup.
- **LLM**: Requires expertise in machine learning and continuous maintenance.
- **Indexed-based Approach**: Requires expertise in setting up and managing vector databases.

#### Cost:
- **Search API**: Potentially costly depending on usage rates.
- **LLM**: High initial cost for training and ongoing costs for maintenance.
- **Indexed-based Approach**: Cost associated with computational resources and database management.

#### Customization:
- **Search API**: Limited customization.
- **LLM**: Highly customizable to specific needs and datasets.
- **Indexed-based Approach**: Customizable in terms of embeddings and indexing strategy.

#### Integration and Maintenance:
- **Search API**: Easy to integrate, minimal maintenance required.
- **LLM**: More complex integration and significant maintenance required.
- **Indexed-based Approach**: Moderate complexity in integration, requires ongoing maintenance.

### Conclusion:
Choosing the best approach depends on your specific needs and resources:
- If accuracy and ease of use are paramount, and cost is not a primary concern, a **Search API** is the best choice.
- If you need high customization and have the resources to invest in machine learning infrastructure, an **LLM** could be suitable despite the risk of inaccuracies.
- If you have the technical capability to manage embeddings and indexes, and seek a scalable solution that offers contextual search, the **indexed-based approach** (Pinecone) could be advantageous.

For most organizations, starting with the **Search API** for its reliability and ease of use, while evaluating the feasibility of other approaches for future improvements, would be a pragmatic strategy.






Sure! The `predict_icd10` function is designed to predict the top ICD-10 codes given a text input (e.g., a chief complaint and impression) using a pre-trained DistilBERT model. Here is a detailed explanation of each step in the function:

### Function Definition
```python
def predict_icd10(text, model, tokenizer, top_k=5):
```
- **text**: The input text that needs to be classified.
- **model**: The pre-trained DistilBERT model used for making predictions.
- **tokenizer**: The tokenizer corresponding to the DistilBERT model that converts text into tokens.
- **top_k**: The number of top predictions to return (default is 5).

### Tokenizing the Input
```python
encoding = tokenizer(text, return_tensors='pt', truncation=True, padding="max_length", max_length=128)
encoding = encoding.to(model.device)
```
- **tokenizer(text, return_tensors='pt', truncation=True, padding="max_length", max_length=128)**:
  - **text**: The input text to be tokenized.
  - **return_tensors='pt'**: Returns the tokenized data as PyTorch tensors.
  - **truncation=True**: Truncates the text if it exceeds the maximum length.
  - **padding="max_length"**: Pads the text to the maximum length specified.
  - **max_length=128**: Specifies the maximum length for padding/truncation.

- **encoding.to(model.device)**: Moves the encoding to the same device (CPU or GPU) as the model.

### Making Predictions
```python
outputs = model(**encoding)
logits = outputs.logits[0]
```
- **model(**encoding)**: Feeds the tokenized input to the model and gets the output.
- **outputs.logits[0]**: Extracts the logits (raw prediction scores) from the model's output for the first (and only) sample in the batch.

### Calculating Probabilities
```python
probs = torch.nn.functional.softmax(logits, dim=-1)
```
- **torch.nn.functional.softmax(logits, dim=-1)**: Applies the softmax function to the logits to convert them into probabilities. The softmax function ensures that the probabilities sum to 1, making it easier to interpret the output as probabilities of different classes.

### Getting Top-K Predictions
```python
top_probs, top_indices = torch.topk(probs, top_k)
```
- **torch.topk(probs, top_k)**: Retrieves the top `k` probabilities and their corresponding indices from the probability tensor.

### Preparing the Output
```python
top_indices = top_indices.cpu().numpy()
top_probs = top_probs.cpu().detach().numpy()
```
- **top_indices.cpu().numpy()**: Moves the indices tensor to the CPU and converts it to a NumPy array.
- **top_probs.cpu().detach().numpy()**: Moves the probabilities tensor to the CPU, detaches it from the computation graph (since gradients are no longer needed), and converts it to a NumPy array.

### Constructing the Predictions List
```python
predictions = [(id_to_code[idx], icd_data[icd_data['ICD'] == id_to_code[idx]]['Description'].values[0], prob) for idx, prob in zip(top_indices, top_probs)]
```
- **[(id_to_code[idx], icd_data[icd_data['ICD'] == id_to_code[idx]]['Description'].values[0], prob) for idx, prob in zip(top_indices, top_probs)]**:
  - **id_to_code[idx]**: Converts the index back to the corresponding ICD-10 code.
  - **icd_data[icd_data['ICD'] == id_to_code[idx]]['Description'].values[0]**: Retrieves the description of the ICD-10 code from the original ICD data.
  - **prob**: The probability of the ICD-10 code as predicted by the model.
  - This constructs a list of tuples where each tuple contains an ICD-10 code, its description, and the associated prediction probability.

### Returning the Predictions
```python
return predictions
```
- The function returns the list of top-k predictions, each containing the ICD-10 code, its description, and the probability.

### Example Usage
To use this function, you can pass a text input along with the pre-trained model and tokenizer. For instance:
```python
query = "Chief complaint of headache and Impression of migraine"
top_predictions = predict_icd10(query, model, tokenizer)

for code, description, prob in top_predictions:
    print(f"{code}: {description}, prediction accuracy: {prob:.2%}")
```

This will print the top predicted ICD-10 codes for the given text, along with their descriptions and prediction probabilities.


Limited Training Data: Having only one description per label limits the model's ability to learn diverse representations.
High Dimensionality: With 73,000 unique labels, the classification task is highly complex.
To improve model accuracy, you can employ the following strategies:

Data Augmentation: Create synthetic data to increase the number of training samples for each ICD-10 code.
Transfer Learning: Fine-tune a pre-trained model on your specific task.
Few-Shot Learning: Use techniques designed for scenarios with limited examples per class.
Siamese Networks: Use a different approach that focuses on learning the similarity between descriptions and patient notes.
Step 1: Data Augmentation
You can augment your data by creating synthetic variations of the descriptions. This can be done using techniques like synonym replacement, back-translation, and paraphrasing.


Step 2: Fine-Tuning with Transfer Learning
Fine-tuning a pre-trained model like ClinicalBERT on your augmented dataset can help the model learn better representations specific to your domain.



Step 3: Few-Shot Learning
Using few-shot learning techniques like prototypical networks can be beneficial. These models are designed to perform well with limited examples per class.

Step 4: Using a Siamese Network for Similarity Learning
Siamese networks are particularly useful when you need to determine whether two texts are similar. This approach might be more suitable given your requirement to match patient notes with ICD-10 descriptions.

Here’s an outline for implementing a Siamese Network with BERT:

Define a Siamese Network Architecture: Use BERT as the base model to generate embeddings for both descriptions and patient notes.
Train on Pairs of Descriptions: Create positive and negative pairs from your augmented data



This is a high-level outline, and the actual implementation would require handling data loading, batching, and other details.

Summary
Data Augmentation: Increase the number of samples per class.
Fine-Tuning: Fine-tune ClinicalBERT or a similar model on the augmented data.
Few-Shot Learning: Consider few-shot learning techniques if you have very limited examples.
Siamese Network: Use a Siamese network to learn similarity between patient notes and ICD-10 descriptions.
By following these steps, you can improve the accuracy of your model even with limited training data.


Siamese Network: Used to learn the similarity between descriptions and patient notes.
Few-Shot Learning with Prototypical Networks: Used to handle cases with limited data per class.
These implementations provide a foundation for training models to improve accuracy in matching patient notes to ICD-10 codes. Adjustments and additional tuning might be necessary based on your specific dataset and requirements.




This comprehensive implementation includes data augmentation, fine-tuning a pre-trained ClinicalBERT model, and two advanced models: Siamese Network and Prototypical Network. This approach should help in achieving better accuracy for matching patient notes to ICD-10 codes. Adjustments and further fine-tuning may be necessary based on the specific dataset and requirements.


Here’s how to adopt the provided LLaMA inference code for the problem of fine-tuning a ClinicalBERT model and implementing Siamese and Prototypical Networks for ICD-10 code prediction:

Step-by-Step Implementation
Set Up the Environment: Ensure that the required libraries are installed.
Prepare Data: Load and preprocess the dataset.
Tokenize Data: Tokenize the text descriptions.
Model Configuration: Load and configure the model, including LoRa configurations.
Inference: Run inference using a multi-threaded approach.

Explanation
Environment Setup: Installs necessary libraries and sets up device configuration.
Data Preparation: Processes and tokenizes input text descriptions.
Tokenization: Converts the text descriptions into token IDs and attention masks.
Model Configuration: Loads the model with LoRa configuration and prepares it for inference.
Inference: Uses a multi-threaded approach to run inference on the dataset, splitting it into subsets for parallel processing.
By following these steps, you can fine-tune a ClinicalBERT model, and implement and run inference using Siamese and Prototypical Networks for ICD-10 code prediction. Adjustments may be necessary based on the specific dataset and requirements.


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Step 1: Data Preparation
icd_data = pd.read_excel('icd10_descriptions.xlsx')
descriptions = icd_data['Description'].tolist()
icd_codes = icd_data['ICD-10 Code'].tolist()

# Step 2: Model Training
model_name = 'bert-base-uncased'
model = SentenceTransformer(model_name)
embeddings = model.encode(descriptions)


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Step 1: Data Preparation
icd_data = pd.read_excel('icd10_descriptions.xlsx')
descriptions = icd_data['Description'].tolist()
icd_codes = icd_data['ICD-10 Code'].tolist()

# Preprocess text and encode labels
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_data = tokenizer(descriptions, padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(icd_codes)

# Split data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(encoded_data, labels, test_size=0.2, random_state=42)

# Step 2: Model Fine-Tuning
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(icd_codes)))
optimizer = AdamW(model.parameters(), lr=5e-5)
train_dataset = TensorDataset(train_inputs.input_ids, train_inputs.attention_mask, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Epoch " + str(epoch)):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print("Average training loss:", avg_train_loss)

# Step 3: Embedding Query (similar to previous steps)

# Step 4: Nearest Neighbor Search (similar to previous steps)

# Step 5: Output ICD-10 Codes (similar to previous steps)

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_model.pth')

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(icd_codes)))
model.load_state_dict(torch.load('fine_tuned_model.pth'))



# Step 3: Embedding Query
query = "description query from patient file"
query_embedding = model.encode([query])[0]

# Step 4: Nearest Neighbor Search
similarity_scores = cosine_similarity([query_embedding], embeddings)[0]

# Step 5: Output ICD-10 Codes
top_indices = similarity_scores.argsort()[-5:][::-1]
top_icd_codes = [icd_codes[idx] for idx in top_indices]

print("Top 5 matching ICD-10 codes:")
for code in top_icd_codes:
    print(code)


