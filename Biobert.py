import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the Excel file
excel_file_path = 'ICDMaster.xlsx'
df = pd.read_excel(excel_file_path)
df = df[20000:23000]
# Extract ICD-10 codes and their descriptions
icd_codes = df['ICD'].tolist()
icd_descriptions = df['Description'].tolist()
icd_descriptions = [text.lower() for text in icd_descriptions]
patient_notes = 'acute upper respiratory inflammatory infection, unspecified'

# Load the biomedical language model from Hugging Face
model_name = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get vector embeddings
def get_embeddings(texts, batch_size=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

# Get embeddings for ICD-10 descriptions
icd_embeddings = get_embeddings(icd_descriptions)

# Get embeddings for patient notes
patient_notes_embedding = get_embeddings([patient_notes])

# Calculate cosine similarity
cos_sim = cosine_similarity(patient_notes_embedding, icd_embeddings)

# Find the most similar ICD-10 code
most_similar_idx = cos_sim.argmax()
most_similar_icd = icd_codes[most_similar_idx]

# Output the corresponding ICD-10 code
print(f"The most similar ICD-10 code is: {most_similar_icd}")

# Save the result back to Excel
df['Most_Similar_ICD10'] = [most_similar_icd] * len(df)
output_excel_path = 'path/to/your/output_excel_file.xlsx'
df.to_excel(output_excel_path, index=False)
