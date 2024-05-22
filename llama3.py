!pip install -q -U bitsandbytes --no-index --find-links ../input/llm-detect-pip/
!pip install -q -U transformers --no-index --find-links ../input/llm-detect-pip/
!pip install -q -U tokenizers --no-index --find-links ../input/llm-detect-pip/
!pip install -q -U peft --no-index --find-links ../input/llm-detect-pip/

import torch
import pandas as pd
import numpy as np
import time
from transformers import AutoTokenizer, LlamaForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from threading import Thread

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

if not torch.cuda.is_available():
    print("Sorry - GPU required!")

MODEL_NAME = '/kaggle/input/llama-3/transformers/8b-chat-hf/1'
WEIGHTS_PATH = '/kaggle/input/lmsys-model/model'
MAX_LENGTH = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda")

train_data = pd.read_csv('/path/to/train_data.csv')
val_data = pd.read_csv('/path/to/val_data.csv')

def process(input_str):
    return input_str.strip()

train_data['description'] = train_data['description'].apply(process)
val_data['description'] = val_data['description'].apply(process)

print(train_data.head())

tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/lmsys-model/tokenizer')

train_tokens = tokenizer(train_data['description'].tolist(), padding='max_length',
                         max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
val_tokens = tokenizer(val_data['description'].tolist(), padding='max_length',
                       max_length=MAX_LENGTH, truncation=True, return_tensors='pt')

train_data['INPUT_IDS'] = train_tokens['input_ids'].tolist()
train_data['ATTENTION_MASKS'] = train_tokens['attention_mask'].tolist()
val_data['INPUT_IDS'] = val_tokens['input_ids'].tolist()
val_data['ATTENTION_MASKS'] = val_tokens['attention_mask'].tolist()


# BitsAndBytes configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Load base model on GPU 0
device0 = torch.device('cuda:0')
base_model_0 = LlamaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map='cuda:0'
)
base_model_0.config.pad_token_id = tokenizer.pad_token_id

# LoRa configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.10,
    bias='none',
    inference_mode=True,
    task_type=TaskType.SEQ_CLS,
    target_modules=['o_proj', 'v_proj']
)

model_0 = get_peft_model(base_model_0, peft_config).to(device0)
model_0.load_state_dict(torch.load(WEIGHTS_PATH), strict=False)
model_0.eval()


def inference(df, model, device, batch_size=BATCH_SIZE):
    input_ids = torch.tensor(df['INPUT_IDS'].values.tolist(), dtype=torch.long)
    attention_mask = torch.tensor(df['ATTENTION_MASKS'].values.tolist(), dtype=torch.long)

    generated_class_a = []
    generated_class_b = []
    generated_class_c = []

    model.eval()

    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_input_ids = input_ids[start_idx:end_idx].to(device)
        batch_attention_mask = attention_mask[start_idx:end_idx].to(device)

        with torch.no_grad():
            with autocast():
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )

        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        generated_class_a.extend(probabilities[:, 0])
        generated_class_b.extend(probabilities[:, 1])
        generated_class_c.extend(probabilities[:, 2])

    df['winner_model_a'] = generated_class_a
    df['winner_model_b'] = generated_class_b
    df['winner_tie'] = generated_class_c

    torch.cuda.empty_cache()

    return df


# Prepare data for inference
data = pd.concat([train_data, val_data], ignore_index=True)
data['text'] = 'User prompt: ' + data['description']

N_SAMPLES = len(data)

# Split the data into two subsets
half = round(N_SAMPLES / 2)
sub1 = data.iloc[0:half].copy()
sub2 = data.iloc[half:N_SAMPLES].copy()


# Function to run inference in a thread
def run_inference(df, model, device, results, index):
    results[index] = inference(df, model, device)


# Dictionary to store results from threads
results = {}

# start threads
t0 = Thread(target=run_inference, args=(sub1, model_0, device0, results, 0))
t0.start()

# Wait for all threads to finish
t0.join()

# Combine results back into the original DataFrame
data = results[0]

print("Processing complete.")
