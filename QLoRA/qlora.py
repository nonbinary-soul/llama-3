#!/home/lee/miniconda3/envs/llama_env/bin/python3
import os
from functools import partial
from datasets import load_dataset
from transformers import (
    Trainer, TrainingArguments,
    AutoTokenizer, BitsAndBytesConfig,
    AutoModelForCausalLM
)

from peft import (
    LoraConfig, get_peft_model,
    prepare_model_for_kbit_training
)

import torch


# print versions for torch and cuda
#print(torch.__version__)
#print(torch.version.cuda)

###################################################################################

def create_model_and_tokenizer(model_name): 
    # quantization set up
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False, # change to True to get better accuracy
        bnb_4bit_quant_type='nf4'
    )

    # model creation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": 0} # asigns the model to GPU 0
    )

    print("Printing model info...")
    print(model)

    # tokenization set up
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.add_tokens(["<start>", "<pad>"])
    tokenizer.pad_token = "<pad>"
    tokenizer.add_special_tokens(dict(eos_token="<end>"))
    
    # Set embeddings matrix of the model
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

# getting model and tokenizer
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
model, tokenizer = create_model_and_tokenizer(MODEL_NAME)

###################################################################################

# load the dataset
templates = [
    "<start>Assistant\n{msg}<end>",
    "<start>User\n{msg}<end>"
]

IGNORE_INDEX = -100 # we use this to ignore user's tokens

def tokenize(input, max_length, tokenizer):
    input_ids, attention_mask, labels = [], [], []

    for i, msg in enumerate([input["command"], input["cfr"]]):
        isHuman = i % 2 == 0
        msg_chatml = templates[isHuman].format(msg=msg)
        msg_tokenized = tokenizer(
        msg_chatml, truncation=False, add_special_tokens=False)
        input_ids += msg_tokenized["input_ids"]
        attention_mask += msg_tokenized["attention_mask"]
        labels += [IGNORE_INDEX] * len(msg_tokenized["input_ids"]
    ) if isHuman else msg_tokenized["input_ids"]

    assert len(input_ids) == len(attention_mask) == len(labels), "Length mismatch in tokenization."

    return {
        "input_ids": input_ids[:max_length], # tokens of the text
        "attention_mask": attention_mask[:max_length], # binary tensor that indicates which tokens should be attended
        "labels": labels[:max_length], # ground truth tokens
    }

###################################################################################

def prepare_dataset(tokenizer):
    dataset = load_dataset('json', data_files='data.json', split='train')
    
    # Showing possible dataset keys to use in 'tokenize' function
    example = dataset[0]
    print("Dataset keys:", example.keys())
    
    return dataset.map(
        partial(tokenize, max_length=1024, tokenizer=tokenizer),
        batched=False, # process each input individually
        num_proc=os.cpu_count(), # use several processes in parallel to accelerate the processing
        remove_columns=dataset.column_names
    )

###################################################################################

# collate to prepare a batch of data for input to a LLM for training
def collate(elements, tokenizer):
    tokens = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokens])
    
    for sample in elements:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]
        pad_len = tokens_maxlen - len(input_ids)
        input_ids.extend(pad_len * [tokenizer.pad_token_id]) # pad_len repetions of pad_token_id value
        attention_mask.extend(pad_len * [0]) # pad_len repetions of 0
        labels.extend(pad_len * [IGNORE_INDEX]) # pad_len repetions of IGNORE_INDEX value

    batch = {
        "input_ids": torch.tensor([e["input_ids"] for e in elements]),
        "labels": torch.tensor([e["labels"] for e in elements]),
        "attention_mask": torch.tensor([e["attention_mask"] for e in elements]),
    }

    return batch

dataset = prepare_dataset(tokenizer)

###################################################################################

def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]
            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)
    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


# LoRA configuration
target_modules = find_target_modules(model) 
print("Target modules to use: ", target_modules)
lora_r = 16 
lora_alpha = 32
lora_dropout = 0.05
modules_to_save = ["lm_head", "embed_tokens"]
task_type="CAUSAL_LM"

lora_config = LoraConfig(
    r=lora_r,# Lora attention dimension
    lora_alpha=lora_alpha,# Scaling factor that changes how the adaptation layer's weights affect the base model's
    target_modules=target_modules,# List of module names or regex expression of the module names to replace with LoRA
    modules_to_save=modules_to_save, # List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint
    lora_dropout=lora_dropout,
    task_type=task_type
)

###################################################################################

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False
model = get_peft_model(model, lora_config)

# Set training arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    max_steps=-1,
    fp16=False,
    bf16=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    weight_decay=0.001,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    group_by_length=True,
    save_steps=0,
    logging_steps=20,
)

# Set supervised fine-tuning parameters
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=partial(collate, tokenizer=tokenizer),
    train_dataset=dataset
)

# Checking vocabulary size
vocab_size = len(tokenizer)
embedding_size = model.get_input_embeddings().weight.size(0)
assert vocab_size == embedding_size, f"Mismatch between vocab size ({vocab_size}) and embedding size ({embedding_size})"

print("training model...")
# Train model
trainer.train()
print("training model finished...")

###################################################################################

# Getting new name for the models
model_name_lower = MODEL_NAME.split("/")[-1].lower
new_model_name = model_name_lower+"-qlora"

# Save trained model
trainer.model.save_pretrained(new_model_name)
print("first model pretrained saved...")

# Save full model
model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)
print("second model pretrained saved...")