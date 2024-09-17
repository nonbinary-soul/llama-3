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
templates = {
    "system": "<start>System\n{msg}<end>",
    "user": "<start>User\n{msg}<end>",
    "assistant": "<start>Assistant\n{msg}<end>"
}

IGNORE_INDEX = -100 # we use this to ignore user's tokens

# This function formats the input text and response into tokenized sequences, ensuring each input and label is aligned and of the correct length
def tokenize(data, max_length, tokenizer):
    input_ids, attention_mask, labels = [], [], []

    for conversation_block in data["conversation"]:
        role = conversation_block["role"]
        content = conversation_block["content"]
        formatted_message = templates[role].format(msg=content)
        tokenized_message = tokenizer(formatted_message, truncation=False, add_special_tokens=False)

        input_ids += tokenized_message["input_ids"]
        attention_mask += tokenized_message["attention_mask"]

        if role == "assistant":
            labels += tokenized_message["input_ids"]
        else:
            labels += [IGNORE_INDEX] * len(tokenized_message["input_ids"])

    assert len(input_ids) == len(attention_mask) == len(labels), "Length mismatch in tokenization."

    return {
        "input_ids": input_ids[:max_length], # IDs of the tokens of the whole text
        "attention_mask": attention_mask[:max_length], # indicates which tokens are real (1) and which are padding (0)
        "labels": labels[:max_length], # IDs of the correct tokens the model should learn to predict
    }

###################################################################################

def prepare_dataset(tokenizer):
    dataset = load_dataset('json', data_files='data.json', split='train')
    
    # Showing possible dataset keys to use in 'tokenize' function. In this case, it is available 'command' and 'cfr'
    example = dataset[0]
    print("Dataset keys:", example.keys()) 
    
    return dataset.map(
        partial(tokenize, max_length=1024, tokenizer=tokenizer),
        batched=False, # process each input individually
        num_proc=os.cpu_count(), # use several processes in parallel to accelerate the processing
        remove_columns=dataset.column_names
    )

###################################################################################

# This function pads the input data and organizes it into batches for training
def collate(elements, tokenizer):
    tokens = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokens])
    
    for sample in elements:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]
        pad_len = tokens_maxlen - len(input_ids)
        input_ids.extend(pad_len * [tokenizer.pad_token_id]) # pad_len repetions of pad_token_id value
        attention_mask.extend(pad_len * [0]) 
        labels.extend(pad_len * [IGNORE_INDEX]) 

    batch = {
        "input_ids": torch.tensor([e["input_ids"] for e in elements]),
        "labels": torch.tensor([e["labels"] for e in elements]),
        "attention_mask": torch.tensor([e["attention_mask"] for e in elements]),
    }

    return batch

###################################################################################
#################################### LoRA #########################################
###################################################################################

def find_target_modules(model):
    
    # Initialize a set to store unique layers
    unique_layers = set()
    
    # Iterate over all named modules in the model
    for name, module in model.named_modules():
        # Check if the module type contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the type of the layer
            layer_type = name.split('.')[-1]
            # Add the layer type to the set of unique layers
            unique_layers.add(layer_type)
    
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
    model=model, # model to train
    tokenizer=tokenizer, # used for tokenizing text during data preparation and potentially for inference
    args=training_arguments, # argument configuration
    data_collator=partial(collate, tokenizer=tokenizer), # prepares data into batches 
    train_dataset=prepare_dataset(tokenizer) # tokenized dataset to adjust the model
)

######################################################################################################
# Example using the tokenizer to decode the results, transforming numeric ids into text
# # Generating predictions
# predictions = trainer.predict(test_dataset)

# # Decoding the predictions
# decoded_predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
######################################################################################################

# Checking vocabulary size
print("Checking vocabulary size...")
vocab_size = len(tokenizer)
embedding_size = model.get_input_embeddings().weight.size(0)
assert vocab_size == embedding_size, f"Mismatch between vocab size ({vocab_size}) and embedding size ({embedding_size})"

# Train model
print("Training model...")
trainer.train()
print("Model trained!!")

###################################################################################

# Getting new name for the models
modelname_lowercase = MODEL_NAME.split("/")[-1].lower
new_model_name = modelname_lowercase+"-qlora"

# Save trained model
trainer.model.save_pretrained(new_model_name)
print("Trained model saved!!")

# Save base model
model.save_pretrained(new_model_name)
print("Base model saved!!")

# Save tokenized model
print("Tokenized model saved!!")
tokenizer.save_pretrained(new_model_name)