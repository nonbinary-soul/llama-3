#!/home/lee/miniconda3/envs/unsloth_env/bin/python
import os
import time 
import torch

from trl import SFTTrainer
from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported

from transformers import (
    TrainingArguments,
    BitsAndBytesConfig
)

from functools import partial
from datasets import load_dataset

# print versions for torch and cuda
#print(torch.__version__)
#print(torch.version.cuda)

###################################################################################

max_seq_length = 2048
dtype=None
load_in_4bit = True

###################################################################################

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

start_time=time.time()

def create_model_and_tokenizer(model_name): 

    # model creation
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length = max_seq_length,
        dtype=dtype,
        load_in_4bit = load_in_4bit
    )

    print("# PRINTING MODEL INFO...")
    print(model)

    return model, tokenizer

# getting model and tokenizer
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
model, tokenizer = create_model_and_tokenizer(MODEL_NAME)

###################################################################################

llama_prompt = """
{system_prompt}

Input JSON: {input_json}

Conversation History:
{conversation_history}
"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    
    # Instrucción general del sistema
    system_prompt = """Given a JSON, present a shopping list where you must provide one incorrect item that doesn't belong there. If the user identifies the incorrect item correctly, you must provide the prices of the remaining items and ask the user for the total. 
    If the user is correct, proceed; if not, start over to achieve the indicated goal."""
    
    for example in examples:
        # extracting parts
        input_json = example["input"]
        assistant_turns = example.get("assistant", "").split("\n")
        user_turns = example.get("user", "").split("\n")
        
        # building the conversation history
        conversation_history = []
        for assistant, user in zip(assistant_turns, user_turns):
            conversation_history.append(f"Assistant: {assistant.strip()}")
            conversation_history.append(f"User: {user.strip()}")
        
        # concatenating the conversation history into a single string
        conversation_history_str = "\n".join(conversation_history)
        
        # formatting using llama_prompt
        formatted_text = llama_prompt.format(
            system_prompt=system_prompt,
            input_json=str(input_json),
            conversation_history=conversation_history_str
        ) + EOS_TOKEN
        
        texts.append(formatted_text)
    
    return {"text": texts}


###################################################################################

dataset = load_dataset('json', data_files='datasets/data-ebo-conversations.json', split='train')

# Showing possible dataset keys to use in 'tokenize' function. In this case, it is available 'command' and 'cfr'
example = dataset[0]
print("Dataset keys:", example.keys()) 

dataset = dataset.map(
    formatting_prompts_func,
    batched=True # process each input individually
)

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
lora_alpha = 16
lora_dropout = 0.05
modules_to_save = ["lm_head", "embed_tokens"]

model.config.use_cache = False

model = FastLanguageModel.get_peft_model(
    model, 
    r=lora_r, # Lora attention dimension
    lora_alpha=lora_alpha, # Scaling factor that changes how the adaptation layer's weights affect the base model's
    lora_dropout=lora_dropout,
    bias="none",
    use_rslora=False,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    loftq_config = None,
    target_modules=target_modules # List of module names or regex expression of the module names to replace with LoRA
#    modules_to_save=modules_to_save # List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint
)

###################################################################################

# Set training arguments
training_arguments = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    max_steps=-1,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    weight_decay=0.001,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    group_by_length=True,
    save_steps=0,
    logging_steps=1000,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model, # model to train
    tokenizer=tokenizer, # used for tokenizing text during data preparation and potentially for inference
    train_dataset=dataset,
    dataset_text_field="text",
    packing=False,
    args=training_arguments, # argument configuration
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
modelname_lowercase = MODEL_NAME.split("/")[-1].lower()
new_model_name = modelname_lowercase+"-qlora"

# Save trained model
trainer.model.save_pretrained(new_model_name)
print("Trained model saved!!")

# Save model and tokenizer
model.save_pretrained(new_model_name)
print("Model saved!!")

model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

tokenizer.save_pretrained(new_model_name)
print("Tokenized model saved")

# Print total execution time
end_time = time.time()
total_time = format_time(end_time - start_time)
print(f"Execution time: {total_time} second(s)")
