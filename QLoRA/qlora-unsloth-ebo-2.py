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

start_time=time.time()

def create_model_and_tokenizer(model_name): 

    # model creation
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length = 2048,
        dtype=None,
        load_in_4bit = True
    )

    print("Printing model info...")
    print(model)

    # tokenization set up
    tokenizer.add_tokens(["<start>", "<pad>"])
    tokenizer.pad_token = "<pad>"
    tokenizer.add_special_tokens(dict(eos_token="<end>"))
    
    # Set embeddings matrix of the model
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

# getting model and tokenizer
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
model, tokenizer = create_model_and_tokenizer(MODEL_NAME)

###################################################################################

# load the dataset
llama_prompt = """<start>
User
{}
<end>
<start>
Assistant
{}
<end>
"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    commands = examples["command"]
    cfrs = examples["cfr"]
    texts = []
    for command, cfr in zip(commands, cfrs):
        text = llama_prompt.format(command, cfr) + EOS_TOKEN 
        texts.append(text)
    return { "text" : texts, }

###################################################################################

def prepare_dataset():
    dataset = load_dataset('json', data_files='default-data.json', split='train')
    
    # Showing possible dataset keys to use in 'tokenize' function. In this case, it is available 'command' and 'cfr'
    example = dataset[0]
    print("Dataset keys:", example.keys()) 
    
    return dataset.map(
        formatting_prompts_func,
        batched=False, # process each input individually
        num_proc=os.cpu_count(), # use several processes in parallel to accelerate the processing
        remove_columns=dataset.column_names
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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
trainer = SFTTrainer(
    model=model, # model to train
    tokenizer=tokenizer, # used for tokenizing text during data preparation and potentially for inference
    args=training_arguments, # argument configuration
    train_dataset=prepare_dataset() 
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

tokenizer.save_pretrained(new_model_name)
print("Tokenized model saved")

FastLanguageModel.for_inference(model)
print("Model prepared for inference!!")

# Getting total time
end_time = time.time()

total_duration = end_time - start_time
print(f"Execution time: {total_duration:.2f} second(s)")


# INFERENCE
# Define prompt template
llama_prompt = "<start>User\n{}<end><start>Assistant\n"

# Define question
question = "can you place the mug to the head of the table"

# Tokenize entry for model
inputs = tokenizer(
    [llama_prompt.format(question, "")],
    return_tensors="pt"  # Retorna tensores de PyTorch
).to("cuda")  # Mueve los tensores a la GPU si es posible

start_response_time=time.time()

# Generar la respuesta

# without streamer
#outputs = model.generate(
#    **inputs, 
#    max_new_tokens=64,  # Controla cuántos tokens nuevos se generan
#    use_cache=True
#)

# response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
 
# TextStreamer
from transformers import TextStreamer

response=TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=response, max_new_tokens=128)


#print("Generated text:", response[0].split('<start>Assistant\n')[1].split('<end>')[0].strip())
#print("All text:", response)

end_response_time=time.time()

total_response_time= end_response_time - start_response_time

print(f"Total response time: {total_response_time:.2f} second(s)")