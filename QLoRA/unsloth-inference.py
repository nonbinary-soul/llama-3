#!/home/lee/miniconda3/envs/unsloth_llamacpp/bin/python
import time, os
from llama_cpp import Llama
import json
import torch
from accelerate import Accelerator

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# initialize accelerator
accelerator = Accelerator()

# load ebo model
model_path = "./model/unsloth.Q4_K_M.gguf"
ebo_model = Llama(model_path=model_path)

ebo_model = accelerator.prepare(ebo_model)

def generate_text_from_prompt(user_prompt, max_tokens = 100, temperature = 0.3, top_p = 0.1, echo = True, stop = ["<|end_of_text|>"]):

   # Define the parameters
   model_output = ebo_model(
       user_prompt,
       max_tokens=max_tokens,
       temperature=temperature,
       top_p=top_p,
       stop=stop,
   )

   return model_output

def extract_assistant_response(model_output):
    assistant_tag = "Expected Output:"
    if assistant_tag in model_output:
        return model_output.split(assistant_tag)[-1].strip()
    return model_output.strip()

def format_json_input(input_json, system_prompt):
    input_str = json.dumps(input_json, ensure_ascii=False)
    prompt = f"{system_prompt}\n\nInput JSON: {input_str}\n\nExpected Output:"
    return prompt

def load_json_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
       
if __name__ == "__main__":
    start_time = time.time()

    system_prompt = """Given a JSON, present a shopping list where you must provide one incorrect item that doesn't belong there. If the user identifies the incorrect item correctly, you must provide the prices of the remaining items and ask the user for the total. If the user is correct, proceed; if not, start over to achieve the indicated goal."""

    input_json_path = "./inference-inputs/input_data.json"
    input_json_example = load_json_from_file(input_json_path)

    user_prompt = format_json_input(input_json_example, system_prompt)

    with torch.no_grad():
        ebo_response = generate_text_from_prompt(user_prompt)
        final_result = extract_assistant_response(ebo_response["choices"][0]["text"])

    print("\nModel response:", final_result)
    print("\n")

    end_time = time.time()    
    execution_time = format_time(end_time - start_time)
    print(f"Execution time: {execution_time} second(s)")
