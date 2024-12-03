#!/home/lee/miniconda3/envs/unsloth_llamacpp/bin/python
import time
import json
from llama_cpp import Llama
from accelerate import Accelerator

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# Initialize accelerator
accelerator = Accelerator()

# Load model
model_path = "./model/unsloth.Q4_K_M.gguf"
ebo_model = Llama(model_path=model_path)

ebo_model = accelerator.prepare(ebo_model)

def generate_text_from_prompt(user_prompt, max_tokens=300, temperature=0.3, top_p=0.1, stop=["<|end_of_text|>"]):
    try:
        model_output = ebo_model(
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        return model_output.get("choices", [{}])[0].get("text", "").strip()
    except Exception as e:
        print(f"Error during text generation: {e}")
        return ""

def format_json_input(input_json, system_prompt, conversation_history=None):
    input_str = json.dumps(input_json, ensure_ascii=False)
    if conversation_history:
        prompt = f"{system_prompt}\n\nInput JSON: {input_str}\n\nConversation History:\n{conversation_history}\n\nExpected Output:"
    else:
        prompt = f"{system_prompt}\n\nInput JSON: {input_str}\n\nExpected Output:"
    return prompt

def load_json_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}

if __name__ == "__main__":
    start_time = time.time()

    system_prompt = """You are a shopping assistant. Follow these steps:
    1. Present a list of items for a user to identify the incorrect one.
    2. If the user identifies correctly, provide the prices of the remaining items and ask for the total cost.
    3. Correct the user if needed and ensure they understand the task.
    Respond in a concise and structured manner based on the JSON input."""

    input_json_path = "./inference-inputs/input_data.json"
    input_json_example = load_json_from_file(input_json_path)

    if input_json_example:
        conversation_history = ""  # Initialize conversation history
        while True:
            user_prompt = format_json_input(input_json_example, system_prompt, conversation_history)
            
            # Debugging the user prompt
            print("\nDebugging Prompt:\n", user_prompt)
            
            assistant_response = generate_text_from_prompt(user_prompt)
            print("\nAssistant:", assistant_response)

            # Simulate user input or exit condition
            user_input = input("Your input (or type 'exit' to finish): ")
            if "exit" in user_input.lower() or "next scene" in assistant_response:
                break
            conversation_history += f"Assistant: {assistant_response}\nUser: {user_input}\n"
    else:
        print("Invalid input JSON. Cannot proceed.")

    end_time = time.time()
    execution_time = format_time(end_time - start_time)
    print(f"Execution time: {execution_time} second(s)")
