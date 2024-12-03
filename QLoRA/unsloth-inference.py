#!/home/lee/miniconda3/envs/unsloth_llamacpp/bin/python
import time
import json
from llama_cpp import Llama

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def generate_text_from_prompt(user_prompt, max_tokens=100, temperature=0.3, top_p=0.1, echo=True, stop=["<|end_of_text|>"]):
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

def extract_assistant_response(model_output):
    assistant_tag = "Expected Output:"
    if assistant_tag in model_output:
        response = model_output.split(assistant_tag, 1)[-1].strip()
        return response.split("\n", 1)[0].strip()
    return model_output.strip()

def format_json_input(input_json, system_prompt):
    input_str = json.dumps(input_json, ensure_ascii=False)
    return f"{system_prompt}\n\nInput JSON: {input_str}\n\nExpected Output:"

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

    system_prompt = """Given a JSON, present a shopping list where you must provide one incorrect item that doesn't belong there. If the user identifies the incorrect item correctly, you must provide the prices of the remaining items and ask the user for the total. If the user is correct, proceed; if not, start over to achieve the indicated goal."""

    input_json_path = "./inference-inputs/input_data.json"
    input_json_example = load_json_from_file(input_json_path)

    if input_json_example:
        user_prompt = format_json_input(input_json_example, system_prompt)
        ebo_response = generate_text_from_prompt(user_prompt)
        final_result = extract_assistant_response(ebo_response)
        print("\nModel response:", final_result)
    else:
        print("Invalid input JSON. Cannot proceed.")

    end_time = time.time()
    execution_time = format_time(end_time - start_time)
    print(f"Execution time: {execution_time} second(s)")
