#!/home/lee/miniconda3/envs/unsloth_llamacpp/bin/python
import time, os
from llama_cpp import Llama

# Find .gguf model
model_path = "./model"
model_file = next((f for f in os.listdir(model_path) if f.endswith(".gguf")), None)

if model_file:
    model_path = os.path.join(model_path, model_file)
    print(f"Modelo encontrado: {model_path}")
else:
    raise FileNotFoundError(f"No model with extension .gguf was found in the {model_path} directory")

# load ebo model
ebo_model = Llama(model_path=model_path)

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
    # Busca el texto después de la etiqueta <ASSISTANT>
    assistant_tag = "<ASSISTANT>"
    if assistant_tag in model_output:
        return model_output.split(assistant_tag)[-1].strip()
    return model_output.strip()
   
if __name__ == "__main__":
    
    start_time = time.time()
    
    my_prompt = "can you place the mug to the head of the table"
    ebo_response = generate_text_from_prompt(my_prompt)
    final_result = extract_assistant_response(ebo_response["choices"][0]["text"])
    print("\n")
    print("Model response:", final_result)
    print("\n")

    end_time = time.time()    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} second(s)")
