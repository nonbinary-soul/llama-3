#!/home/lee/miniconda3/envs/unsloth_env/bin/python
import time 
from llama_cpp import Llama

model_path = "./model/unsloth.Q8_0.gguf"
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
    print("Model response:\n", final_result)

    end_time = time.time()    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} second(s)")
