#!/home/lee/miniconda3/envs/unsloth_env/bin/python
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

def load_model_and_tokenizer(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(model_name)

    FastLanguageModel.for_inference(model)
    
    model.to("cuda")  # Mover el modelo a la GPU
    return model, tokenizer

def generate_response(model, tokenizer, question):
    llama_prompt = """<start>You are an assistant
<USER>
{}

<ASSISTANT>
{}"""
    
    # Formatear el prompt para la generación
    prompt = llama_prompt.format(question, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Crear un streamer para capturar la salida generada
    response_streamer = TextStreamer(tokenizer)

    # Generar la respuesta
    _ = model.generate(**inputs, streamer=response_streamer, max_new_tokens=128)

if __name__ == "__main__":
    model_name = "llama-3-8b-bnb-4bit-qlora"  # Nombre del modelo guardado
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Ejemplo de pregunta para obtener respuesta
    question = "can you place the mug to the head of the table"
    print("Generated response: ")
    generate_response(model, tokenizer, question)
