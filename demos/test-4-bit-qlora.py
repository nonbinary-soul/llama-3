#!/home/lee/miniconda3/envs/unsloth_env/bin/python

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ruta donde guardaste el modelo y el tokenizador
MODEL_PATH = "../QLoRA/llama-3-8b-bnb-4bit-qlora"

# Carga del modelo y el tokenizador
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Configuración para GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ejemplo de entrada para la inferencia
input_text = "<start>User\nHello, how are you?<end>"

# Tokeniza la entrada
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generación de la respuesta
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50, do_sample=True)

# Decodifica la salida
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Muestra la predicción generada
print(f"Entrada: {input_text}")
print(f"Salida: {output_text}")
