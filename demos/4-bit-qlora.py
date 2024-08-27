from llama_cpp import Llama
import time 

# Ruta al modelo local
model_path = "../QLoRA/llama-3-8b-bnb-4bit-qlora/adapter_model.safetensors"

# Crear el LLM usando la ruta al modelo local
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  # Para usar la aceleración GPU si es necesario
    n_ctx=2048 * 4,  # Aumenta la ventana de contexto si es necesario
)

# Get the response and execution time
def get_response_and_time(prompt):
    start_time = time.time()  # Captura el tiempo inicial
    response = llm(
        prompt=prompt,
        max_tokens=2048,
        stop=["### Instruction:\n"],
        temperature=0
    )
    end_time = time.time()  # Captura el tiempo final
    response_time = end_time - start_time  # Calcula el tiempo de respuesta
    return response["choices"][0]["text"], response_time

# Queries to the model
prompt=(
        "### Instruction:\n"
        "Quiero que me transformes frases a comandos, estos serían ejemplos:\n"
        "- there are a lot of couches in the living room = BEING_LOCATED(theme:\"a lot of couches\",location:\"in the living room\")\n"
        "- place the mug on the sink nearest to the refrigerator = PLACING(goal:\"on the sink nearest to the refrigerator\",theme:\"the mug\")\n"
        "\nAhora tradúceme la siguiente frase:\n"
        "can you place the mug to the head of the table\n\n"
        "### Assistant:\n"
    )

response_text, response_time = get_response_and_time(prompt)

print("\nModel response:")
print(response_text)
print(f"\nResponse time: {response_time:.2f} second(s)")

prompt=(
        "### Instruction:\n"
        "y esta can you place the mug to the head of the table\n\n"
        "### Assistant:\n"
    )

response_text, response_time = get_response_and_time(prompt)

print("\nModel response:")
print(response_text)
print(f"\nResponse time: {response_time:.2f} second(s)")