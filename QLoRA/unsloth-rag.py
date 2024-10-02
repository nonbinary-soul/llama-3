#!/home/lee/miniconda3/envs/unsloth_llamacpp/bin/python
# this code has been copied from other source. 
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import atexit, time, os

start_time=time.time()

# load ebo model
model_path = "./model/unsloth.Q4_K_M.gguf"
ebo_model = LlamaCpp(model_path=model_path, n_gpu_layers=-1, temperature=0.5, top_p=0.5, stop=["<|end_of_text|>"])

# Ensures the model is closed properly before Python shuts down
# to avoid resource cleanup errors with llama_cpp.
@atexit.register
def free_model():
    ebo_model.client.close()

# create the embeddings
model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}

embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# create the vector db and the retriever
db = Chroma(embedding_function=embeddings_model)

retriever = db.as_retriever()

db.add_texts(["harrison has one apple and two orange",
              "bears has two apples and one banana"])

# create the prompt
template = """Answer the question based only on the following context: 
{context}

<USER>
{question}

<ASSISTANT>
"""
prompt = PromptTemplate.from_template(template)

# create the chain
output_parser = StrOutputParser()

def format_docs(docs):

    text = ""

    for d in docs:
        text += f"- {d.page_content}\n"

    return text

setup_and_retrieval = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt | ebo_model | output_parser

def request_model(question):

    # Check if question is already in database
    results = retriever.invoke(question)

    # If so, show them
    answer=""
    if results:
        print("Question asked before:")
        for doc in results:
            answer += f"{doc.page_content}\n"
        return answer.strip()

    # If not, request model a response
    print("No results, requesting model...")
    
    # request question
    response = chain.invoke(question)    
    # Add question and response to the database
    db.add_texts([f"Pregunta: {question} | Respuesta: {response}"])
    
    return response

"""
start_response_time=time.time()
# prompt the LLM
print(request_model("what does harrison and bears have?"))
end_response_time=time.time()
response_time=end_response_time-start_response_time
print("Response time: ", response_time)

start_response_time=time.time()
# prompt the LLM
print(request_model("what does bears have?"))
end_response_time=time.time()
response_time=end_response_time-start_response_time
print("Response time: ", response_time)
"""

start_response_time=time.time()
# prompt the LLM
print(request_model("can you place the mug to the head of the table"))
end_response_time=time.time()
response_time=end_response_time-start_response_time
print("Response time: ", response_time)

end_time=time.time()
total_time=end_time-start_time
print("Execution time: ", total_time)
