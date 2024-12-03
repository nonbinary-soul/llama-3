#!/home/lee/miniconda3/envs/unsloth_llamacpp/bin/python
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import atexit, time

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
embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
embedding_kwargs = {"device": "cuda"}
embedding_function = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_kwargs,
    encode_kwargs={"normalize_embeddings": True}
)

# create the vector db and the retriever
db = Chroma(embedding_function)

retriever = db.as_retriever()

# create the prompt
template = """
Given the context below, respond as follows:
1. Identify any incorrect item in the shopping list.
2. Provide the corrected list and their individual prices.
3. Ask the user to calculate the total.
4. Confirm if their answer is correct.

Context:
{context}

User: 
{question}

Assistant:
"""

prompt = PromptTemplate.from_template(template)
output_parser = StrOutputParser()

# chain setup: retrieval and LLM generation
def format_docs(docs):
    """Format retrieved documents for context injection."""
    return "\n".join(f"- {doc.page_content}" for doc in docs)

setup_and_retrieval = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt | ebo_model | output_parser

def request_model(question):
    """
    Main pipeline for retrieving or generating a response.
    """
    # retrieve context if it exists
    results = retriever.invoke(question)
    if results:
        formatted_context = "\n".join(doc.page_content for doc in results)
        print(f"Retrieved context:\n{formatted_context}")
    else:
        formatted_context = "No relevant information found in the database."

    # query the model
    response = chain.invoke(question)

    # add the new question-response pair to the database
    db.add_texts([f"Question: {question} | Answer: {response}"])
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
