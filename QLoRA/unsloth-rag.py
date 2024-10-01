#!/home/lee/miniconda3/envs/unsloth_llamacpp/bin/python
# this code has been copied from other source. 
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# create the prompt
template = """Answer the question based only on the following context: 
{context}

---
Answer the question based on the above context: {question}
"""
prompt = PromptTemplate.from_template(template)

# load ebo model
model_path = "./model/unsloth.Q8_0.gguf"
ebo_model = LlamaCpp(model_path=model_path, n_gpu_layers=-1, temperature=0.7, top_p=0.9, stop=["<ASSISTANT>"])

# create the embeddings
model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {"device": "cpu"}
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

# prompt the LLM
print(chain.invoke("what does harrison have?"))

# prompt the LLM
#print(chain.invoke("what does bears have?"))
