from langchain.llms import ChatGLM
from langchain.llms import LlamaCpp,GPT4All

from langchain import PromptTemplate, LLMChain
from langchain.chains.summarize import load_summarize_chain
import json

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import MiniMaxEmbeddings

file_path = "/root/llm/baichuan-13B/docs/ecs_introdution.txt"
loaders = TextLoader(file_path=file_path)
doc = loaders.load()
text_spliter = CharacterTextSplitter(chunk_size=200,chunk_overlap=0)
docs = text_spliter.split_documents(doc)

embeddings = MiniMaxEmbeddings()
db = FAISS.from_documents(docs,embeddings)

query = "什么是ecs？"
similarity_result=db.similarity_search(query)

from langchain.chains.question_answering import load_qa_chain

endpoint_url = "http://127.0.0.1:8000/chat"
llm = ChatGLM(
    endpoint_url=endpoint_url
)

chain = load_qa_chain(llm=llm,chain_type="stuff")
response= chain({"input_documents": docs, "question": query},return_only_outputs=True)

print(response)
