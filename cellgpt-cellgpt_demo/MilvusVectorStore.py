#!pip install llama-index pymilvus milvus python-dotenv 
#!pip install --upgrade pyarrow
import logging
import sys

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import MilvusVectorStore
from IPython.display import Markdown, display
import textwrap

#Setup OpenAI
import openai

openai.api_key = "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"

#Download Data
#!mkdir -p 'data/paul_graham/'
#!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

#Generate our data
# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

print("Document ID:", documents[0].doc_id)

#Create an index across the data
# Create an index over the documnts
from llama_index.storage.storage_context import StorageContext


vector_store = MilvusVectorStore(dim=1536, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

#Query the data
query_engine = index.as_query_engine()
response = query_engine.query("What did the author learn?")
print(textwrap.fill(str(response), 100))

response = query_engine.query("What was a hard moment for the author?")
print(textwrap.fill(str(response), 100))
#This next test shows that overwriting removes the previous data
vector_store = MilvusVectorStore(dim=1536, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [Document(text="The number that is being searched for is ten.")],
    storage_context,
)
query_engine = index.as_query_engine()
res = query_engine.query("Who is the author?")
print("Res:", res)
#The next test shows adding additional data to an already existing index
del index, vector_store, storage_context, query_engine

vector_store = MilvusVectorStore(overwrite=False)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
query_engine = index.as_query_engine()
res = query_engine.query("What is the number?")
print("Res:", res)

res = query_engine.query("Who is the author?")
print("Res:", res)


