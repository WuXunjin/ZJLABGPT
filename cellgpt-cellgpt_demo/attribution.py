!pip install milvus llama-index python-dotenv pymilvus

from llama_index.llms import OpenAI
from llama_index.query_engine import CitationQueryEngine
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
)
from llama_index.vector_stores import MilvusVectorStore
from milvus import default_server

from dotenv import load_dotenv
import os
load_dotenv()
open_api_key = os.getenv("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")

# Get test data
wiki_titles = ["Toronto", "Seattle", "San Francisco", "Chicago", "Boston", "Washington, D.C.", "Cambridge, Massachusetts", "Houston"]
from pathlib import Path

import requests
for title in wiki_titles:
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))
    wiki_text = page['extract']

    data_path = Path('data')
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", 'w') as fp:
        fp.write(wiki_text)

# vector store
default_server.start()
vector_store = MilvusVectorStore(
    collection_name="citations",
    host="127.0.0.1",
    port=default_server.listen_port
)

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0)
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("./data/").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)

# Query with reference
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # 此处我们可以控制引用来源的粒度，默认值为 512
    citation_chunk_size=512,
)
response = query_engine.query("Does Seattle or Houston have a bigger airport?")
print(response)
for source in response.source_nodes:
    print(source.node.get_text())