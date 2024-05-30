from typing import List

from transformers import AutoTokenizer, AutoModel
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pprint import pprint
import torch


def split_text() -> List[Document]:
    loader = JSONLoader(
        file_path="/root/text_model/bge_large_en/pubmedtemp.jsonl",
        jq_schema=".abstract",
        text_content=False,
        json_lines=True
    )
    data = loader.load()
    data = [Doc.page_content for Doc in data]

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    texts_doc = text_splitter.create_documents(data)
    return texts_doc


def load_model_from_local():
    model_path = "/root/text_model/bge_large_en/model"
    return AutoModel.from_pretrained(model_path, trust_remote_code=True)


def load_tokenizer_from_local():
    model_path = "/root/text_model/bge_large_en/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


def connect_milvus() -> None:
    connections.connect("default", db_name="text_embedding", host="127.0.0.1", port="19530")


def load_milvus_collection() -> None:
    pass


def insert_vectors():
    pass


def search_vectors(vectors_to_search: list):
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 5}
    }
    bge_test_collection = Collection("bge_test")
    result = bge_test_collection.search(vectors_to_search, "vector", search_params, limit=3)
    for hits in result:
        for hit in hits:
            pprint(hit)

def build_vector_data():
    """
    从本地文件中读取文本，然后分词，分词后，将分词后的文本进行向量化，然后将向量插入到Milvus
    :return:
    """
    texts_doc = split_text()
    has = utility.has_collection("bge_test")
    print(f"Dose exist in Milvus: {has})")
    bge_test_collection = Collection("bge_test")

    model = load_model_from_local()
    model.eval()
    tokenizer = load_tokenizer_from_local()

    entities = []
    vectors = []
    vec_texts = []

    for doc in texts_doc:
        sentences = doc.page_content
        # pprint(sentences)

        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            # print("Model output:", model_output)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        #        print("Length:", len(sentence_embeddings[0]))
        #        print("Sentence embeddings:", sentence_embeddings[0])

        vectors.append(sentence_embeddings[0].tolist())
        vec_texts.append(sentences)

    entities.append(vectors)
    entities.append(vec_texts)
    pprint(entities)
    insert_result = bge_test_collection.insert(entities)
    pprint(insert_result)
    bge_test_collection.flush()


def search_sentences_from_milvus(sentences: str):
    """
    生成text的向量，并查询milvus
    :param text:
    :return:
    """
    model = load_model_from_local()
    model.eval()
    tokenizer = load_tokenizer_from_local()

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        print("Model output:", model_output)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    print("Length:", len(sentence_embeddings[0]))
    print("Sentence embeddings:", sentence_embeddings[0])
    search_vectors(sentence_embeddings.tolist())


if __name__ == "__main__":
    connect_milvus()
    build_vector_data()
    #search_sentences_from_milvus('Analysis of the subcellular distribution of VEGFR-1 revealed the appearance of an 80-kDa C-terminal domain in the cytosol of cells treated with VEGF and PEDF that correlated with a decrease of the full-length receptor in the nuclear and cytoskeletal fractions.')

