from langchain.document_loaders import TextLoader
from langchain.document_loaders.python import PythonLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores.vearch import Vearch
from transformers import AutoModel, AutoTokenizer
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch



if __name__ == "__main__":
    model_path = "/root/code_model/codet5p-110m-embedding"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    file_path = "/root/git_clone/CellPLM/CellPLM/utils/data.py"  # Your local file path"
    # file_path="D:\\work_software\\zj_code\\src\\utils.py"
    loader = PythonLoader(file_path)
    documents = loader.load()
    text_splitter = PythonCodeTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    # print(texts[1].page_content)
    # print("==========================")
    # print(type(texts[1]))
    # embeddings = HuggingFaceEmbeddings(model_name=model_path)
    # # print("================================")
    # # print(documents)
    # es = Elasticsearch("https://111.229.112.73:9200", basic_auth=('elastic', 'Zjlab@123'), verify_certs=False)
    # # db = ElasticsearchStore()
    #
    # # elastic_vector_search = ElasticsearchStore.from_documents(
    # #     documents,
    # #     embeddings,
    # #     es_url="https://172.17.0.9:9200",
    # #
    # #     es_user="elastic",
    # #     es_password="Zjlab@123",
    # # )
    # db = ElasticsearchStore.from_documents(
    #     documents,
    #     embeddings,
    #     index_name="code_test_index",
    #     es_connection=es
    # )
    # print(texts)
    for i in range(len(texts)):
        inputs=tokenizer.encode(texts[i].page_content,return_tensors="pt").to(device)
        embedding = model(inputs)[0]
        print(embedding)
        print("=================")




