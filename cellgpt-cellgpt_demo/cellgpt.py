import streamlit as st
from PIL import Image

from llama_index import StorageContext, ServiceContext, load_index_from_storage
from llama_index.query_engine import CitationQueryEngine
from llama_index.llms import AzureOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
import os

st.set_page_config(layout="wide", page_title="CellGPT - Zhejiang Lab", page_icon="🧬")
st.image(Image.open('zjlab.png'), width=200)
st.title("CellGPT demo")
st.markdown("by Ma Lab @ Zhejiang Lab")
st.write("CellGPT is a biological large language model focusing on single cell sequencing research. The response is based on research papers. Sample questions to ask CellGPT:")
st.write("""
+ Advances of single cell multi-omics in cancer research.
+ 介绍一下单细胞多组学融合算法。
""")




@st.cache_resource
def make_servicecontext():
    return ServiceContext.from_defaults(
        llm=AzureOpenAI(
            model="gpt-35-turbo",
            deployment_name="reportai",
            api_key=os.environ["AZURE_OPENAI_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version="2023-05-15",
        ),
        chunk_size=2056
    )
service_context = make_servicecontext()


@st.cache_resource
def load_index():
    return load_index_from_storage(StorageContext.from_defaults(persist_dir='./citation'), service_context=service_context)

pubmed_index = load_index()

@st.cache_resource
def make_query_engine():
    return CitationQueryEngine.from_args(
        pubmed_index,
        similarity_top_k=3,
        citation_chunk_size=2056,
        streaming = True
    )

query_engine = make_query_engine()

# st.sidebar.title("CellGPT")
# st.sidebar.markdown("CellGPT is a question answering system for single cell papers.")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    print(message)
    with st.chat_message("user", avatar="👩🏻‍⚕️"):
        st.markdown(message[0])
    with st.chat_message("assistant", avatar="🧬"):
        st.markdown(message[1])
        with st.expander("References"):
            st.markdown(message[2])

if prompt := st.chat_input("Ask single cell questions!"):
    
    with st.chat_message("user", avatar="👩🏻‍⚕️"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🧬"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner('Searching and thinking...'):
            response = query_engine.query(prompt)
            for chunk in response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response)
        reference = ""
        print(response.source_nodes)
        if len(response.source_nodes)>1:
            for i, node in enumerate(response.source_nodes):
                # Assuming node.node.metadata is a dictionary, use get() to avoid KeyError
                # authors = node.node.metadata.get("authors", "")



                title = node.node.metadata.get("title", "")
                if title == "":
                    title = node.node.metadata.get("Title of this paper", "No title")
                venue = node.node.metadata.get("venue", "")
                if venue == "":
                    venue = node.node.metadata.get("Journal it was published in:", "No venue")
                year = str(node.node.metadata.get("year", ""))
                doi = node.node.metadata.get("externalIds", {}).get("DOI", None)
                if doi == None:
                    doi = node.node.metadata.get("URL", "No URL")

                    # Format the message
                message = f"[{i+1}] {title}. {venue}, {year}. {doi}"
            
                # message_placeholder.markdown(message)
                # Write the message
                reference += message + " \n \n"
            
        complete_response = full_response + " \n \n" + reference
        message_placeholder.markdown(full_response)
        with st.expander("References"):
            st.markdown(reference)
        st.success('Searched papers and here is the answer! RAG can also provide false/incomplete answers, please verify!') 

    st.session_state.messages.append((prompt, full_response, reference))   

