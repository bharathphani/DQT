# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import chromadb
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tiktoken

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


# Load document
def load_document(file):
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file)
    data = loader.load()
    return data


# Chunking Data
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    data_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = data_splitter.split_documents(data)
    return chunks


def calc_embeddings_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


def create_embeddings_with_chroma(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key='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')
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def doc_query_tool(vector_store, q, k=3):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1,
                     openai_api_key='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')
    retriever = vector_store.as_retriever(search_type='similarity', kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.run(q)
    return answer


def doc_query_tool_with_memory(vector_store, q, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1,
                     openai_api_key='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')
    retriever = vector_store.as_retriever(search_type='similarity', kwargs={'k': 3})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': q, 'chat_history': chat_history})
    chat_history.append((q, result['answer']))
    return result, chat_history


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_dotenv(find_dotenv(), override=True)
    st.image('doc1.jpeg', width=500)
    st.subheader('Document Query tool - Answers your questions in Natural Language')
    with st.sidebar:
        # api_key = st.text_input('Open API Key', type='password')
        # if api_key:
        # os.environ['OPEN_API_KEY'] = api_key
        uploaded_file = st.file_uploader('Upload a file', type=['pdf'])
        chunk_size = st.number_input('Chunk Size', min_value=100, max_value=2048, value=512, key='cs')
        k = st.number_input('K', min_value=1, max_value=20, value=3, key='kvalue')
        overlap = st.number_input('Overlap', min_value=20, max_value=80, value=20, key='chunkoverlap')
        add_data = st.button('Add Data')

        if uploaded_file and add_data:
            with st.spinner('Loading and indexing the doc .....'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', 'temp_files', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size, overlap)
                st.write(f'chunk size : {chunk_size}, chunks : {len(chunks)}')
                tokens, embed_cost = calc_embeddings_cost(chunks)
                st.write(f'Embeddings cost for tokens {tokens}  : {embed_cost:.4f}')
                vector_store = create_embeddings_with_chroma(chunks)
                st.session_state.vs = vector_store
                st.success('Loaded and indexing is successful')

    question = st.text_input('Ask a question about the content of the document :')
    ans = ''
    if question:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            # k is the no that llm use most similar chunks for the final answer
            st.write(f'k ; {k}')
            ans = doc_query_tool(vector_store, question, k)
            st.text_area('LLM Answer :', ans)
    st.divider()
    if 'history' not in st.session_state:
        st.session_state['history'] = ''
    if not st.session_state['history'].__contains__(question):
        value = f' Q: {question} \n A: {ans}'
        st.session_state['history'] = f'{value} \n {"*" * 100} \n {st.session_state.history}'
        st.text_area(label='Chat History', value=st.session_state.history, key='history', height=400)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
