"""
Question 1 QnA with Streamlit
"""

# Import all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, ChatMessage, AIMessage
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from PIL import Image, ImageDraw
import re
import tiktoken

##########
# Vector Store
##########

def load_document(file):
    """
    Load a document from a file and return the text content.
    """
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """
    Split the data into chunks of a given size and overlap.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    print(f'Chunk Size: {chunk_size}, Number of chunks: {len(chunks)}')
    return chunks

# create embeddings and store in chroma db
def create_embeddings(chunks, embeddings):
    """
    Create embeddings for the given chunks and store them in a vector store.
    """
    
    # store in chroma db
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    vector_store.persist()
    return vector_store

# load chroma db
def load_chroma_db(embeddings):
    """
    Load the vector store from the disk.
    """
    vector_store = Chroma(persist_directory="./mychroma_db", embedding_function=embeddings)
    return vector_store

# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    embedding_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {embedding_tokens}')
    # text-embedding-3-small	$0.02 / 1M tokens
    print(f'Embedding Cost in USD: {embedding_tokens / 1000000 * 0.0002:.6f}')
    return embedding_tokens

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, 
    model='text-embedding-3-small'
    )

# load all pdfs in the directory
pdf_list = [f for f in os.listdir('.') if f.endswith('.pdf')]
load_pdf = False

# check if vector store file path exists
if os.path.exists('./mychroma_db'):
    # load the vector store from the disk
    vector_store = load_chroma_db(embeddings)
    # check if the vector store is loaded
    if vector_store._collection.count() > 0:
        print('Vector store loaded successfully')
    else:
        load_pdf = True
else:
    load_pdf = True

# load the pdf files if needed
if load_pdf:
    for file in pdf_list:
        # Load a PDF document and split it into sections
        data = load_document(file)

        # Split the data into chunks
        chunks = chunk_data(data, chunk_size=1024, chunk_overlap=256)

        # print the cost of the embeddings
        embedding_tokens = calculate_embedding_cost(chunks)

        # Create embeddings
        vector_store = create_embeddings(chunks, embeddings)

        print('###'*20)

    print(f'{vector_store._collection.count()} chunks of documents loaded in the vector store') 
else:
    print(f'Vector store already loaded with {vector_store._collection.count()} chunks of documents')

##########
# OpenAI Chat Model
##########

def num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model('gpt-4-0125-preview')
    return len(enc.encode(text))

def message_list_info(user_query, vector_store):

    # Retrieval from the vector database, only get 2 most relevant chunks
    docs = vector_store.similarity_search(user_query, k=2)
    context = "\nFrom vector database: \n"
    
    for i in range(len(docs)):
        context = context + f"Retrieval document [{i}]: \n" + docs[i].page_content + "\n"

    # set the system message with the retrieved context from the vector database
    system_message = f'''You are an expert economist.

Your task is to answer questions about the top 5 largest economy in the world - USA, China, Germany, Japan, India, and UK.

WARNING: Your max response length is 200 tokens. So, please keep your responses concise.

Please refer to retrieved context (RAG) about the economy by country from Wikipedia:
{context}
'''
    message_list = [
        SystemMessage(content=system_message)
    ]

    system_message_token = num_tokens(system_message)

    # Load the conversation history from a CSV file
    df_history = pd.read_csv('static/' + 'df_history.csv')
    history_token = 0

    # Append the conversation history to the conversation history
    if len(df_history) < 10:
        for _, row in df_history.sort_index().iterrows():
        # for last 10 messages to reduce cost
            if row['entity'] == 'user':
                message_list.append(
                    HumanMessage(content=row['message'])
                )
            else:
                message_list.append(
                    AIMessage(content=row['message'])
                )
            history_token += num_tokens(row['message'])

    else:
        for _, row in df_history.sort_index().tail(10).iterrows():
            if row['entity'] == 'user':
                message_list.append(
                    HumanMessage(content=row['message'])
                )
            else:
                message_list.append(
                    AIMessage(content=row['message'])
                )
            history_token += num_tokens(row['message'])


    # Append the latest user query to the conversation history
    message_list.append(
        HumanMessage(content=user_query)
    )
    user_query_token = num_tokens(user_query)
    
    token_dict = {
        'system_message_token': system_message_token,
        'history_token': history_token,
        'user_query_token': user_query_token
    }

    return message_list, user_query, token_dict

def append_to_history(ai_response_text, user_query):
    csv_file = 'static/df_history.csv'
    df_history = pd.read_csv(csv_file)

    df_history = pd.concat([
        df_history,
        pd.DataFrame({
            'entity': ['user', 'ai_copilot'],
            'message':[user_query, ai_response_text]
        })
    ], ignore_index=True)
    df_history.to_csv(csv_file, index=False)


def get_ai_response(open_ai_chat_model, user_query, vector_store):
    # Get the message list
    message_list, user_query, token_dict = message_list_info(user_query, vector_store)

    # Get the AI response
    openai_response = open_ai_chat_model(messages=message_list)
    ai_response_text = openai_response.content

    # Save the conversation history to a CSV file
    append_to_history(ai_response_text, user_query)

    gpt4_cost = [10/1e6, 30/1e6] # input, output cost per token
    # print the tokens
    print(f"System message token: {token_dict['system_message_token']}")
    print(f"History message token: {token_dict['history_token']}")
    print(f"User query token: {token_dict['user_query_token']}")
    print(f"AI response token: {num_tokens(ai_response_text)}")
    # print the cost
    print(f"System message cost: {token_dict['system_message_token'] * gpt4_cost[0]:.6f}")
    print(f"History message cost: {token_dict['history_token'] * gpt4_cost[0]:.6f}")
    print(f"User query cost: {token_dict['user_query_token'] * gpt4_cost[0]:.6f}")
    print(f"AI response cost: {num_tokens(ai_response_text) * gpt4_cost[1]:.6f}")

    print("###"*20)
    print(f"User Query: {user_query}\nAI Response: {ai_response_text}")

    return ai_response_text

##############
# WARNING! Only run this cell once to create the static folder and the history dataframe
# 
# Rerun this cell to clear the history dataframe
##############
# # create static folder
# if not os.path.exists('static'):
#     os.makedirs('static')
# # create df for history and save to csv
# df = pd.DataFrame(columns=['entity', 'message'])
# df.to_csv('static/df_history.csv', index=False)

# Initialize the OpenAI chat model
openai_chat_model = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model='gpt-4-0125-preview',
    temperature=0.5,
    max_tokens=350,
)

##########
# Streamlit App
##########

import streamlit as st
import time

st.title('Top 6 Largest Economies QnA with OpenAI')

def response_generator(ai_response_text):
    for word in ai_response_text.split():
        yield word + " "
        time.sleep(0.05)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me a question about the top 6 largest economies in the world"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = get_ai_response(openai_chat_model, prompt, vector_store)
        response = st.write_stream(response_generator(response))
    st.session_state.messages.append({"role": "assistant", "content": response})