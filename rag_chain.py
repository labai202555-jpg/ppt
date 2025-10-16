from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import uuid
import os
import httpx
client = httpx.Client(verify=False)

import os

tiktoken_cache_dir = "C://Users//GenAICHNSIRUSR19//Desktop//Hackathon//restaurant_chatbot//token//tiktoken_cache//"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

# 1. Load document
loader = TextLoader("C:/Users/GenAICHNSIRUSR19/Desktop/Hackathon/restaurant_chatbot/data/Hotel_Restaurant_Reviews_Dataset_Full.txt")
docs = loader.load()

# 2. Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# 3. Create embeddings and vector store

embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-sMJ3EiUlIPvsL1Uv3WSvFw",  # Replace with your actual API key
    http_client=client
)
#embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
vectorstore = Chroma(
    embedding_function=embeddings
)
# 4. Create retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm=ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-35-turbo",
    api_key="sk-sMJ3EiUlIPvsL1Uv3WSvFw",  # Replace with your actual API key
    http_client=client
)
### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def save_message(session_id, input_text):
    
    file_name = f"file_{session_id}.txt"
    with open(file_name, 'a') as file:
        file.write(input_text)

# Invoke the chain and save the messages after invocation
def invoke_and_save(session_id, input_text):
    # Save the user question with role "human"
    save_message(session_id, input_text)
    
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    # Save the AI answer with role "ai"
    save_message(session_id, result)
    return result


