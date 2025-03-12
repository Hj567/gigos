from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

import streamlit as st
from dotenv import load_dotenv
import os

import langchain
langchain.verbose = False

# Load environment variables
load_dotenv()

# Access API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define a fixed file path for the .txt file
TEXT_FILE_PATH = "test-data.txt"

# Process text from the fixed .txt file
def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base


def main():
    st.title("📜 Baggage Rules Assistant – Ask Me Anything")

    with open(TEXT_FILE_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    # Initialize knowledge base
    knowledge_base = process_text(text)

    # Setup Chat LLM
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.7)

    # Set up memory buffer for multi-turn conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup Conversational QA Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=knowledge_base.as_retriever(),
        memory=memory,
        verbose=False
    )

    # Session state for storing conversation history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("💬 Ask your question about baggage rules:")

    cancel_button = st.button("❌ Clear Chat")

    if cancel_button:
        st.session_state.chat_history = []
        st.experimental_rerun()

    if query:
        with get_openai_callback() as cost:
            response = qa_chain.run(query)
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Jarvis", response))
            print(cost)

    # Display chat history
    st.markdown("### 🧠 Chat History")
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑 {sender}:** {message}")
        else:
            st.markdown(f"**🤖 {sender}:** {message}")


if __name__ == "__main__":
    main()
