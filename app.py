from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import streamlit as st

import langchain
langchain.verbose = False

# Insert your API key here (Avoid sharing your code with the API key exposed)
OPENAI_API_KEY = "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"


# Define a fixed file path for the .txt file
TEXT_FILE_PATH = "test-data.txt"

# Process text from the fixed .txt file
def process_text(text):
    # Split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


def main():
    st.title("Hi, I am JARVIS. How can I assist you?")

    # Read the text file content directly
    with open(TEXT_FILE_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    # Create a knowledge base object
    knowledgeBase = process_text(text)

    query = st.text_input('Ask a question about the text file...')

    cancel_button = st.button('Cancel')

    if cancel_button:
        st.stop()

    if query:
        docs = knowledgeBase.similarity_search(query)

        llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, max_tokens=1000, temperature=0.7)

        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cost:
            response = chain.invoke(input={"question": query, "input_documents": docs})
            print(cost)

            st.write(response["output_text"])


if __name__ == "__main__":
    main()

