from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()

    st.set_page_config(page_title="Ask Your PDF", page_icon="🗂️", layout="wide")

    with st.sidebar:
        st.title("Project Info 📝")
        st.markdown("""
        **QueryMyDocs 💬**  
        This web app allows you to upload a PDF file and ask questions about its contents.  
        It utilizes advanced AI models for PDF text extraction and question answering.
        
        - **PDF Extraction**: Extracts text from your PDF.
        - **Embeddings**: Uses HuggingFace embeddings - sentence-transformers/all-MiniLM-L6-v2.
        - **LLM**: Azure OpenAI GPT for generating responses - gpt-35-turbo.
        """)


    # Main content area
    st.title("Ask Your PDF 💬")
    st.markdown("<style> body { background-color: white; } </style>", unsafe_allow_html=True)

    # Upload the PDF file
    pdf = st.file_uploader("📄 Upload your PDF", type=["pdf"])

    # Extract text from the uploaded file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Create chunks of text
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings using HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Build the FAISS vector store from chunks
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_input = st.text_input("🧐 Ask a question about your PDF:")
        if user_input:
            docs = knowledge_base.similarity_search(user_input)

            llm = AzureChatOpenAI(
                temperature=0,
                openai_api_base="your_endpoint_url",
                openai_api_key="your_azure_openai_api_key",
                openai_api_version="2023-03-15-preview",
                deployment_name="gpt-35-turbo"
            )

            # Load the QA chain with the LLM
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_input)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
