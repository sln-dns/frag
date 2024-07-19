import os
import logging
import streamlit as st
from dotenv import load_dotenv
import shutil
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Global configuration variables
openai_api_key = os.getenv("OPENAI_API_KEY")
local_model = "gpt-4o"
embedding_model = "text-embedding-3-large"
persist_directory = "db-streamlit"
collection_name = "documents"

# Setup models and objects
embedding = OpenAIEmbeddings(model=embedding_model)
llm = ChatOpenAI(model=local_model, temperature=0, api_key=openai_api_key)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant to an artificial intelligence language model. Your job is to generate five
    different versions of a given user question to retrieve relevant documents from the
    vector database. By generating different perspectives on the user's question, your
    goal is to help the user overcome some of the limitations of distance-based similarity searches
    similarity search. Give these alternative questions separated by newlines.
    Original question: {question}""",
)

# Global variables for vector database and chain
vector_db = None
chain = None

def initialize_vector_db():
    """Initialize the vector database"""
    global vector_db
    try:
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory, exist_ok=True)
        vector_db = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embedding)
        logging.info("Vector database loaded or created.")
    except Exception as e:
        logging.error(f"Error initializing vector database: {e}")

def initialize_chain():
    """Initialize the chain for processing questions"""
    global chain
    try:
        if vector_db is None:
            initialize_vector_db()
        
        if vector_db is not None:
            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(),
                llm,
                prompt=QUERY_PROMPT
            )
            template = """Answer the question based ONLY on the following context:
            {context}
            Question: {question}"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            logging.info("Chain initialized successfully.")
        else:
            logging.error("Failed to initialize vector database.")
    except Exception as e:
        logging.error(f"Error initializing chain: {e}")

def load_file_to_db(uploaded_file):
    """Load file into the database"""
    try:
        logging.info("Starting file upload to database")
        file_path = os.path.join(persist_directory, uploaded_file.name)
        logging.info(f"Saving uploaded file to: {file_path}")
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist after uploading.")
        
        loader = PyPDFLoader(file_path=file_path)
        data = loader.load()
        logging.info(f"File {file_path} loaded")

        for doc in data:
            doc.metadata['source'] = file_path
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)
        logging.info(f"File split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            logging.info(f"Chunk {i + 1}: {chunk.page_content[:100]}... (Metadata: {chunk.metadata})")

        if vector_db is None:
            logging.info("vector_db not initialized, attempting initialization...")
            initialize_vector_db()
            logging.info("vector_db initialized")

        if vector_db is not None:
            logging.info("Adding chunks to vector database")
            vector_db.add_documents(chunks)
            logging.info("PDF file processed and added to vector database")
            st.success("PDF file processed and added to vector database")
            if uploaded_file.name not in st.session_state['uploaded_files']:
                st.session_state['uploaded_files'].append(uploaded_file.name)
        else:
            logging.error("Failed to initialize vector database")
            st.error("Failed to initialize vector database")
    except Exception as e:
        logging.error(f"Error processing PDF file: {e}")
        st.error(f"Error uploading file: {e}")

def delete_all_docs_from_db():
    """Delete all files from the database"""
    try:
        logging.info("Starting deletion of all files from the database")
        
        if vector_db is None:
            logging.info("vector_db not initialized, attempting initialization...")
            initialize_vector_db()
            logging.info("vector_db initialized")

        if vector_db is not None:
            logging.info("Attempting to delete all documents from collection")
            all_docs = vector_db._collection.get()
            ids = all_docs['ids']
            delete_result = vector_db._collection.delete(ids=ids)
            logging.info(f"Result of deleting all documents from vector database: {delete_result}")

            for file_name in st.session_state['uploaded_files']:
                pdf_path = os.path.join(persist_directory, file_name)
                logging.info(f"PDF file path: {pdf_path}")

                if os.path.exists(pdf_path):
                    logging.info(f"File {pdf_path} exists, attempting deletion")
                    os.remove(pdf_path)
                    logging.info(f"File {pdf_path} deleted")
                else:
                    logging.warning(f"File {pdf_path} does not exist")

            st.session_state['uploaded_files'] = []
            logging.info("All files and their vectors have been deleted from the database.")
            st.success("All files and their vectors have been deleted from the database.")
            st.experimental_rerun()  # Rerun the script to refresh the page
        else:
            logging.error("Failed to initialize vector database.")
            st.error("Failed to initialize vector database.")
    except Exception as e:
        logging.error(f"Error deleting all files from the database: {e}")
        st.error(f"Error deleting all files: {e}")

def handle_question():
    """Handle user's question"""
    try:
        question = st.session_state['question_input']
        if chain is not None:
            result = chain.invoke(question)
            logging.info(f"Response to user: {result}")
            st.session_state['response'] = f"Question: {question}\n\nAnswer: {result}"
            st.session_state['question_input'] = ""  # Clear input field
        else:
            st.warning("No chain available for processing questions.")
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        st.error(f"Error processing question: {e}")

# Streamlit interface
st.set_page_config(layout="wide")

# Sidebar for file operations
with st.sidebar:
    st.header("File Operations")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        load_file_to_db(uploaded_file)
        if 'uploaded_files' not in st.session_state:
            st.session_state['uploaded_files'] = [file for file in os.listdir(persist_directory) if file.endswith('.pdf')]

    st.subheader("Uploaded Files")
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = [file for file in os.listdir(persist_directory) if file.endswith('.pdf')]
    st.session_state['uploaded_files'] = list(set(st.session_state['uploaded_files']))  # Remove duplicates
    for file in st.session_state['uploaded_files']:
        st.write(file)

    if st.button("Delete All Files"):
        delete_all_docs_from_db()

# Main content
st.title("Document QA System")

# Initialize the chain
initialize_chain()

# Question input and answer display
if 'question_input' not in st.session_state:
    st.session_state['question_input'] = ""

question_input = st.text_input("Your question here:", key='question_input')
if st.button("Submit Question", on_click=handle_question):
    pass

# Display response
if 'response' in st.session_state and st.session_state['response']:
    st.write(st.session_state['response'])
