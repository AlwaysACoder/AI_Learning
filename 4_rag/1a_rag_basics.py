import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_location = os.getenv("EMBEDDING_MODEL_FILEPATH")
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path, encoding='utf-8')  # or 'latin-1' if utf-8 fails
    documents = loader.load()

    # Split the document into chunks
    # Recommended Chunk Size and Overlap for sentence-transformers/all-MiniLM-L6-v2
    # Chunk Size: 256–384 tokens
    # Chunk Overlap: 20%–25% of chunk size (usually 50–100 tokens)

    text_splitter = CharacterTextSplitter(
        separator=" ",           # Split on paragraph boundaries (or "\n" for line)
        chunk_size=256,
        chunk_overlap=64,
        length_function=len         # Default is len(), which measures characters
    )



    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    # Define the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_location,
        model_kwargs={"local_files_only": True}
    )
    
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
