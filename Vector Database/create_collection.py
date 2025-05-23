import argparse
from langchain.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

def create_collection(embedding_model="deepseek-r1:14b", host="localhost", port=6333, name: str = "vector-db-1"):

    """
    Creates a new vector collection in the Qdrant vector database using embeddings generated by the DeepSeek R1 model via Ollama.

    Parameters:
    ----------
    name : str, optional
        Name to use for the embedding model (currently unused in collection naming), by default "vector-db-1".

    Description:
    -----------
    - Initializes the OllamaEmbeddings object with the DeepSeek R1 14B model.
    - Connects to a local instance of Qdrant (assumes it is running on localhost:6333).
    - Creates a collection with the specified name, defaulting to "vector-db-1
    """

    # Initialize the embedding model
    embedding = OllamaEmbeddings(model=embedding_model)

    # Connect to Qdrant instance
    qdrant = QdrantClient(host=host, port=port)

    # Get the embedding dimension by embedding a sample string
    embedding_dimension = len(embedding.embed_query("test"))

    try:
        # Create a new collection in Qdrant
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE)
        )
        print(f"Collection '{name}' created successful!")
    except Exception as e:
        print(f"Error creating collection '{name}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Qdrant collection with DeepSeek R1 embeddings.")
    parser.add_argument(
        "--name", 
        type=str, 
        default="vector-db-1", 
        help="Name of the Qdrant collection to create (default: 'vector-db-1')"
    )
    parser.add_argument(
        "--embedding_model", 
        type=str, 
        default="deepseek-r1:14b", 
        help="Name of the embedding model used (default: 'deepseek-r1:14b'')"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Name of the host (default: 'localhost')"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6333, 
        help="Port number of the database container (default: 6333)"
    )
    args = parser.parse_args()
    create_collection(embedding_model=args.embedding_model, host=args.host, port=args.port,name=args.name)
    