import argparse
from qdrant_client import QdrantClient

def drop_collection(host="localhost", port=6333, name: str = "vector-db-1"):
    
    """
    Deletes a collection from the Qdrant vector database.

    Parameters:
    ----------
    name : str, optional
        The name of the collection to delete. Defaults to "vector-db-1".

    Description:
    -----------
    - Connects to a Qdrant instance running locally on port 6333.
    - Attempts to delete the specified collection.
    - Prints a success message if deletion succeeds.
    - Prints the error message if deletion fails.
    """

    # Connect to Qdrant instance
    qdrant = QdrantClient(host=host, port=port)

    try:
        # Delete the specified collection
        qdrant.delete_collection(collection_name=name)
        print(f"Collection '{name}' dropped successfully!")
    except Exception as e:
        print(f"Error deleting collection '{name}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop a Qdrant collection given its name.")
    parser.add_argument(
        "--name", 
        type=str, 
        default="vector-db-1", 
        help="Name of the Qdrant collection to drop (default: 'vector-db-1')"
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
    drop_collection(host=args.host, port=args.port,name=args.name)

