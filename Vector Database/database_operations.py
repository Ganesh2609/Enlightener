import uuid
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict



class CustomQdrantClient:
    """
    A custom client for interacting with Qdrant vector database using LangChain embeddings.
    This class provides methods for adding documents to collections, performing similarity searches,
    and managing documents within the vector database.
    """

    def __init__(self, embedding_model="deepseek-r1:14b", host="localhost", port=6333, collection_name="vector-db-1", splitter_chunk_size=3000, splitter_chunk_overlap=400, returned_docs:int=16, similarity_cutoff:int=0):
        """
        Initialize the CustomQdrantClient with the specified parameters.
        
        Args:
            embedding_model (str): The name of the Ollama embedding model to use.
            host (str): The hostname of the Qdrant server.
            port (int): The port number of the Qdrant server.
            collection_name (str): The name of the collection to use.
            splitter_chunk_size (int): The size of chunks for text splitting.
            splitter_chunk_overlap (int): The overlap between chunks for text splitting.
            returned_docs (int): The number of documents to return in search results.
        """
        
        self.embedding_model = OllamaEmbeddings(model=embedding_model)
        self.qdrant = QdrantClient(host=host, port=port)
        self.text_processor = RecursiveCharacterTextSplitter(chunk_size=splitter_chunk_size, chunk_overlap=splitter_chunk_overlap, add_start_index=True)
        self.collection_name = collection_name
        self.returned_docs = returned_docs
        self.similarity_cutoff = similarity_cutoff
    

    def add_to_collection(self, data: List[Dict[str, str]]) -> List[int]:
        """
        Add documents to the Qdrant collection.
        
        Args:
            data (List[Dict[str, str]]): A list of dictionaries containing text and path information.
                Each dictionary should have 'text' key and optionally a 'path' key.
                
        Returns:
            List[int]: A list of IDs for the added documents.
            
        Raises:
            Exception: If there's an error inserting documents into the collection.
        """

        points = []
        ids = []
        
        for item in data:
            doc = Document(page_content=item.get('text'))
            path = item.get('path', '')
            text_data = self.text_processor.split_documents([doc])
            text_data = [text.page_content for text in text_data]
            embeddings = self.embedding_model.embed_documents(text_data)
            for i in range(len(embeddings)):
                unique_id = str(uuid.uuid4())
                ids.append({'id':unique_id, 'path':path})
                points.append({
                    "id": unique_id,
                    "vector": embeddings[i],
                    "payload": {"text": text_data[i], "path": path}
                })

        try:
            response = self.qdrant.upsert(collection_name=self.collection_name, points=points)
            print(f"Successfully added {len(points)} document(s) to {self.collection_name}!")
        except Exception as e:
            print(f"Error inserting to collection '{self.collection_name}': {e}")

        return ids
    

    def similarity_search(self, query_vector:str) -> List[str]:
        """
        Perform a similarity search in the Qdrant collection.
        
        Args:
            query_vector (str): The query text to search for.
            
        Returns:
            List[str]: A list of text from the most similar documents found.
            
        Raises:
            Exception: If there's an error performing the similarity search.
        """
        
        query_vector = self.embedding_model.embed_query(query_vector)

        try:
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.returned_docs,
                score_threshold=self.similarity_cutoff,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []

        return [result.payload["text"] for result in search_results]
    

    def delete_by_id(self, ids=List[str]) -> None:
        """
        Delete documents from the Qdrant collection by their IDs.
        
        Args:
            ids (List[str]): A list of document IDs to delete.
            
        Returns:
            None
            
        Raises:
            Exception: If there's an error deleting the documents.
        """

        try:
            self.qdrant.delete(
                collection_name = self.collection_name,
                points_selector = models.PointIdsList(points=ids)
            )
            print("Documents deleted successfully!")
        except Exception as e:
            print(f"Error deleting documents: {e}")

        return