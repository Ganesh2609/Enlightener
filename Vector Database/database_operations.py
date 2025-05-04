import uuid
import re
from typing import List, Dict, Any, Tuple, Optional
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models


class CustomQdrantClient:
    """
    An enhanced client for interacting with Qdrant vector database using LangChain embeddings.
    This class provides improved methods for adding documents to collections, performing semantically
    rich similarity searches, and managing documents within the vector database.
    """

    def __init__(self, 
                 embedding_model="deepseek-r1:14b", 
                 host="localhost", 
                 port=6333, 
                 collection_name="vector-db-1", 
                 splitter_chunk_size=1000,       # Reduced from 3000 
                 splitter_chunk_overlap=200,     # Adjusted for shorter chunks
                 returned_docs=30,               # Increased from 16
                 similarity_cutoff=0.35):        # Lowered threshold
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
            similarity_cutoff (float): The minimum similarity score for results.
        """
        
        self.embedding_model = OllamaEmbeddings(model=embedding_model)
        self.qdrant = QdrantClient(host=host, port=port)
        self.text_processor = RecursiveCharacterTextSplitter(
            chunk_size=splitter_chunk_size, 
            chunk_overlap=splitter_chunk_overlap, 
            add_start_index=True
        )
        self.collection_name = collection_name
        self.returned_docs = returned_docs
        self.similarity_cutoff = similarity_cutoff
    
    def add_to_collection(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Add documents to the Qdrant collection with enhanced metadata extraction.
        
        Args:
            data (List[Dict[str, str]]): A list of dictionaries containing text and path information.
                Each dictionary should have 'text' key and optionally a 'path' key.
                
        Returns:
            List[Dict[str, str]]: A list of IDs and paths for the added documents.
            
        Raises:
            Exception: If there's an error inserting documents into the collection.
        """
        points = []
        ids = []
        
        for item in data:
            doc = Document(page_content=item.get('text', ''))
            path = item.get('path', '')
            
            # Extract quote, explanation, and keywords if available
            quote, explanation, keywords = self._extract_quote_explanation_keywords(doc.page_content)
            
            # Process the document into chunks
            text_data = self.text_processor.split_documents([doc])
            text_data = [text.page_content for text in text_data]
            
            # Generate embeddings for each chunk
            embeddings = self.embedding_model.embed_documents(text_data)
            
            # Process each chunk
            for i in range(len(embeddings)):
                unique_id = str(uuid.uuid4())
                ids.append({'id': unique_id, 'path': path})
                
                # Enhanced payload with structured data
                payload = {
                    "text": text_data[i],
                    "path": path,
                    "chunk_index": i,
                    "total_chunks": len(text_data)
                }
                
                # Add quote/explanation metadata if available (for first chunk or all chunks)
                if quote:
                    payload["quote"] = quote
                if explanation:
                    payload["explanation"] = explanation
                if keywords:
                    payload["keywords"] = keywords
                
                # Extract practical terms for improved search
                practical_terms = self._extract_practical_terms(text_data[i])
                if practical_terms:
                    payload["practical_terms"] = practical_terms
                
                # Create point with enhanced payload
                points.append({
                    "id": unique_id,
                    "vector": embeddings[i],
                    "payload": payload
                })

        try:
            # Insert points to Qdrant
            response = self.qdrant.upsert(
                collection_name=self.collection_name, 
                points=points
            )
            print(f"Successfully added {len(points)} document(s) to {self.collection_name}!")
        except Exception as e:
            print(f"Error inserting to collection '{self.collection_name}': {e}")

        return ids
    
    def similarity_search(self, query_vector: str) -> List[str]:
        """
        Perform an improved two-pass similarity search in the Qdrant collection.
        
        Args:
            query_vector (str): The query text to search for.
            
        Returns:
            List[str]: A list of text from the most relevant documents found.
        """
        # Embed the query
        embedded_query = self.embedding_model.embed_query(query_vector)
        
        try:
            # First pass: Get more candidates than needed
            initial_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=embedded_query,
                limit=self.returned_docs * 2,  # Get twice as many for re-ranking
                score_threshold=self.similarity_cutoff,
                with_payload=True,
                with_vectors=False,
            )
            
            if not initial_results:
                return []
            
            # Extract keywords from the query for additional matching
            query_keywords = self._extract_keywords(query_vector)
            
            # Second pass: Re-rank based on multiple factors
            reranked_results = []
            
            for result in initial_results:
                # Base vector similarity score (0-1)
                base_score = result.score
                
                # Extract text content and other payload data
                text_content = result.payload.get("text", "")
                explanation = result.payload.get("explanation", "")
                quote = result.payload.get("quote", "")
                keywords = result.payload.get("keywords", "")
                practical_terms = result.payload.get("practical_terms", [])
                
                # Content to search against - prioritize explanation if available
                search_content = explanation if explanation else text_content
                
                # Calculate keyword overlap score (0-1)
                keyword_score = self._calculate_keyword_match(query_keywords, search_content)
                
                # Calculate practical relevance score based on emotional and situational terms (0-1)
                practical_score = self._calculate_practical_relevance(query_vector, search_content)
                
                # Special boost for quote matches
                quote_match_score = 0.0
                if quote and any(kw.lower() in quote.lower() for kw in query_keywords if len(kw) > 3):
                    quote_match_score = 0.2
                
                # Special boost for keyword field matches
                keyword_field_score = 0.0
                if keywords and any(kw.lower() in keywords.lower() for kw in query_keywords if len(kw) > 3):
                    keyword_field_score = 0.1
                
                # Composite score with weightings
                composite_score = (
                    (base_score * 0.5) + 
                    (keyword_score * 0.15) + 
                    (practical_score * 0.15) + 
                    (quote_match_score) +
                    (keyword_field_score)
                )
                
                reranked_results.append((result, composite_score))
            
            # Sort by composite score and return
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            return [result[0].payload["text"] for result in reranked_results[:self.returned_docs]]
        
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
    
    def delete_by_id(self, ids: List[str]) -> None:
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
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids)
            )
            print("Documents deleted successfully!")
        except Exception as e:
            print(f"Error deleting documents: {e}")
    
    def _extract_quote_explanation_keywords(self, text: str) -> Tuple[str, str, str]:
        """
        Extract quote, explanation, and keywords from formatted text.
        
        Args:
            text (str): Input text to extract structured data from.
            
        Returns:
            Tuple[str, str, str]: Extracted quote, explanation, and keywords.
        """
        quote = ""
        explanation = ""
        keywords = ""
        
        # Extract quote section
        quote_match = re.search(r'QUOTE:\s*(.*?)(?=EXPLANATION:|KEYWORDS:|$)', text, re.DOTALL)
        if quote_match:
            quote = quote_match.group(1).strip()
        
        # Extract explanation section
        explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=KEYWORDS:|$)', text, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        # Extract keywords section
        keywords_match = re.search(r'KEYWORDS:\s*(.*?)(?=$)', text, re.DOTALL)
        if keywords_match:
            keywords = keywords_match.group(1).strip()
        
        return quote, explanation, keywords
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.
        
        Args:
            text (str): Input text to extract keywords from.
            
        Returns:
            List[str]: List of extracted keywords.
        """
        # Remove common stopwords and punctuation
        stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", 
                     "like", "through", "over", "before", "between", "after", "since", "without", "under", 
                     "within", "along", "following", "across", "behind", "beyond", "plus", "except", "but", 
                     "up", "down", "off", "above", "below", "use", "using", "used", "am", "is", "are", "was", 
                     "were", "be", "being", "been", "have", "has", "had", "do", "does", "did", "will", "would", 
                     "shall", "should", "may", "might", "must", "can", "could", "i", "you", "he", "she", "it", 
                     "we", "they", "me", "him", "her", "us", "them"]
        
        # Normalize text
        text = text.lower()
        
        # Replace punctuation with spaces
        for char in ",.!?;:()[]{}-\"'`":
            text = text.replace(char, " ")
        
        # Split into words and filter
        words = text.split()
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords

    def _extract_practical_terms(self, text: str) -> List[str]:
        """
        Extract practical and emotional terms from text that might be useful for search.
        
        Args:
            text (str): Input text to extract practical terms from.
            
        Returns:
            List[str]: List of practical and emotional terms.
        """
        # Lists of terms indicating practical issues and emotional states
        practical_terms = [
            "work", "job", "career", "family", "relationship", "marriage", "health",
            "focus", "concentration", "stress", "anxiety", "depression", "worry",
            "fear", "anger", "sadness", "conflict", "problem", "challenge", "difficult",
            "struggle", "balance", "peace", "happiness", "fulfillment", "purpose",
            "meaning", "direction", "goal", "achievement", "success", "failure",
            "meditation", "practice", "spiritual", "growth", "development", "progress",
            "mind", "body", "soul", "spirit", "heart", "emotions", "thoughts",
            "karma", "dharma", "devotion", "surrender", "service", "compassion",
            "love", "forgiveness", "acceptance", "patience", "discipline"
        ]
        
        # Convert to lowercase for comparison
        text_lower = text.lower()
        
        # Find matches
        found_terms = []
        for term in practical_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _calculate_keyword_match(self, query_keywords: List[str], document_text: str) -> float:
        """
        Calculate a score based on keyword overlap between query and document.
        
        Args:
            query_keywords (List[str]): Keywords extracted from the query.
            document_text (str): Document text to check for keyword matches.
            
        Returns:
            float: Score between 0 and 1 representing keyword match quality.
        """
        document_text = document_text.lower()
        
        # Count how many query keywords appear in the document
        matches = sum(1 for keyword in query_keywords if keyword in document_text)
        
        if not query_keywords:
            return 0.0
        
        # Calculate match percentage
        match_percentage = matches / len(query_keywords)
        return match_percentage

    def _calculate_practical_relevance(self, query: str, document_text: str) -> float:
        """
        Calculate a relevance score based on practical/emotional terms.
        
        Args:
            query (str): Original query text.
            document_text (str): Document text to analyze.
            
        Returns:
            float: Score between 0 and 1 representing practical relevance.
        """
        # Lists of terms indicating practical issues and emotional states
        practical_terms = [
            "work", "job", "career", "family", "relationship", "marriage", "health",
            "focus", "concentration", "stress", "anxiety", "depression", "worry",
            "fear", "anger", "sadness", "conflict", "problem", "challenge", "difficult",
            "struggle", "balance", "peace", "happiness", "fulfillment", "purpose",
            "meaning", "direction", "goal", "achievement", "success", "failure",
            "meditation", "practice", "spiritual", "growth", "development", "progress",
            "mind", "body", "soul", "spirit", "heart", "emotions", "thoughts",
            "karma", "dharma", "devotion", "surrender", "service", "compassion",
            "love", "forgiveness", "acceptance", "patience", "discipline"
        ]
        
        # Convert to lowercase for comparison
        query_lower = query.lower()
        document_lower = document_text.lower()
        
        # Count practical terms in query
        query_practical_count = sum(1 for term in practical_terms if term in query_lower)
        
        # Count matching practical terms in document
        matching_practical_count = sum(1 for term in practical_terms 
                                     if term in query_lower and term in document_lower)
        
        # Calculate match percentage
        if query_practical_count == 0:
            return 0.5  # Neutral score if no practical terms in query
        
        return matching_practical_count / query_practical_count