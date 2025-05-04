import sys
sys.path.append('../Vector Database')

import os
import re
import json
from database_operations import CustomQdrantClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferMemory
from streamer import Streamer


class Chatbot:
    """
    Enhanced RAG-based spiritual chatbot that retrieves and delivers Amma's teachings
    with improved semantic search and context relevance.
    """

    def __init__(self, model:str="deepseek-r1:14b", vector_db="vector-db-1", host="localhost", port=6333, user="Ganesh", chat_name="chat_trial_1"):
        """
        Initialize the chatbot with the specified parameters.
        
        Args:
            model (str): The language model to use
            vector_db (str): The name of the vector database collection
            host (str): The hostname of the Qdrant server
            port (int): The port number of the Qdrant server
            user (str): The username for chat storage
            chat_name (str): The name of the chat session
        """
        # Initialize the vector database client with improved parameters
        self.qdrant_client = CustomQdrantClient(
            embedding_model=model, 
            host=host, 
            port=port, 
            collection_name=vector_db,
            splitter_chunk_size=1000,
            splitter_chunk_overlap=200,
            returned_docs=30,
            similarity_cutoff=0.35
        )

        # Set up conversation memory
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        self.chat_storage_path = os.path.join("../User chats", user)
        os.makedirs(self.chat_storage_path, exist_ok=True)
        self.chat_storage_path = os.path.join(self.chat_storage_path, chat_name.strip() + '.json')

        # Load existing chat or create new one
        if os.path.exists(self.chat_storage_path):
            self.load_messages()
        else:
            self.save_messages()

        # Load prompt templates
        with open('query_vector_prompt.txt', 'r') as f:
            query_prompt = f.read()
            self.query_prompt_template = ChatPromptTemplate.from_messages([
                ("system", query_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{user_input}")
            ])

        with open('system_prompt.txt', 'r') as f:
            system_prompt = f.read()
            self.chat_prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt + "\n\nContext information:\n{context_str}"),
                MessagesPlaceholder(variable_name="history"),
                ("user", "{user_input}")
            ])

        # Initialize the language model and streamer
        self.language_model = OllamaLLM(model=model, streaming=True)
        self.streamer = Streamer(save_reasoning=False)
    
    def load_messages(self):
        """Load conversation history from storage"""
        with open(self.chat_storage_path, 'r') as f:
            memory_dict = json.load(f)
            self.memory.model_construct(memory_dict)

    def save_messages(self):
        """Save conversation history to storage"""
        with open(self.chat_storage_path, 'w') as file:
            json.dump(self.memory.model_dump(), file, indent=4)
    
    def retrieve_context_docs(self, user_input, stream=False):
        """
        Generate an optimized query and retrieve relevant context documents.
        
        Args:
            user_input (str): The user's input
            stream (bool): Whether to stream the output
            
        Returns:
            list: Retrieved context documents
        """
        # Get conversation history
        history_data = self.memory.load_memory_variables({})["history"]
        
        # Create the query reformulation chain
        response_chain = self.query_prompt_template | self.language_model
        
        # Generate the optimized search query
        self.streamer.clear_buffer()
        for token in response_chain.stream({"user_input": user_input, "history": history_data}):
            self.streamer.stream_to_buffer(token, stream=stream)
        optimized_query = self.streamer.return_buffer()
        
        # Log the optimized query for debugging if needed
        print(f"Optimized query: {optimized_query[:100]}..." if len(optimized_query) > 100 else optimized_query)
        
        # Retrieve relevant documents using improved search
        context_docs = self.qdrant_client.similarity_search(optimized_query)
        
        return context_docs
    
    def rank_document_relevance(self, docs, user_input):
        """
        Rank the relevance of retrieved documents to the user's question.
        
        Args:
            docs (list): List of retrieved documents
            user_input (str): The user's input
            
        Returns:
            list: List of documents with relevance scores
        """
        # Simple keyword-based relevance scoring
        keywords = re.findall(r'\b\w{4,}\b', user_input.lower())
        
        scored_docs = []
        for i, doc in enumerate(docs):
            # Count keyword occurrences
            score = sum(1 for kw in keywords if kw in doc.lower())
            
            # Check for quote-explanation format and give bonus points
            if "QUOTE:" in doc and "EXPLANATION:" in doc:
                score += 2
            
            # Check for emotional or practical terms that match the query
            emotional_terms = ["peace", "happiness", "love", "fear", "anxiety", "stress", 
                              "balance", "harmony", "conflict", "struggle", "challenge"]
            for term in emotional_terms:
                if term in user_input.lower() and term in doc.lower():
                    score += 1
            
            # Final score includes position in results (earlier = better)
            final_score = score - (i * 0.1)  # Slight penalty for later positions
            scored_docs.append((doc, final_score))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked documents
        return [doc for doc, score in scored_docs]
    
    def generate_chatbot_response(self, user_input, stream=True):
        """
        Generate a response to the user's input using retrieved context documents.
        
        Args:
            user_input (str): The user's input
            stream (bool): Whether to stream the output
            
        Returns:
            str: The chatbot's response
        """
        # Retrieve and rank context documents
        context_docs = self.retrieve_context_docs(user_input)
        ranked_docs = self.rank_document_relevance(context_docs, user_input)
        
        # Format context for the prompt
        context_str = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(ranked_docs[:10])])
        
        # Get conversation history
        history_data = self.memory.load_memory_variables({})["history"]
        
        # Create the response generation chain
        response_chain = self.chat_prompt_template | self.language_model
        
        # Generate the response
        self.streamer.clear_buffer()
        for token in response_chain.stream({
            "user_input": user_input, 
            "context_str": context_str, 
            "history": history_data
        }):
           self.streamer.stream_to_buffer(token, stream=stream)
        
        response = self.streamer.return_buffer()
        
        # Update conversation memory
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)
        self.save_messages()
        
        return response