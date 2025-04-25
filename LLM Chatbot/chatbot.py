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



    def __init__(self, model:str="deepseek-r1:14b", vector_db="vector-db-1", host="localhost", port=6333, user="Ganesh", chat_name="chat_trial_1"):

        self.qdrant_client = CustomQdrantClient(embedding_model=model, host=host, port=port, collection_name=vector_db)

        self.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
        self.chat_storage_path = os.path.join("../User chats", user)
        os.makedirs(self.chat_storage_path, exist_ok=True)
        self.chat_storage_path = os.path.join(self.chat_storage_path, chat_name.strip() + '.json')

        if os.path.exists(self.chat_storage_path):
            self.load_messages()
        else:
            self.save_messages()

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

        self.language_model = OllamaLLM(model=model, streaming=True)
        self.streamer = Streamer(save_reasoning=False)
    


    def load_messages(self):

        with open(self.chat_storage_path, 'r') as f:
            memory_dict = json.load(f)
            self.memory.model_construct(memory_dict)

        return


    def save_messages(self):
        
        with open(self.chat_storage_path, 'w') as file:
            json.dump(self.memory.model_dump(), file, indent=4)
        
        return


    
    def retrieve_context_docs(self, user_input, stream=False):

        history_data = self.memory.load_memory_variables({})["history"]
        response_chain = self.query_prompt_template | self.language_model
        response = ""

        self.streamer.clear_buffer()
        for token in response_chain.stream({"user_input": user_input, "history": history_data}):
            self.streamer.stream_to_buffer(token, stream=stream)
        response = self.streamer.return_buffer()
        context_docs = self.qdrant_client.similarity_search(response)

        return context_docs
    


    def generate_chatbot_response(self, user_input, stream=True):

        context_docs = self.retrieve_context_docs(user_input)
        context_str = f"\n\nContext information:\n" + "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
        history_data = self.memory.load_memory_variables({})["history"]
        response_chain = self.chat_prompt_template | self.language_model
        response = ""

        self.streamer.clear_buffer()
        for token in response_chain.stream({"user_input": user_input, "context_str": context_str, "history": history_data}):
           self.streamer.stream_to_buffer(token, stream=stream)
        response = self.streamer.return_buffer()
        
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)
        self.save_messages()

        return response