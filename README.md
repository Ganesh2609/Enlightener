# Enlightener

## A RAG-Based Spiritual Chatbot for Amma's Teachings

Enlightener is an advanced Retrieval-Augmented Generation (RAG) chatbot that serves as a devoted spiritual guide, sharing the wisdom and teachings of Mata Amritanandamayi Devi (Amma). This system uses cutting-edge AI technologies to provide authentic spiritual guidance based on Amma's direct talks, books, and quotes.

## Features

- **Multi-Source Knowledge Base**: Processes text transcripts, PDF books, and quote images
- **Advanced OCR Technology**: Uses an ensemble approach combining PaddleOCR, EasyOCR, and Tesseract
- **Vector Database Integration**: Implements semantic search with Qdrant
- **Two-Stage RAG Architecture**: Optimizes query formulation and response generation
- **Conversation Memory**: Maintains context across user interactions
- **Streaming Responses**: Delivers answers progressively with optional reasoning
- **Configurable Components**: Easily customize models, database settings, and more

## System Architecture

### Document Processing Module

- **OCR Engine**: Extracts text from quote images using multiple OCR systems and LLM correction
- **PDF Processor**: Parses books and long-form content with intelligent chunking
- **Text Handler**: Manages direct transcripts and teachings

### Vector Database

- Custom Qdrant implementation for efficient semantic search
- Configurable similarity thresholds and document retrieval
- Docker integration for easy deployment

### Chatbot System

- Two-prompt approach for natural and relevant responses
- Spiritual voice alignment with Amma's teachings
- Conversation history management
- Streaming output for better user experience

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (for Qdrant vector database)
- Ollama (for local language models)

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/enlightener.git
   cd enlightener
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Start Qdrant with Docker
   ```
   docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

4. Create a vector database collection
   ```
   python Vector\ Database/create_collection.py --name vector-db-1
   ```

### Usage

1. Process documents and images to populate the vector database

2. Start the chatbot:
   ```
   python LLM\ Chatbot/chatbot.py
   ```

3. [Coming Soon] Launch the Streamlit UI:
   ```
   streamlit run app.py
   ```

## Project Structure

```
Enlightener/
├── Document Processor/
│   ├── document_processor.py  # PDF and text processing
│   ├── ocr_engine.py          # Image-to-text conversion
│   └── prompt_template.txt    # OCR processing prompt
├── Vector Database/
│   ├── create_collection.py     # Database initialization
│   ├── database_operations.py   # Vector DB interactions
│   └── delete_collection.py     # Collection management
├── LLM Chatbot/
│   ├── chatbot.py             # Main chatbot implementation
│   ├── query_vector_prompt.txt # Query reformulation prompt
│   ├── streamer.py            # Token streaming handler
│   └── system_prompt.txt      # Response generation prompt
├── User chats/                # Conversation storage
│   └── [username]/
│       └── [chat_name].json
├── config.json                # Configuration settings (Coming soon)
└── app.py                     # Streamlit UI (Coming Soon)
```

## Configuration

Edit `config.json` to customize:
- Language models used for embeddings and generation
- Vector database connection details
- Document processing parameters
- UI preferences
