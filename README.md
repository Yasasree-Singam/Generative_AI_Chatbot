# Generative_AI_Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that answers questions about uploaded PDF documents using a local LLM (Ollama/gemma3:4b), HuggingFace embeddings, and FAISS for vector search.

## Features
- Upload a PDF and ask questions about its content
- Uses local LLM (Ollama) for privacy and speed
- Embeddings via HuggingFace sentence-transformers
- In-memory FAISS vector store (no persistence)
- All logic in a single Python file (`chatbot.py`)

## Quickstart
1. Clone this repo
2. Set up Python 3.11+ and create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Start Ollama with the gemma3:4b model
5. Run: `streamlit run chatbot.py`

## Project Structure
- `chatbot.py` - Main Streamlit app
- `requirements.txt` - Python dependencies
- `Notes.ipynb` - LLM/AI learning notes

## License
MIT License
