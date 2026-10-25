# DocuMind — Multi-Document RAG Assistant

DocuMind is an AI-powered document question-answering app built with Streamlit, LangChain, Chroma, and OpenAI.

## Features
- Upload multiple PDF and TXT documents
- Ask questions about uploaded documents
- Hybrid retrieval using BM25 keyword search and Chroma semantic search
- Answers grounded in retrieved document excerpts
- Supporting source filenames and page references

## Tech Stack
- Python
- Streamlit
- LangChain
- Chroma
- OpenAI embeddings and chat model

## Setup

1. Clone the repository and navigate to this project folder.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your API key:

   ```text
   OPENAI_API_KEY=your_api_key_here
   ```

5. Run the app:

   ```bash
   streamlit run app.py
   ```

## Security
Keep your `.env` file and API key private. Do not commit secrets to GitHub.

## Note
Use sample or authorized documents. Verify important answers against the supporting sources.
