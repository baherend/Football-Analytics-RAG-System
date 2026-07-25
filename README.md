# FIFA World Cup 2022 — RAG Analytics System

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about the FIFA World Cup 2022 using StatsBomb open data.

## Features

- **Hybrid Retrieval** — BM25 lexical search + sentence embeddings with Reciprocal Rank Fusion (RRF)
- **2,835+ documents** — Match summaries, key events, player stats, team analysis
- **64 matches** — Complete FIFA World Cup 2022 coverage
- **Groq LLM** — Fast answer generation with source citations
- **Streamlit UI** — Interactive chat interface

## Project Structure

```
├── 01_documents.py              # Load documents from StatsBomb data
├── 02_preprocessing.py          # Text cleaning & normalization
├── 03_chunking.py               # Document chunking with overlap
├── 04_vector_representation.py  # BM25 + dense embeddings + hybrid search
├── 05_create_chroma_store.py    # ChromaDB vector store creation
├── 06_retrieve_context.py       # Hybrid retrieval with RRF fusion
├── 07_prompting.py              # Prompt building + Groq LLM integration
├── streamlit_app.py             # Streamlit web interface
├── requirements.txt             # Python dependencies
└── output/                      # Pre-built pipeline artifacts
```

## Pipeline

```
Documents → Preprocessing → Chunking → Vector Representation → ChromaDB
                                                    ↓
User Query → BM25 Search + Dense Search → RRF Fusion → Context → LLM → Answer
```

## Setup

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/baherend/Football-Analytics-RAG-System.git
cd Football-Analytics-RAG-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
export GROQ_API_KEY="your-groq-api-key"

# 4. Run the Streamlit app
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. In "Secrets", add:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

## Example Queries

| Category | Example |
|----------|---------|
| Player stats | "How many goals did Messi score?" |
| Superlative | "Who scored the most goals?" |
| Comparison | "Compare Messi and Mbappé's performance" |
| Match analysis | "How did France play in the final?" |
| Team style | "What was Argentina's playing style?" |

## Data Source

Uses [StatsBomb Open Data](https://github.com/statsbomb/open-data) for the FIFA World Cup 2022 (competition_id=43, season_id=106).

## API Key Rules

- **NEVER** hardcode API keys in Python files
- **NEVER** commit `.env` files or `secrets.toml`
- Use **Streamlit secrets** for deployment
- Use **environment variables** for local development

## License

Educational project. StatsBomb data subject to their open data license.
