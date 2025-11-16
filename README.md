# Healthcare Chat Assistant

![Healthcare AI](https://img.shields.io/badge/Healthcare-AI%20Assistant-4caf50)
![Agentic AI](https://img.shields.io/badge/Agentic-AI-FF5722)
![LLM Powered](https://img.shields.io/badge/Powered%20by-Gemini-9C27B0)
![Snowflake](https://img.shields.io/badge/Database-Snowflake-29B5E8)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

A production-ready LLM-powered chatbot that transforms complex healthcare data into conversational insights through natural language queries. Built with RAG architecture, Google Gemini, and Pinecone vector search.

---

## 🎯 Overview

**Problem:**  
Healthcare professionals need insights from hospital discharge data but lack SQL expertise and face complex schemas with 200+ columns.

**Solution:**  
A Text-to-SQL chatbot enabling natural language queries like:
- *"What are the top 5 costliest diseases in New York?"*
- *"Show readmission trends for diabetic patients"*

The system generates SQL, executes against 2M+ rows of NY SPARCS discharge data, and returns conversational answers with visualizations.

---

## ✨ Key Features

- **Natural Language to SQL**: Query healthcare data without writing SQL
- **RAG-Powered Retrieval**: Semantic search via Pinecone vector database
- **Multi-Turn Conversations**: Context-aware follow-up questions
- **Dynamic Visualizations**: Auto-generated charts (bar, pie, line)
- **Web Search Fallback**: Handles out-of-scope medical questions
- **Production Logging**: Structured logs for debugging and monitoring
- **Rate Limiting**: Prevents API quota exhaustion

---

## 🏗️ Architecture

### Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 |
| **Vector DB** | Pinecone |
| **Database** | Snowflake |
| **Framework** | LangChain (ReAct agent) |
| **Backend** | FastAPI |
| **Frontend** | React |

### System Flow
```
User Query → Embedding → Vector Search → SQL Generation → Execution → Response
```

**Single-agent architecture** with 6 specialized tools:
1. Schema retrieval
2. Metadata search (Pinecone)
3. SQL generation & execution
4. Response formatting
5. Visualization generation
6. Web search fallback

---

## 📂 Project Structure
```
backend/
├── core/              # Main agent orchestrator
├── tools/             # 6 agent tools
├── utils/             # Rate limiter, logging, helpers
├── connectors/        # Database connections
├── evaluation/        # Test framework (50 queries)
├── gen_color_plot.py  # Visualization service
├── web_search_handler.py  # Search service
└── app.py             # FastAPI server

frontend/
└── src/               # React UI components
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Snowflake account
- Pinecone API key
- Google Gemini API key

### Setup
```bash
# Clone repo
git clone https://github.com/Sankalp20487/Agentic_Healthcare_Chatbot.git
cd Agentic_Healthcare_Chatbot

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run
python app.py
```

Backend runs at `http://localhost:8000`

### Frontend
```bash
cd frontend
npm install
npm start
```

Frontend runs at `http://localhost:3000`

---

## 🧠 How It Works

### RAG Architecture

The system uses Retrieval-Augmented Generation for accurate SQL:

1. User query is embedded (384-dim vector)
2. Pinecone retrieves top-5 relevant columns (semantic search)
3. Gemini generates SQL using retrieved context
4. SQL executes on Snowflake (2M+ row dataset)
5. Results formatted into natural language

**Why RAG?**
- Handles schema ambiguity (e.g., "cost" → correct column)
- Adapts to schema changes without retraining
- Better than fine-tuning for evolving databases

---

## 📊 Performance

### Baseline Metrics

Measured during development with active connections:

| Metric | Value |
|--------|-------|
| SQL Accuracy | ~85-90% |
| Avg Latency | 3.2s |
| Success Rate | 92% |
| Cost/Query | ~$0.04 |

### Evaluation Framework

Includes 50 test queries (`evaluation/test_queries.py`) covering:
- Simple aggregations (AVG, SUM, COUNT)
- Top-N ranking queries
- Multi-dimensional comparisons
- Complex filters

Pattern-based validation checks for expected SQL components (ORDER BY, GROUP BY, etc.)

---

## 🔧 Monitoring & Logging

### Production Logging

Structured logs to console (INFO) and daily rotating files (DEBUG):
```
14:23:45 - INFO - Processing SQL query: top 5 diseases...
14:23:47 - INFO - Query executed in 2.3s - 5 rows returned
```

Logs capture: SQL queries, execution times, errors, vector search results

### LangSmith Integration

Enable via environment variables:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
```

Provides: Prompt tracing, token tracking, latency metrics, cost analysis

---

## 🚢 Deployment

### Docker
```bash
docker-compose up
```

Runs both backend and frontend in containers.

### Production Considerations

For enterprise deployment:
- Migrate to Databricks for unified data + AI platform
- Implement Unity Catalog for data governance
- Add MLflow for model/prompt versioning
- Set up CI/CD pipeline for automated testing
- Configure monitoring dashboards

---

## 🎯 Example Queries
```
"What is the average cost of knee replacement?"
"Show me the top 10 most common diagnoses"  
"Compare hospital costs by age group"
"Create a bar chart of the top 5 procedures by volume"
"What are readmission rates for diabetic patients?"
```

---

## 🔮 Future Work

- Multi-database federation (combine multiple data sources)
- Predictive analytics integration
- Advanced caching for frequent queries
- Enhanced visualization types
- User authentication and RBAC

---

## 📝 License

MIT License - see LICENSE file for details.

---

## 👤 Author

**Sankalp Biswal**  
Data Scientist | AI/ML Engineer

---

## 🙏 Acknowledgments

Dataset: NY SPARCS Inpatient Discharge Data  
Technologies: LangChain, Google Gemini, Pinecone, Snowflake, FastAPI, React
