# 🤖 Agentic RAG 2.0 - Intelligent Web-Powered Q&A System

An advanced Retrieval-Augmented Generation (RAG) system powered by LangChain agents that can dynamically search the web, fetch webpage content, and query local knowledge bases to answer questions with real-time information.

## 🌟 Features

- **🌐 Dynamic Web Search**: Real-time information from DuckDuckGo (no API key needed)
- **📄 Webpage Fetching**: Extract and analyze content from specific URLs
- **💾 Local Knowledge Base**: Optional pgvector-powered semantic search on scraped content
- **🧠 Intelligent Agent**: Multi-step reasoning with tool selection
- **🔗 Source Citations**: Automatic source attribution with URLs
- **⚡ FastAPI Backend**: Modern async web framework with streaming responses
- **🎨 Clean UI**: Simple, responsive web interface

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

## 🏗️ Architecture

### Agent Workflow

```
User Query → Agent Planning → Multi-Tool Execution → Response Generation
                    ↓
      [Web Search | Fetch Webpage | Local Knowledge Base]
```

The agent intelligently selects and combines tools based on the query:
- **Current events/facts**: Uses web search
- **Specific URLs**: Fetches and analyzes webpage content  
- **Domain-specific queries**: Searches local vector database (if available)

### Technology Stack

- **Agent Framework**: LangChain 1.0+ with `langchain-classic` agents
- **LLM**: OpenAI GPT OSS 20B via OpenRouter (free tier)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (384 dimensions)
- **Vector DB**: PostgreSQL 14+ with pgvector extension
- **Web Framework**: FastAPI with async streaming
- **Frontend**: Vanilla HTML/CSS/JavaScript

## 📦 Prerequisites

### System Requirements

- Python 3.12+
- PostgreSQL 14+ with pgvector extension (optional, for local knowledge base)
- 4GB+ RAM
- Internet connection (for web search and LLM API)

### Required Accounts

1. **OpenRouter Account** (Free tier available)
   - Sign up at https://openrouter.ai/
   - Get your API key from the dashboard
   - Free tier includes: OpenAI GPT OSS 20B, DeepSeek models, etc.

2. **PostgreSQL with pgvector** (Optional)
   - Only needed if you want local knowledge base functionality
   - See [PostgreSQL Setup](#postgresql-setup-optional) below

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nyang64/agentic-rag20
cd  agentic-rag20
```

### 2. Create Virtual Environment

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. PostgreSQL Setup (Optional)

If you want to use the local knowledge base feature:

```bash
# Install PostgreSQL with pgvector
# On macOS:
brew install postgresql@18
brew install pgvector

# Start PostgreSQL
brew services start postgresql@18

# Create database and enable extension
createdb myprojdb
psql myprojdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql myprojdb -c "CREATE SCHEMA IF NOT EXISTS scraper;"
psql myprojdb -c "CREATE USER myuser WITH PASSWORD 'mypassword';"
psql myprojdb -c "GRANT ALL PRIVILEGES ON DATABASE myprojdb TO myuser;"
psql myprojdb -c "GRANT ALL ON SCHEMA scraper TO myuser;"
```

## ⚙️ Configuration

### 1. Environment Variables

Create or edit the `.env` file in the project root:

```bash
# Required - OpenRouter API Configuration
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Required - Model Selection (OpenAI Free Model via OpenRouter)
OPENAI_FREE_MODEL=openai/gpt-oss-20b:free

# Optional - PostgreSQL for Local Knowledge Base
PGVECTOR_DB_URL=postgresql://myuser:mypassword@localhost:5432/myprojdb

# Optional - LangSmith Tracing (for debugging)
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key

# System Configuration
TOKENIZERS_PARALLELISM=false
SCRAPY_SETTINGS_MODULE=scraper.settings
```

### 2. Key Configuration Notes

#### Why OpenAI Free Model?

The agent uses `openai/gpt-oss-20b:free` because:
- ✅ **Function calling support**: Required for agent tool selection
- ✅ **Reliable via OpenRouter**: Stable routing and availability
- ✅ **Free tier**: No cost for testing and development
- ❌ **DeepSeek models**: Don't support function calling via OpenRouter

#### OpenRouter Free Models

Available free models on OpenRouter:
- `openai/gpt-oss-20b:free` - Best for agents (function calling support) ⭐
- `x-ai/grok-4.1-fast:free` - Fast responses
- `tngtech/deepseek-r1t2-chimera:free` - Good for general queries
- `deepseek/deepseek-chat-v3.1:free` - Advanced reasoning

**Important**: Only `openai/gpt-oss-20b:free` currently supports the function calling needed for agentic workflows.

## 🏃 Running the Application

### 1. Quick Start

```bash
# Activate virtual environment
source env/bin/activate  # On Windows: env\Scripts\activate

# Run the web application
python web_app.py
```

The application will start on `http://localhost:8000`

### 2. Alternative: Run with Uvicorn

```bash
# With auto-reload for development
uvicorn web_app:app --reload --port 8000

# Production mode
uvicorn web_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Access the Application

Open your browser and navigate to:
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (FastAPI auto-generated)

## 🧪 Testing

### 1. Run Complete Test Suite

```bash
python test_agent.py
```

This diagnostic script checks:
1. ✓ Environment variables
2. ✓ Package imports
3. ✓ LLM connection (OpenRouter)
4. ✓ Web search functionality (DuckDuckGo)
5. ✓ Vector database connection (optional)
6. ✓ Agent import and initialization
7. ✓ Agent execution with sample query

### 2. Quick Agent Test

```bash
# Test agent directly from command line
python -c "
from scraper.agent_classic import create_agentic_rag
agent = create_agentic_rag()
result = agent.invoke({'input': 'What is the weather in New York today?'})
print(result['output'])
"
```

### 3. Test Individual Components

```bash
# Test LLM connection
python -c "
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(
    model=os.getenv('OPENAI_FREE_MODEL'),
    openai_api_key=os.getenv('OPENROUTER_API_KEY'),
    openai_api_base='https://openrouter.ai/api/v1'
)
print(llm.invoke('Say hello!').content)
"

# Test web search
python -c "
from scraper.agent_classic import web_search
result = web_search.func('Python programming')
print(result[:500])
"

# Test vector database (if configured)
python -c "
from scraper.raq_query import retrieve_top3
docs = retrieve_top3('test query')
print(f'Retrieved {len(docs)} documents')
"
```

## 💡 Usage Examples

### Example Queries

Try these queries in the web interface:

#### Current Events & News
```
What are the latest developments in AI?
What happened in the news today?
Who won the recent elections?
```
Below is a sample run screen shot, asking "what is the weather like in New York City today?"

![Rag Web UI](static/img/Agentic-RAG-localRun.png)

#### Weather & Real-time Data
```
What is the weather like in San Francisco today?
What is the current stock price of Apple?
What is the exchange rate USD to EUR?
```

#### Research & Information Gathering
```
Compare the top 3 fastest trains in the world
What are the benefits of electric vehicles?
Explain quantum computing in simple terms
```

#### Multi-step Reasoning
```
Find information about renewable energy adoption in Europe and compare it to Asia
What are the main causes of climate change and what solutions are being proposed?
```

### Using the API Directly

#### Basic Query

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=What is the capital of France?"
```

#### Verbose Mode (See Agent Reasoning)

```bash
curl -X POST "http://localhost:8000/ask_verbose" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=What is the weather in Tokyo?"
```

Response format:
```json
{
  "answer": "The current weather in Tokyo...",
  "steps": [
    {
      "tool": "web_search",
      "input": "Tokyo weather today",
      "output": "Current weather data..."
    }
  ]
}
```

#### Streaming Response (Watch Agent Think)

```bash
curl -X POST "http://localhost:8000/ask_streaming" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=Tell me about the Mars rover"
```

## 📁 Project Structure

```
agentic-rag20/
├── README.md                          # This file
├── .env                               # Environment variables (create this)
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata
├── web_app.py                         # FastAPI web application
├── test_agent.py                      # Diagnostic test suite
├── test_imports.py                    # Import verification
├── adhoc_query.py                     # Ad-hoc vector DB queries
│
├── scraper/                           # Main package directory
│   ├── __init__py                     # Package initialization
│   ├── agent_classic.py               # ⭐ Main agent (WORKING)
│   ├── agent.py                       # ❌ Has library issues
│   ├── agent_simple.py                # ❌ Has library issues  
│   ├── langgraph_agent.py             # ❌ Has library issues
│   ├── raq_query.py                   # Vector DB RAG queries
│   ├── embed_and_store.py             # Embedding & storage utilities
│   ├── cache.py                       # Caching utilities
│   ├── settings.py                    # Scrapy settings
│   ├── pipelines.py                   # Data processing pipelines
│   └── spiders/                       # Scrapy spiders (for data collection)
│
├── static/                            # Frontend assets
│   ├── index.html                     # Web UI
│   └── img/                           # Images
│
├── env/                               # Virtual environment (git-ignored)
├── __pycache__/                       # Python cache (git-ignored)
└── .git/                              # Git repository
```

### Key Files

- **`scraper/agent_classic.py`**: ⭐ The working agent implementation using `langchain-classic`
- **`web_app.py`**: FastAPI application with streaming endpoints
- **`test_agent.py`**: Comprehensive diagnostic and testing script
- **`scraper/raq_query.py`**: Vector database retrieval (optional feature)

## 🌐 API Endpoints

### `GET /`
Returns the main web interface (HTML)

### `POST /ask`
Standard query endpoint with streaming response

**Request:**
```
Content-Type: application/x-www-form-urlencoded
query=Your question here
```

**Response:**
```
Streaming text/plain response with answer and sources
```

### `POST /ask_verbose`
Returns detailed agent reasoning trace

**Request:**
```
Content-Type: application/x-www-form-urlencoded
query=Your question here
```

**Response:**
```json
{
  "answer": "The answer to your question...",
  "steps": [
    {
      "tool": "web_search",
      "input": "search query",
      "output": "search results..."
    }
  ]
}
```

### `POST /ask_streaming`
Streams agent's thought process in real-time

**Request:**
```
Content-Type: application/x-www-form-urlencoded
query=Your question here
```

**Response:**
```
Streaming text/plain with agent thoughts:
🤔 Thinking about your question...
🔧 Using tool: web_search
💡 Found: [results]
✅ Answer: [final answer]
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'langchain_classic'"

**Solution:**
```bash
pip install langchain-classic==1.0.0
```

#### 2. "OpenRouter API Key Error"

**Symptoms:** `openai.AuthenticationError: Error code: 401`

**Solution:**
- Verify your API key is correct in `.env`
- Check that `.env` file exists in project root
- Ensure `OPENROUTER_API_KEY=sk-or-v1-...` is set correctly
- Get a new key from https://openrouter.ai/ if needed

#### 3. "Rate Limit Exceeded"

**Symptoms:** `RateLimitError: 429`

**Solution:**
- Free tier has rate limits (check OpenRouter dashboard)
- Wait a few minutes between requests
- Consider upgrading to paid tier for higher limits

#### 4. "Cannot connect to PostgreSQL"

**Symptoms:** `psycopg2.OperationalError`

**Solution:**
- Check PostgreSQL is running: `brew services list` (macOS)
- Verify database exists: `psql -l | grep myprojdb`
- Check credentials in `.env` match your PostgreSQL setup
- Note: PostgreSQL is optional - agent works without it for web search

#### 5. "Web search returns no results"

**Symptoms:** `No search results found`

**Solution:**
- Check internet connection
- DuckDuckGo might be rate-limiting (wait a minute)
- Try a different query
- Check if you're behind a firewall/proxy

#### 6. "Agent only uses one tool"

**Symptoms:** Agent doesn't combine multiple tools

**Solution:**
- This is normal behavior - agent picks the most appropriate tool
- Try queries that clearly need multiple sources
- Check agent reasoning in verbose mode
- Adjust system prompt in `agent_classic.py` if needed

### Getting Help

If you encounter issues:

1. **Run diagnostics**: `python test_agent.py`
2. **Check logs**: Look for error messages in terminal
3. **Verify environment**: Ensure all `.env` variables are set
4. **Test components**: Use individual component tests above
5. **Check documentation**: Review this README and `AGENTIC_RAG_UPGRADE_GUIDE.md`

## 💻 Development Notes

### Why agent_classic.py?

The project has multiple agent files, but **only `agent_classic.py` works** because:

- ✅ `agent_classic.py`: Uses `langchain-classic` 1.0.0 (stable, working)
- ❌ `agent.py`: Uses newer LangChain APIs with dependency conflicts
- ❌ `agent_simple.py`: Simplified version with library compatibility issues
- ❌ `langgraph_agent.py`: Advanced LangGraph implementation with dependency issues

**Recommendation**: Use `agent_classic.py` for production. Other files are kept for reference.

### Agent Tools

The agent has access to three tools:

1. **`web_search`**: DuckDuckGo search (no API key needed)
   - Used for: current events, facts, recent information
   - Returns: top 5 search results with titles, snippets, URLs

2. **`fetch_webpage`**: Webpage content extraction
   - Used for: getting detailed content from specific URLs
   - Returns: cleaned text content (up to 5000 chars)

3. **`search_local_knowledge`**: Vector database search (optional)
   - Used for: domain-specific previously scraped content
   - Returns: top 3 most relevant documents from local DB

### Customizing the Agent

#### Change System Prompt

Edit `scraper/agent_classic.py`:

```python
system_message = """You are an intelligent research assistant...
[Customize instructions here]
"""
```

#### Add New Tools

```python
@tool
def your_custom_tool(input: str) -> str:
    """Description of what this tool does"""
    # Your implementation
    return result

# Add to tools list
tools = [
    web_search,
    fetch_webpage,
    search_local_knowledge,
    your_custom_tool,  # Add here
]
```

#### Change LLM Model

Edit `.env`:

```bash
# Try different free models
OPENAI_FREE_MODEL=openai/gpt-oss-20b:free
# or
OPENAI_FREE_MODEL=x-ai/grok-4.1-fast:free
```

Note: Function calling support required for agents!

### Performance Optimization

#### Enable Caching (TODO)

```python
# scraper/cache.py provides a caching layer
from scraper.cache import SearchCache

cache = SearchCache(ttl_hours=24)
# Implement in agent_classic.py
```

#### Parallel Tool Execution (TODO)

```python
# Execute multiple tools simultaneously
from langchain_core.runnables import RunnableParallel
# Implementation needed
```

### Contributing

1. Create a new branch
2. Make your changes
3. Test with `python test_agent.py`
4. Submit a pull request

### Future Enhancements

- [ ] Implement search result caching
- [ ] Add Brave Search API support (better results)
- [ ] Parallel tool execution
- [ ] Better source validation and credibility scoring
- [ ] Enhanced UI with agent visualization
- [ ] User authentication and query history
- [ ] API rate limiting
- [ ] Docker containerization

## 📚 Additional Documentation

- **`AGENTIC_RAG_UPGRADE_GUIDE.md`**: Detailed guide on upgrading from RAG 1.0 to 2.0
- **FastAPI Docs**: Auto-generated at http://localhost:8000/docs
- **LangChain Docs**: https://python.langchain.com/docs/
- **OpenRouter Docs**: https://openrouter.ai/docs

## 📝 License

This project is for educational and development purposes.

## 👤 Author

Nanchang Yang  
Email: nyang63@gmail.com

## 🙏 Acknowledgments

- LangChain for the agent framework
- OpenRouter for free LLM access
- PostgreSQL and pgvector for vector storage
- FastAPI for the web framework

---

**Last Updated**: December 2024  
**Status**: ✅ Ready / Working solution  
**Version**: 2.0.0
