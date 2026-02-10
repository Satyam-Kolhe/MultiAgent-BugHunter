# Infineon Bug Hunter - Multi-Agent System

A comprehensive multi-agent system for detecting and fixing bugs in embedded code, specifically designed for Infineon AURIX microcontrollers.

## 🏗️ Architecture

The system consists of 4 specialized agents working together:

1. **📚 Librarian Agent**: Context retrieval and documentation search using RAG
2. **🔍 Inspector Agent**: Bug detection using pattern matching and LLM analysis
3. **🔬 Diagnostician Agent**: Root cause analysis and impact assessment
4. **🔧 Fixer Agent**: Automated code fix generation

## 📁 Project Structure

```
Infineon/
├── backend/
│   ├── app.py                 # FastAPI backend server
│   ├── agents/
│   │   ├── librarian.py       # Librarian agent
│   │   ├── inspector.py       # Inspector agent
│   │   ├── diagnostician.py   # Diagnostician agent
│   │   └── fixer.py           # Fixer agent
│   ├── utils/
│   │   ├── llm_client.py      # Unified LLM client (Gemini/Groq/HuggingFace)
│   │   └── document_processor.py  # Document parsing and code extraction
│   ├── data/
│   │   ├── input_format_guide.md
│   │   ├── code_extraction_patterns.md
│   │   └── document_processing_instructions.md
│   ├── tests/
│   │   └── test_full_pipeline.py
│   └── requirements.txt
├── frontend/
│   ├── app.py                 # Streamlit UI
│   └── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create/update `backend/.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
GROQ_API_KEY=your_groq_api_key
```

### 3. Start Backend Server

```bash
cd backend
python app.py
# Or: uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 4. Start Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

The UI will be available at `http://localhost:8501`

## 📡 API Endpoints

### Main Endpoints

- `POST /api/analyze` - Analyze code (JSON body with code string)
- `POST /api/analyze/file` - Analyze code from uploaded file
- `GET /health` - Health check
- `GET /api/stats` - System statistics

### Individual Agent Endpoints

- `POST /api/librarian/analyze` - Run only Librarian agent
- `POST /api/inspector/analyze` - Run only Inspector agent

## 💻 Usage Examples

### Using the API Directly

```python
import requests

# Analyze code
response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "code": """
        void read_sensor() {
            int buffer[10];
            for(int i=0; i<=10; i++) {
                buffer[i] = *SENSOR;
            }
        }
        """,
        "context_type": "embedded",
        "include_fixes": True
    }
)

result = response.json()
print(result)
```


## 🔧 Supported File Formats

- **Source Code**: `.c`, `.cpp`, `.h`, `.hpp`, `.py`
- **Text Files**: `.txt`, `.md` (with code blocks)
- **Documents**: `.pdf`, `.docx` (with embedded code)

The system automatically extracts code from documents using pattern matching and RAG-based extraction.

## 🎯 Features

### Code Analysis
- ✅ Buffer overflow detection
- ✅ Memory leak detection
- ✅ Missing volatile qualifiers
- ✅ MISRA-C rule violations
- ✅ Embedded system specific issues
- ✅ Safety-critical code analysis

### Automated Fixes
- ✅ Code fix generation
- ✅ Safety notes for Infineon hardware
- ✅ Test case generation
- ✅ Before/after code comparison

### Document Processing
- ✅ Automatic code extraction from documents
- ✅ Multiple format support
- ✅ Code block detection
- ✅ Language detection

## 🧪 Testing

Run the full pipeline test:

```bash
cd backend
python -m tests.test_full_pipeline --demo
```

Run individual agent tests:

```bash
python -m tests.test_api_connections
```

## 📊 RAG Data Files

The system uses RAG (Retrieval-Augmented Generation) for intelligent document processing. RAG data files are stored in `backend/data/`:

- `input_format_guide.md` - Instructions for handling different input formats
- `code_extraction_patterns.md` - Patterns for extracting code from documents
- `document_processing_instructions.md` - Processing workflow and best practices

These files are automatically loaded into the ChromaDB vector database for semantic search.

## 🔑 API Keys

The system uses multiple LLM providers:

- **Gemini (Google)**: Used by Librarian agent
- **Groq**: Used by Diagnostician and Fixer agents
- **HuggingFace**: Used by Inspector agent (optional fallback)

All keys are configured via environment variables in `backend/.env`.

## 🐛 Troubleshooting

### API Not Connecting
- Ensure backend is running on port 8000
- Check `API_BASE_URL` in frontend (default: `http://localhost:8000`)
- Verify CORS settings in `backend/app.py`

### Rate Limiting Errors
- The system includes automatic retry logic for rate limits
- Wait times are automatically calculated from error messages
- Consider using different API keys for different agents

### File Upload Issues
- Ensure file extensions are supported
- Check file size limits
- Verify file encoding (UTF-8 recommended)

## 📝 License

This project is part of the Infineon Hackathon demo.

## 🤝 Contributing

This is a hackathon project. For improvements, please create issues or pull requests.

---

**Built with**: FastAPI, Streamlit, ChromaDB, LangChain, and multiple LLM providers (Gemini, Groq, HuggingFace)
