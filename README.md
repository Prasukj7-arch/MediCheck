# Medical Report RAG Extractor

A complete web-based application that uses Retrieval-Augmented Generation (RAG) to analyze medical reports and provide expert insights in response to user queries.

## 🏥 Features

- **PDF Processing**: Upload and extract text from medical report PDFs
- **Semantic Chunking**: Intelligently split documents into meaningful segments
- **Vector Embeddings**: Create embeddings using sentence-transformers
- **Vector Database**: Store and retrieve relevant chunks using ChromaDB
- **AI-Powered Analysis**: Generate expert insights using Mistral-7B-Instruct via OpenRouter
- **Modern Web Interface**: Beautiful, responsive frontend with drag-and-drop upload

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key (free tier available)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**:
   - Get a free API key from [OpenRouter](https://openrouter.ai/)
   - Create a `.env` file in the project root:
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and go to `http://localhost:5000`

## 📋 Usage

### 1. Upload Medical Report
- Drag and drop a PDF file or click "Choose PDF File"
- The system will automatically:
  - Extract text from the PDF
  - Split it into semantic chunks
  - Create vector embeddings
  - Store in the vector database

### 2. Ask Questions
- Type your question in the text area
- Click "Ask Question" or press Enter
- Get AI-powered insights based on the uploaded report

### Example Questions:
- "What are the main findings in this medical report?"
- "What medications were prescribed to the patient?"
- "What are the patient's vital signs and lab results?"
- "Are there any abnormal or concerning results?"
- "What is the diagnosis and recommended treatment plan?"

## 🏗️ Architecture

### RAG Pipeline Components:

1. **Data Ingestion**:
   - PDF text extraction using PyPDF2
   - Semantic chunking with overlap
   - Vector embedding creation using sentence-transformers

2. **Vector Storage**:
   - ChromaDB for efficient similarity search
   - Cosine similarity for retrieval
   - Persistent storage with metadata

3. **Retrieval**:
   - Query embedding generation
   - Top-k similarity search
   - Context chunk selection

4. **Generation**:
   - Prompt engineering for medical expertise
   - Mistral-7B-Instruct model via OpenRouter
   - Professional medical response generation

## 🔧 Technical Details

### Backend (Flask)
- **Framework**: Flask with CORS support
- **PDF Processing**: PyPDF2 for text extraction
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB with DuckDB backend
- **LLM**: Mistral-7B-Instruct via OpenRouter API

### Frontend
- **HTML5/CSS3**: Modern, responsive design
- **JavaScript**: Vanilla JS for interactivity
- **Features**: Drag-and-drop upload, real-time status, example queries

### API Endpoints
- `GET /`: Main application page
- `POST /upload`: PDF file upload and processing
- `POST /query`: Query processing and response generation
- `GET /status`: System health check

## 📁 Project Structure

```
medical-report-rag/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── env_example.txt       # Environment variables template
├── templates/
│   └── index.html        # Frontend interface
├── uploads/              # Upload directory (auto-created)
└── chroma_db/           # Vector database storage (auto-created)
```

## 🔒 Security & Privacy

- **Local Processing**: All PDF processing happens locally
- **No Data Persistence**: Uploaded files are not permanently stored
- **API Key Security**: Use environment variables for API keys
- **Input Validation**: File type and size restrictions

## 🛠️ Configuration

### Environment Variables
- `OPENROUTER_API_KEY`: Your OpenRouter API key

### Customization Options
- **Chunk Size**: Modify `chunk_size` parameter in `chunk_text()` method
- **Overlap**: Adjust `overlap` parameter for chunk overlap
- **Top-k Retrieval**: Change `top_k` parameter in search functions
- **Model**: Switch to different models via OpenRouter

## 🚨 Troubleshooting

### Common Issues:

1. **"No module named 'sentence_transformers'"**
   - Run: `pip install -r requirements.txt`

2. **"OpenRouter API key not found"**
   - Create `.env` file with your API key
   - Get free API key from https://openrouter.ai/

3. **"PDF text extraction failed"**
   - Ensure PDF contains extractable text (not just images)
   - Try a different PDF file

4. **"No relevant information found"**
   - Upload a medical report first
   - Try different question phrasing

### Performance Tips:
- Use smaller chunk sizes for detailed analysis
- Increase overlap for better context preservation
- Adjust top-k retrieval based on document size

## 📈 Future Enhancements

- [ ] Support for multiple file formats (DOCX, TXT)
- [ ] Batch processing for multiple reports
- [ ] Export functionality for analysis results
- [ ] User authentication and session management
- [ ] Advanced medical terminology recognition
- [ ] Integration with medical databases
- [ ] Real-time collaboration features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This application is for educational and research purposes. It should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.

---

**Built with ❤️ for the medical AI community**
