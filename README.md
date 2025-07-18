# Plumbing Analysis System

A sophisticated AI-powered construction document analysis system that uses advanced multi-agent workflows to extract plumbing materials, quantities, and infrastructure details from construction plans.

## Features

### 🤖 Multi-Stage AI Analysis Pipeline
- **Preprocessor Agent**: Classifies construction documents and determines optimal analysis approach
- **Context Extraction Agent**: Identifies legends, symbols, drawing types, and document organization
- **Plumbing Analysis Agent**: Performs comprehensive material extraction and quantity analysis
- **Material Intelligence**: Advanced deduplication, validation, and confidence scoring

### 📊 Advanced Material Analysis
- **Smart Material Recognition**: Identifies plumbing components, pipes, fittings, and infrastructure
- **Quantity Estimation**: Analyzes drawings to estimate material quantities and measurements
- **Confidence Scoring**: AI provides confidence levels for each extracted material
- **Material Deduplication**: Intelligent matching and consolidation of similar materials across sheets
- **Size & Specification Extraction**: Automatically extracts dimensions and technical specifications

### 💾 Export & Data Management
- **Excel/CSV Export**: Export consolidated material lists with modern save dialog support
- **Sortable Material Tables**: Interactive tables with sorting by quantity, confidence, sheet reference
- **Real-time Progress Tracking**: Live progress updates during analysis with Server-Sent Events
- **Sheet-by-Sheet Results**: Detailed breakdown of analysis results per construction sheet

### 🎨 Modern User Interface
- **Dark Mode Design**: Professional dark theme optimized for technical drawings
- **Responsive Layout**: Split-screen view with PDF viewer and analysis results
- **Collapsible Sections**: Organized display of context, materials, and analysis stages
- **Progress Visualization**: Visual progress bars with stage-specific status icons

### 🔧 Developer & Debug Tools
- **Comprehensive Error Tracking**: Structured error logging with agent-specific debugging
- **Intermediate Result Caching**: Access to preprocessor, context, and analysis outputs
- **Material Validation Insights**: Debug endpoints for confidence distribution and filtering
- **Text Extraction Analysis**: Raw text and legend extraction debugging

## Tech Stack

### Frontend
- **Next.js 15** with App Router and Turbopack
- **React 19** with TypeScript
- **Tailwind CSS 4** for modern styling
- **XLSX.js** for advanced Excel export functionality
- **Server-Sent Events** for real-time progress updates

### Backend  
- **FastAPI** with async support and comprehensive error handling
- **Python 3.13+** with type hints throughout
- **Azure OpenAI + AutoGen** for multi-agent AI workflows
- **PyMuPDF** for advanced PDF processing and text extraction
- **Pydantic** for structured data validation and modeling
- **Pillow** for image processing and optimization

### AI & Analysis
- **AutoGen AgentChat** for sophisticated multi-agent coordination
- **Azure OpenAI GPT-4** for intelligent document analysis
- **Custom Material Intelligence** algorithms for deduplication and validation
- **Structured Data Models** for consistent agent communication

## Setup

### Prerequisites
- Node.js 18+
- Python 3.13+
- Azure OpenAI API access with GPT-4 deployment

### Environment Variables

Create a `.env` file in the project root with:

```env
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_gpt4_deployment_name

# Application URLs
FRONTEND_URL=http://localhost:3000
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
BACKEND_URL=http://localhost:8000
```

### Installation

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd plumbing-analysis
```

2. **Install frontend dependencies:**
```bash
npm install
```

3. **Setup backend environment:**
```bash
cd backend
pip install -r requirements.txt
```

### Running the Application

1. **Start the AI analysis backend:**
```bash
cd backend
python main.py
```

2. **Start the frontend interface:**
```bash
npm run dev
```

**Access Points:**
- Frontend Application: http://localhost:3000
- Backend API Documentation: http://localhost:8000/docs
- Real-time Progress Streaming: http://localhost:8000/api/progress/{task_id}/stream

## API Reference

### Core Analysis Endpoints
- `POST /api/upload` - Upload PDF for analysis
- `POST /api/analyze` - Trigger multi-stage AI analysis
- `GET /api/result/{task_id}` - Retrieve complete analysis results
- `GET /api/progress/{task_id}/stream` - Stream real-time progress updates

### Debug & Development Endpoints
- `GET /api/debug/intermediate/{task_id}` - Access raw agent outputs
- `GET /api/debug/material-validation/{task_id}` - Material validation insights
- `GET /api/debug/text-extraction/{task_id}` - Text extraction debugging
- `DELETE /api/cache/clear` - Clear analysis cache

## Analysis Workflow

### Stage 1: Document Preprocessing
- Document classification and complexity assessment
- Optimal analysis strategy determination
- Sheet type identification (plumbing, civil, mixed, etc.)

### Stage 2: Context Extraction  
- Legend and symbol identification
- Drawing type classification
- Technical standards recognition
- Document organization analysis

### Stage 3: Material Analysis
- Comprehensive visual analysis of construction drawings
- Material identification with quantity estimation
- Confidence scoring and validation
- Cross-sheet material consolidation

### Stage 4: Intelligence Processing
- Material deduplication using fuzzy matching algorithms
- Quantity validation and outlier detection
- Confidence-based filtering and quality assurance
- Export-ready data preparation

## Project Structure

```
plumbing-analysis/
├── src/                          # Frontend React application
│   ├── app/                     # Next.js app router pages
│   │   ├── page.tsx            # Main application interface
│   │   └── layout.tsx          # App layout and global styles
│   └── components/              # React components
│       ├── PlumbingAnalysis.tsx # Advanced analysis results display
│       ├── PdfUploader.tsx     # File upload with progress tracking
│       └── PdfViewer.tsx       # PDF document viewer
├── backend/                     # Python FastAPI backend
│   ├── main.py                 # FastAPI application and API endpoints
│   ├── models.py               # Pydantic data models for structured communication
│   ├── pdf_converter.py        # Advanced PDF processing and text extraction
│   └── config/                 # Agent configuration and system prompts
│       └── agent_config.py     # AI agent system messages and settings
├── public/                     # Static frontend assets
└── package.json               # Frontend dependencies and scripts
```

## Material Intelligence Features

### Smart Deduplication
- Fuzzy string matching for material names
- Size and specification comparison
- Cross-sheet material consolidation
- Confidence-weighted material combination

### Validation & Quality Control
- Quantity reasonableness validation
- Confidence threshold filtering (default: 60%)
- Precision flags for suspiciously exact measurements
- Large quantity flagging for manual review

### Export Capabilities
- **Excel Export**: Full material details with formatting
- **CSV Export**: Lightweight data export for analysis
- **Modern Save Dialogs**: Browser-native file save experience
- **Sortable Data**: Interactive sorting by any column

## Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Implement changes with proper type hints and error handling
4. Test with various construction document types
5. Update documentation if needed
6. Submit a Pull Request

### AI Agent Development
- Modify system prompts in `backend/config/agent_config.py`
- Test agent interactions with `backend/test_improvements.py`
- Use debug endpoints to validate agent outputs
- Ensure proper Pydantic model validation

## Security & Performance

- **Environment-based configuration** for sensitive API keys
- **CORS protection** with configurable origins
- **Input sanitization** for uploaded files
- **Async processing** for responsive user experience
- **Progress streaming** to prevent timeout issues
- **Structured error handling** with comprehensive logging

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This application requires Azure OpenAI access for AI-powered analysis. Ensure your Azure OpenAI deployment supports GPT-4 for optimal results.
