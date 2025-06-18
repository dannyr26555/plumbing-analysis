# Plumbing Analysis System

A modern web application for analyzing plumbing systems in construction plans using Azure OpenAI and AutoGen.

## Features

- PDF upload and conversion to images
- AI-powered analysis of plumbing systems
- Modern dark mode UI
- Real-time analysis results
- Secure API endpoints

## Tech Stack

### Frontend
- Next.js 14
- TypeScript
- Tailwind CSS
- React Components

### Backend
- FastAPI
- Python 3.13
- Azure OpenAI
- AutoGen
- PyMuPDF

## Setup

### Prerequisites
- Node.js 18+
- Python 3.13+
- Azure OpenAI API access

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# Frontend Configuration
FRONTEND_URL=http://localhost:3000
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd plumbing-analysis
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### Running the Application

1. Start the backend server:
```bash
cd backend
python main.py
```

2. Start the frontend development server:
```bash
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Project Structure

```
plumbing-analysis/
├── src/                    # Frontend source code
│   ├── app/               # Next.js app directory
│   ├── components/        # React components
│   └── styles/           # Global styles
├── backend/               # Backend source code
│   ├── config/           # Configuration files
│   ├── main.py           # FastAPI application
│   └── requirements.txt  # Python dependencies
└── README.md             # Project documentation
```

## API Endpoints

- `POST /api/analyze`: Upload and analyze PDF files
- `GET /api/images/{image_path}`: Retrieve converted images

## Security

- Environment variables are used for sensitive configuration
- CORS is configured for secure cross-origin requests
- API endpoints are protected with proper error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
