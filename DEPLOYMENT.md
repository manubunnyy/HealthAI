# HealthAI Deployment Guide

## Prerequisites
- Python 3.9+
- Node.js 18+
- Git

## Backend Setup

1. **Create virtual environment:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set environment variables:**
Create `backend/.env`:
```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
TOKENIZERS_PARALLELISM=false
```

4. **Run backend:**
```bash
uvicorn main:app --reload --port 8000
```

## Frontend Setup

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Run development server:**
```bash
npm run dev
```

3. **Build for production:**
```bash
npm run build
npm start
```

## Deployment

### Backend (FastAPI)
- Deploy to: Render, Railway, or Heroku
- Use `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend (Next.js)
- Deploy to: Vercel (recommended), Netlify, or Railway
- Set environment variable: `NEXT_PUBLIC_API_URL=your_backend_url`

## Environment Variables

### Backend
- `GOOGLE_API_KEY` - Google Gemini API key
- `GROQ_API_KEY` - Groq API key
- `TOKENIZERS_PARALLELISM` - Set to `false`

### Frontend
- `NEXT_PUBLIC_API_URL` - Backend API URL (default: http://localhost:8000)
