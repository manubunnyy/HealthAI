# HealthAI - Vercel Deployment

## Quick Deploy

### Option 1: Deploy Frontend Only (Recommended)

1. **In Vercel Dashboard:**
   - Import your GitHub repository
   - **Root Directory:** Set to `frontend`
   - **Framework Preset:** Next.js
   - Click "Deploy"

### Option 2: Use Vercel CLI

```bash
cd frontend
vercel
```

## Environment Variables

Add these in Vercel Dashboard → Settings → Environment Variables:

```
NEXT_PUBLIC_API_URL=https://your-backend-url.com
```

## Backend Deployment

Deploy backend separately on:
- **Render** (recommended for FastAPI)
- **Railway**
- **Heroku**

### Render Deployment (Backend):

1. Create new Web Service
2. Connect your GitHub repo
3. **Root Directory:** `backend`
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - `GOOGLE_API_KEY`
   - `GROQ_API_KEY`
   - `TOKENIZERS_PARALLELISM=false`

## Update Frontend API URL

After deploying backend, update the frontend API calls:

1. In Vercel, set environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
   ```

2. Update frontend code to use:
   ```javascript
   const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
   ```

## Troubleshooting

### 404 Error
- Make sure Root Directory is set to `frontend` in Vercel
- Check that `frontend/package.json` exists

### Build Fails
- Ensure all dependencies are in `frontend/package.json`
- Check build logs for specific errors

### API Errors
- Verify backend is deployed and running
- Check CORS settings in backend
- Verify `NEXT_PUBLIC_API_URL` is set correctly
