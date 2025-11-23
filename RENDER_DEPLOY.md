# Render Backend Deployment Guide

## Quick Setup

### 1. Create New Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `https://github.com/manubunnyy/HealthAI`

### 2. Configure Service

**Basic Settings:**
- **Name:** `healthai-backend` (or your choice)
- **Region:** Choose closest to you
- **Branch:** `major`
- **Root Directory:** `backend`
- **Runtime:** `Python 3`

**Build & Deploy:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 3. Environment Variables

Add these in Render Dashboard → Environment:

```
GOOGLE_API_KEY=your_actual_google_api_key
GROQ_API_KEY=your_actual_groq_api_key
TOKENIZERS_PARALLELISM=false
PYTHON_VERSION=3.9.18
```

### 4. Advanced Settings (Optional)

- **Auto-Deploy:** Yes (deploys on git push)
- **Health Check Path:** `/` or `/health`

## Common Deployment Errors & Fixes

### Error 1: "Module not found"
**Cause:** Missing dependencies in requirements.txt
**Fix:** 
```bash
# Locally, generate fresh requirements
cd backend
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push origin major
```

### Error 2: "Port binding failed"
**Cause:** Not using Render's `$PORT` variable
**Fix:** Ensure start command is:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Error 3: "Build failed - Python version"
**Cause:** Python version mismatch
**Fix:** Add environment variable:
```
PYTHON_VERSION=3.9.18
```

### Error 4: "Import Error: google.generativeai"
**Cause:** Missing API keys or dependencies
**Fix:**
1. Check environment variables are set
2. Ensure `google-generativeai` is in requirements.txt

### Error 5: "Torch/Large dependencies timeout"
**Cause:** Build takes too long
**Fix:** 
- Remove unused heavy dependencies
- Or upgrade to paid plan for longer build time

## Check Your Deployment Logs

### Via Dashboard:
1. Go to https://dashboard.render.com/
2. Click on your service
3. Go to "Logs" tab
4. Look for errors in red

### Common Log Errors:

**"No module named 'app'"**
```bash
# Fix: Ensure main.py is in backend/ directory
# Check Root Directory is set to "backend"
```

**"Address already in use"**
```bash
# Fix: Use $PORT variable in start command
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**"Failed to load .env"**
```bash
# Fix: Don't rely on .env file, use Render's environment variables
```

## Your Deployment ID

Your service ID: `rnd_930QWWLbskBkfi8XcmaSPnIj7vZg`

To check logs:
1. Go to: https://dashboard.render.com/
2. Find service with this ID
3. Click "Logs" to see build/runtime errors

## Test Your Deployment

Once deployed, test with:

```bash
# Replace with your Render URL
curl https://your-app.onrender.com/

# Should return: {"message": "HealthAI Backend API"}
```

## Update Frontend with Backend URL

After successful deployment:

1. Copy your Render URL (e.g., `https://healthai-backend.onrender.com`)
2. In Vercel, add environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://healthai-backend.onrender.com
   ```
3. Redeploy frontend

## Troubleshooting Checklist

- [ ] Root Directory set to `backend`
- [ ] Build command: `pip install -r requirements.txt`
- [ ] Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] All environment variables added
- [ ] Python version specified (3.9.18)
- [ ] requirements.txt is complete
- [ ] No .env file in repository (use Render env vars)

## Need Help?

1. **Check Render Logs:** Most errors show there
2. **Common fix:** Redeploy after fixing config
3. **Still stuck?** Share the error message from logs
