# MindPulse Deployment Guide

## 📦 Deployment to Render

This guide will help you deploy the MindPulse application to Render (formerly Render Cloud).

### Prerequisites
- GitHub account with the repository pushed
- Render account (https://render.com)
- Git installed locally

### Step 1: Prepare Your Repository

1. Make sure all deployment files are in place:
   - `requirements.txt` - Python dependencies
   - `runtime.txt` - Python version specification
   - `Procfile` - Process configuration
   - `render.yaml` - Render-specific configuration
   - `app.py` - Main Flask application
   - `.gitignore` - Git ignore rules

2. Commit all changes:
   ```bash
   git add .
   git commit -m "Add deployment configuration for Render"
   git push origin main
   ```

### Step 2: Connect Render to GitHub

1. Go to https://render.com and sign up/log in
2. Click "New +" button and select "Web Service"
3. Choose "Deploy an existing repository"
4. Connect your GitHub account and select your repository

### Step 3: Configure the Web Service

1. **Name**: `mindpulse`
2. **Environment**: `Python 3`
3. **Region**: Select closest to your users (e.g., Oregon, Frankfurt)
4. **Plan**: Choose your plan (free tier available)

5. **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Start Command**:
   ```bash
   gunicorn app:app
   ```

7. **Environment Variables** (optional):
   - Add any environment variables your app needs

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will automatically start the build and deployment
3. Wait for the deployment to complete (2-5 minutes)
4. Your app will be available at: `https://mindpulse-xxxxx.onrender.com`

### Step 5: Verify Deployment

Test your endpoints:

```bash
# Health check
curl https://your-app-url.onrender.com/health

# Sentiment analysis
curl -X POST https://your-app-url.onrender.com/api/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this application!"}'

# Emotion detection
curl -X POST https://your-app-url.onrender.com/api/emotion \
  -H "Content-Type: application/json" \
  -d '{"text":"I am very happy today!"}'

# Sarcasm detection
curl -X POST https://your-app-url.onrender.com/api/sarcasm \
  -H "Content-Type: application/json" \
  -d '{"text":"Oh sure, that was totally great!"}'
```

### Available API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve index.html |
| `/health` | GET | Health check |
| `/api/sentiment` | POST | Analyze sentiment |
| `/api/emotion` | POST | Detect emotions |
| `/api/sarcasm` | POST | Detect sarcasm |

### Request Format

All POST endpoints expect JSON:
```json
{
  "text": "Your text here"
}
```

### Response Format

**Sentiment Response**:
```json
{
  "polarity": 0.5,
  "subjectivity": 0.7,
  "confidence": 75.0,
  "verdict": "Positive",
  "tone": "positive",
  "scores": {"pos": 75.0, "neg": 15.0, "neu": 10.0}
}
```

**Emotion Response**:
```json
{
  "text": "I am very happy",
  "top_emotion": "joy",
  "confidence": 85.5,
  "scores": {"joy": 85.5, "sadness": 5.0, ...}
}
```

**Sarcasm Response**:
```json
{
  "text": "Oh, that's brilliant",
  "is_sarcastic": true,
  "confidence": 78.5,
  "score": 0.785
}
```

### Troubleshooting

#### Build Fails
- Check `requirements.txt` for correct package versions
- Ensure Python version in `runtime.txt` is supported
- Check Render's build logs for specific errors

#### Model Loading Errors
- Ensure model files are in the repository (git-lfs may be needed for large files)
- Check file paths in `app.py` match your repository structure

#### Service Timeout
- Free tier on Render spins down after inactivity
- Upgrade to a paid plan for always-on service

#### CORS Errors
- CORS is enabled in `app.py` with `*` origin
- Modify the `add_cors_headers` function if you need restricted origins

### Monitor Your Deployment

1. Go to your service dashboard on Render
2. View logs: Click on your service → Logs tab
3. View metrics: Click on your service → Metrics tab
4. Restart service: Click on your service → Manual Deploy

### Update Your Deployment

To redeploy after making changes:

1. Push changes to GitHub:
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin main
   ```

2. On Render, click "Manual Deploy" on the service page

Or enable "Auto-Deploy" to automatically redeploy on every push to main branch.

### Free Tier Limitations

- Service spins down after 15 minutes of inactivity
- First request after spin-down may be slow
- Upgrade to paid plan for always-on service

### Next Steps

1. Update your frontend HTML files to use the deployed API URLs
2. Configure database if needed
3. Set up custom domain (optional)
4. Monitor performance and logs

---

For more information, visit: https://render.com/docs
