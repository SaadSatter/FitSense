# ML Backend Troubleshooting Guide

## Quick Fix: Port 5000 Conflict (macOS)

**Problem:** Port 5000 is already in use by macOS AirPlay/Control Center

**Solution:** The ML server now uses port 5001 by default ✅

### What Changed
- ML server default port: `5000` → `5001`
- Node.js proxy updated to: `http://localhost:5001`
- No action needed from you!

### If You Need a Different Port
Add to `.env`:
```bash
ML_PORT=5002
ML_BACKEND_URL=http://localhost:5002
```

---

## Error: "Check server is running and API key is valid"

This error appears when the Custom ML backend is selected but the ML server isn't running.

### Steps to Fix:

#### 1. Start the ML Server

**Option A: Use the startup script (Recommended)**
```bash
cd /Users/saadsatter/Documents/GitHub/computer_vision_fashion_sense/FitSense
./start_ml_server.sh
```

**Option B: Run directly**
```bash
cd /Users/saadsatter/Documents/GitHub/computer_vision_fashion_sense/FitSense
python3 ml_server.py
```

You should see:
```
Initializing ML Backend Server...
Loading model from ./custom_ml_model/model/best_multitask_vit.pth...
Using device: cpu (or cuda)
Model loaded successfully!

ML Backend Server starting on port 5001...
Health check: http://localhost:5001/api/ml/health
```

#### 2. Keep the ML Server Running

The ML server must stay running in a separate terminal while you use the app.

**Terminal 1 - ML Server:**
```bash
./start_ml_server.sh
```

**Terminal 2 - Node Server:**
```bash
cd server
npm start
```

#### 3. Test the ML Server

In another terminal:
```bash
curl http://localhost:5001/api/ml/health
```

Expected response:
```json
{
  "status": "ok",
  "message": "ML Backend is running",
  "device": "cpu",
  "model_loaded": true
}
```

---

## Common Errors & Solutions

### Error: "Model file not found"

**Cause:** `best_multitask_vit.pth` doesn't exist

**Solution:**
```bash
cd custom_ml_model/model
bash download_models.sh
```

Or train your own:
```bash
python custom_ml_model/train.py
```

---

### Error: "No module named 'flask'"

**Cause:** Python dependencies not installed

**Solution:**
```bash
pip install -r requirements_ml_backend.txt
```

---

### Error: "No module named 'train'"

**Cause:** Old import path in ml_server.py

**Solution:** Already fixed! The server now imports from `model_architecture.py`

If you still see this, make sure line 20 of `ml_server.py` says:
```python
from model_architecture import MultiTaskViT
```

NOT:
```python
from train import MultiTaskViT
```

---

### Error: "[Errno 2] No such file or directory: 'fashion-product-images-small.zip'"

**Cause:** `train.py` tries to extract dataset when imported

**Solution:** Already fixed! We now import from `model_architecture.py` instead

---

### Error: "Connection refused" or "Failed to analyze"

**Causes:**
1. ML server not running
2. Wrong port configured
3. Firewall blocking connection

**Solution:**

**Check if ML server is running:**
```bash
lsof -i :5001 | grep LISTEN
```

Should show `Python` or `python3` process

**Check Node.js is configured correctly:**
```bash
grep ML_BACKEND_URL server/server.js
```

Should show: `http://localhost:5001`

**Restart both servers:**
```bash
# Terminal 1
./start_ml_server.sh

# Terminal 2
cd server && npm start
```

---

### Slow Performance (First Request)

**Cause:** Model loading takes 30-60 seconds on first request

**Solution:** This is normal! Subsequent requests are much faster (1-2s per image)

The model stays loaded in memory after the first request.

---

### Error: "CUDA out of memory"

**Cause:** GPU doesn't have enough VRAM

**Solution:** The server automatically falls back to CPU. Performance will be slower but it will work.

If you want to force CPU mode:
```bash
CUDA_VISIBLE_DEVICES="" python3 ml_server.py
```

---

## Verification Checklist

Before using Custom ML backend, verify:

- ✅ Model file exists: `ls -lh custom_ml_model/model/best_multitask_vit.pth`
- ✅ Python packages installed: `pip list | grep -E 'flask|torch|transformers'`
- ✅ ML server running: `curl http://localhost:5001/api/ml/health`
- ✅ Node server running: `curl http://localhost:3000/api/health`
- ✅ Gemini API key set (for recommendations): `echo $GEMINI_API_KEY`

---

## Debug Mode

### Enable Verbose Logging

**ML Server:**
Already logs each step. Watch the terminal for:
```
Analyzing 3 wardrobe items for context: Office
  Analyzing wardrobe item 1/3...
    Predictions: Shirts, Blue, Formal
  ...
Vision output: 2 tops, 1 bottom, 0 onepieces
  Getting outfit recommendation from Gemini...
```

**Browser Console:**
Open DevTools (F12) and check Console tab for:
```
Using ML backend for wardrobe analysis...
Analyzing 3 wardrobe items with ML backend for: Office
```

**Node Server:**
Check terminal for:
```
Forwarding 3 wardrobe items to ML backend for context: Office
```

---

## Still Having Issues?

### Restart Everything

```bash
# Stop both servers (Ctrl+C in each terminal)

# Clear any cached processes
pkill -f ml_server.py
pkill -f "node.*server.js"

# Start fresh
# Terminal 1
cd /Users/saadsatter/Documents/GitHub/computer_vision_fashion_sense/FitSense
./start_ml_server.sh

# Wait for "Model loaded successfully!"

# Terminal 2
cd /Users/saadsatter/Documents/GitHub/computer_vision_fashion_sense/FitSense/server
npm start

# Browser: Hard refresh (Cmd+Shift+R)
```

### Check Port Usage

```bash
# See what's on port 5001
lsof -i :5001

# See what's on port 3000
lsof -i :3000
```

### Test Manually

**1. Test ML server directly:**
```bash
curl -X POST http://localhost:5001/api/ml/analyze-images \
  -F "images=@path/to/test-image.jpg"
```

**2. Test through Node proxy:**
```bash
curl -X POST http://localhost:3000/api/ml/analyze-images \
  -F "images=@path/to/test-image.jpg"
```

---

## Summary

**Most common issue:** ML server not running

**Quick fix:**
```bash
# Terminal 1
./start_ml_server.sh

# Terminal 2  
cd server && npm start

# Browser
Toggle to "Backend: Custom ML"
```

**Remember:** The ML server must stay running in a separate terminal!
