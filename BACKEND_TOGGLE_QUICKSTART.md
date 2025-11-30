# Backend Toggle Feature - Quick Start

## What's New?

You can now switch between **Gemini AI** and your **Custom ML Backend** using a toggle button in the navigation bar!

## Quick Setup (3 Steps)

### 1. Install Dependencies

```bash
# Install Node.js dependencies
cd server
npm install

# Install Python ML dependencies
cd ..
pip install -r requirements_ml_backend.txt
```

### 2. Start Both Servers

**Terminal 1 - ML Backend:**
```bash
python ml_server.py
```

**Terminal 2 - Node.js Server:**
```bash
cd server
npm start
```

### 3. Toggle Backends

1. Open `http://localhost:3000`
2. Click **"Backend: Gemini"** button in the navbar
3. Switch to **"Backend: Custom ML"** to use your trained model
4. Switch back anytime!

## What You Need

✅ Trained ViT model at `./model/best_multitask_vit.pth`  
✅ Gemini API key in `.env` file (for recommendations)  
✅ Both servers running simultaneously  

## Default Behavior

- **By default**: Uses Gemini backend (no ML server needed)
- **After toggle**: Uses your custom ML backend
- **State persists**: Your selection stays until you change it

## Files Changed

**New Files:**
- `ml_server.py` - Flask server for ML backend
- `requirements_ml_backend.txt` - Python dependencies
- `ML_BACKEND_README.md` - Full documentation
- `BACKEND_TOGGLE_QUICKSTART.md` - This file

**Modified Files:**
- `frontend/index.html` - Added toggle button
- `frontend/styles.css` - Toggle button styling
- `frontend/js/state.js` - Added `useMLBackend` state
- `frontend/js/app.js` - Toggle button logic
- `frontend/js/analysis.js` - Dynamic backend switching
- `server/server.js` - ML backend proxy endpoints
- `server/package.json` - Added form-data, node-fetch

## Benefits

🎯 **Flexibility**: Switch backends without restarting  
🔒 **Privacy**: ML analysis happens locally  
💰 **Cost**: Reduce Gemini API usage  
🎓 **Learning**: See your model in action  
⚡ **Performance**: Compare both backends  

## Troubleshooting

**Toggle button doesn't appear?**
- Clear browser cache
- Hard refresh (Cmd+Shift+R / Ctrl+Shift+F5)

**ML backend errors?**
- Ensure `python ml_server.py` is running
- Check model file exists: `./model/best_multitask_vit.pth`
- Verify dependencies: `pip list | grep flask`

**Still using Gemini after toggle?**
- Check browser console for errors
- Verify both servers are running
- Check ML server URL in `.env`: `ML_BACKEND_URL=http://localhost:5000`

## Full Documentation

See `ML_BACKEND_README.md` for:
- Detailed architecture
- API endpoints
- Configuration options
- Performance benchmarks
- Development notes

Enjoy your dual-backend system! 🚀
