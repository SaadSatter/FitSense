# Demo Mode Guide

## What is Demo Mode?

Demo Mode allows you to run FashionSense **without a Gemini API key** by using mock/simulated data. This is perfect for:
- Testing the application
- Development without API costs
- Demonstrating the UI/UX
- Learning how the system works

## How to Enable Demo Mode

Demo Mode is **automatically enabled** when:

1. No `GEMINI_API_KEY` is set in `.env`
2. The `GEMINI_API_KEY` is empty: `GEMINI_API_KEY=`
3. The `GEMINI_API_KEY` is a common placeholder like:
   - `YOUR_API_KEY_HERE`
   - `your_api_key_here`
   - `your_actual_api_key_here`
   - `INSERT_YOUR_KEY_HERE`
   - `REPLACE_WITH_YOUR_KEY`
   - `YOUR-API-KEY`
   - `YOUR_GEMINI_API_KEY`

## Current Setup

Your `.env` file currently has:
```
GEMINI_API_KEY=your_actual_api_key_here
```

This **will trigger Demo Mode** ✅

## Checking Demo Mode Status

When you start the server (`npm start`), look for:

**Demo Mode Enabled:**
```
Checking API Key...
Demo Mode: ENABLED (using mock data)
Reason: No valid API key detected
Found placeholder value: your_actual_api_key_here
```

**Real API Key:**
```
Checking API Key...
API Key: VALID
API Key length: 39
First 10 chars: AIzaSyBw9Z...
```

## Mock Data Behavior

In Demo Mode:

### Image Analysis (Vision)
- Returns realistic mock clothing attributes
- Randomly generates: name, color, texture, category, confidence
- Simulates ~3 different clothing items
- No actual AI image analysis

### Recommendations (LLM)
- Returns pre-written style recommendations
- Matches the context you provide (Office, Casual, etc.)
- Includes realistic outfit combinations
- No actual AI-generated text

## Switching to Real API

To use real Gemini AI:

1. **Get a free API key:**
   - Visit: https://ai.google.dev/
   - Click "Get API Key"
   - Create/select project
   - Copy your key (starts with `AIza...`)

2. **Update `.env`:**
   ```bash
   GEMINI_API_KEY=AIzaSyBw9Z...  # Your actual key
   PORT=3000
   ```

3. **Restart the server:**
   ```bash
   cd server
   npm start
   ```

4. **Verify it worked:**
   - Server should show: "API Key: VALID"
   - At bottom: "Full AI Mode Enabled"

## Troubleshooting

**Issue: "Check that the API key is valid" error**
- ✅ **FIXED** - This was caused by the placeholder not being recognized
- Server now properly detects `your_actual_api_key_here` as a placeholder
- Restart server to apply the fix: `cd server && npm start`

**Issue: Still seeing errors in Demo Mode**
- Check server console for actual error message
- Verify server is running: `http://localhost:3000`
- Try clearing browser cache and refresh

**Issue: Want to test with real API but getting errors**
- Verify API key is correct (starts with `AIza`)
- Check API key has Gemini API enabled
- Verify no extra spaces in `.env` file
- Try the key in Google AI Studio first

## Demo Mode Limitations

Demo Mode provides realistic test data, but:
- ❌ No actual image analysis
- ❌ Same mock recommendations every time
- ❌ Limited variety in responses
- ❌ Can't test edge cases (unusual clothing, etc.)

For full functionality, get a free Gemini API key!

## Mixed Mode (Custom ML + Demo)

You can combine:
- **Custom ML Backend** for image analysis (ViT model)
- **Demo Mode** for recommendations (if no API key)

Just:
1. Keep placeholder in `.env` (Demo Mode for recommendations)
2. Start ML server: `python ml_server.py`
3. Start Node server: `cd server && npm start`
4. Toggle to "Custom ML" in the app
5. Image analysis uses your ViT model
6. Recommendations use mock data (no Gemini API calls)

This lets you test your ML model without an API key!
