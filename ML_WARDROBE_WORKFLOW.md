# ML Wardrobe Workflow

## Overview

When the **Custom ML backend** is selected, wardrobe recommendations use a special workflow that leverages your trained ViT model and the outfit selection logic from `vit_llm_infer.py`.

## Architecture

### Standard Gemini Workflow (Wardrobe)
```
Frontend (Wardrobe) → Convert to Files → Node.js → Gemini API → Results
```

### Custom ML Workflow (Wardrobe) ✨
```
Frontend (Wardrobe) → Send base64 Images → Node.js Proxy → 
Python ML Server → ViT Inference (vit_infer.py logic) → 
LLM Outfit Selection (vit_llm_infer.py logic) → Results
```

## How It Works

### 1. User Selects Wardrobe Items
- User selects 2-10 items from their saved wardrobe
- Images are stored as base64 data URLs in IndexedDB
- User chooses an occasion/context (Office, Casual, etc.)

### 2. Frontend Detects ML Backend
When "Get Style Recommendation" is clicked:
- `wardrobe.js` checks `state.useMLBackend`
- If `true`, calls `getWardrobeRecommendationML()` from `ml-wardrobe.js`
- If `false`, uses standard Gemini workflow

### 3. ML Wardrobe Analysis Endpoint
**Frontend → Node.js:**
```javascript
POST /api/ml/analyze-wardrobe
{
  "wardrobeItems": [
    { "id": "...", "imageData": "data:image/jpeg;base64,...", "fileName": "..." },
    ...
  ],
  "context": "Office"
}
```

**Node.js → Python ML Server:**
- Forwards request to `http://localhost:5000/api/ml/analyze-wardrobe`
- Same JSON payload

### 4. Python ML Server Processing

**Step 1: ViT Inference on Each Item** (`predict()` from `vit_infer.py`)
```python
for each wardrobe_item:
    1. Decode base64 image data
    2. Save to temporary file
    3. Load image with PIL
    4. Run ViT inference
    5. Predict: gender, articleType, baseColour, season, usage
    6. Clean up temp file
```

**Step 2: Categorize Items** (`build_vision_output()` from `vit_llm_infer.py`)
```python
vision_output = {
    "tops": [...],      # Shirts, Blazers, etc.
    "bottoms": [...],   # Jeans, Trousers, etc.
    "onepieces": [...], # Dresses, Jumpsuits, etc.
    "event": "Office"
}
```

**Step 3: LLM Outfit Selection** (`build_prompt()` + Gemini from `vit_llm_infer.py`)
```python
1. Build structured prompt with categorized items
2. Call Gemini API with OutfitSelection schema
3. LLM selects:
   - Best top + bottom combination, OR
   - Best one-piece outfit
4. Returns: outfit_type, selected items, style, fit, reason
```

**Step 4: Format & Return**
```python
{
  "success": true,
  "recommendation": "**Style:** Business Casual\n\n**Selected Outfit:**...",
  "selectedItems": [0, 3],  # Indices of selected wardrobe items
  "visionOutput": {...},
  "predictions": [...]
}
```

### 5. Frontend Displays Results
- Converts ML predictions to frontend-compatible format
- Highlights selected wardrobe items
- Shows recommendation with styling advice
- User can save to history

## File Structure

```
FitSense/
├── ml_server.py                      # ML backend Flask server
│   └── /api/ml/analyze-wardrobe      # New wardrobe endpoint
├── custom_ml_model/
│   ├── vit_infer.py                  # ViT inference logic (used by ml_server)
│   ├── vit_llm_infer.py              # Outfit selection logic (used by ml_server)
│   └── model/
│       └── best_multitask_vit.pth    # Trained model
├── server/
│   └── server.js                     # Node.js proxy
│       └── POST /api/ml/analyze-wardrobe
└── frontend/js/
    ├── ml-wardrobe.js                # New: ML wardrobe analysis
    ├── wardrobe.js                   # Updated: detects ML backend
    └── state.js                      # useMLBackend flag
```

## Usage

### Requirements
1. Trained ViT model at `./custom_ml_model/model/best_multitask_vit.pth`
2. Gemini API key (for outfit selection)
3. ML server running: `python ml_server.py`
4. Node server running: `cd server && npm start`

### Steps
1. **Start ML Server:**
   ```bash
   python ml_server.py
   # Output: ML Backend Server starting on port 5000...
   ```

2. **Start Node Server:**
   ```bash
   cd server && npm start
   # Output: Fashion Sense Server Running!
   ```

3. **Use the App:**
   - Open `http://localhost:3000`
   - Toggle to **"Backend: Custom ML"** in navbar
   - Go to **"My Wardrobe"** tab
   - Select 2-10 clothing items (click to select)
   - Choose an occasion
   - Click **"Get Style Recommendation"**

4. **ML Processing:**
   - ViT analyzes each wardrobe item individually
   - Items categorized as tops/bottoms/onepieces
   - Gemini LLM selects best outfit combination
   - Results displayed with selected items highlighted

## Advantages

### vs. Standard Upload Flow
- ✅ **Uses wardrobe items directly** - no need to re-upload
- ✅ **Preserves image quality** - no re-encoding
- ✅ **Persistent storage** - wardrobe items saved in IndexedDB

### vs. Gemini Wardrobe Flow
- ✅ **Uses your trained model** - custom ViT predictions
- ✅ **Local image analysis** - privacy-preserving
- ✅ **Structured outfit logic** - explicit top/bottom/onepiece categorization
- ✅ **Consistent with vit_llm_infer.py** - same logic as CLI tool

## API Reference

### `/api/ml/analyze-wardrobe` (POST)

**Request:**
```json
{
  "wardrobeItems": [
    {
      "id": "wardrobe_item_id",
      "imageData": "data:image/jpeg;base64,...",
      "fileName": "shirt.jpg"
    }
  ],
  "context": "Office"
}
```

**Response:**
```json
{
  "success": true,
  "recommendation": "**Style:** Business Casual\n\n**Selected Outfit:**\n- Top: Men Blue Formal Shirt\n- Bottom: Men Black Casual Trousers\n\n**Fit Assessment:** Appropriate\n\n**Reasoning:** This combination...",
  "selectedItems": [0, 3],
  "visionOutput": {
    "tops": [...],
    "bottoms": [...],
    "onepieces": [],
    "event": "Office"
  },
  "predictions": [
    {
      "gender": "Men",
      "articleType": "Shirts",
      "baseColour": "Blue",
      "season": "Summer",
      "usage": "Formal",
      "wardrobe_item_id": "...",
      "index": 0
    }
  ]
}
```

## Debugging

**Check ML server logs:**
```
Analyzing 5 wardrobe items for context: Office
  Analyzing wardrobe item 1/5...
    Predictions: Shirts, Blue, Formal
  Analyzing wardrobe item 2/5...
    Predictions: Trousers, Black, Casual
  ...

Vision output: 3 tops, 2 bottoms, 0 onepieces
  Getting outfit recommendation from Gemini...
```

**Check Node server logs:**
```
Forwarding 5 wardrobe items to ML backend for context: Office
```

**Check browser console:**
```
Using ML backend for wardrobe analysis...
Analyzing 5 wardrobe items with ML backend for: Office
```

## Troubleshooting

**Error: "Failed to analyze wardrobe with ML backend"**
- ✅ Ensure `python ml_server.py` is running
- ✅ Check ML server logs for errors
- ✅ Verify model file exists: `./custom_ml_model/model/best_multitask_vit.pth`

**Error: "No module named 'train'"**
- ✅ Ensure `train.py` exists in `custom_ml_model/`
- ✅ Check sys.path in `ml_server.py`

**Error: "Connection refused"**
- ✅ ML server must be on port 5000
- ✅ Check firewall settings
- ✅ Verify `ML_BACKEND_URL` in `.env`

**No items highlighted in results**
- This is expected - wardrobe recommendations show all items
- `selectedItems` array indicates which were chosen by LLM
- Future enhancement: visual highlighting

## Future Enhancements

- [ ] Visual highlighting of selected items in results view
- [ ] Show confidence scores from ViT model
- [ ] Support for attribute-based filtering
- [ ] Batch processing optimization
- [ ] Cache ViT predictions for wardrobe items
- [ ] Allow mixing wardrobe + uploaded items

## Performance

**Typical timing (M1 Mac / RTX 3060):**
- 5 wardrobe items: ~5-7 seconds total
  - ViT inference: ~1-2 seconds per item
  - LLM outfit selection: ~2-3 seconds
- First request slower (model loading cached)

Enjoy your AI-powered wardrobe styling! 👔👗✨
