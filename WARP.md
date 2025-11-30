# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

FitSense (also referred to as FashionSense in some files) is an AI-powered fashion recommendation system that combines computer vision and LLM capabilities. The system analyzes clothing images, extracts attributes, and provides personalized style recommendations based on context/occasion.

**Key Technologies:**
- **ML Backend:** PyTorch, Vision Transformers (ViT), U2NET for cloth segmentation
- **Web Backend:** Node.js + Express
- **Frontend:** Vanilla JavaScript with ES6 modules, IndexedDB for storage
- **AI Services:** Google Gemini API for image analysis and recommendations

## Architecture

### Three-Tier System

1. **ML Layer (Python):**
   - `model_architecture.py`: MultiTaskViT model definition (5 attributes: gender, articleType, baseColour, season, usage)
   - `train.py`: Training script for MultiTaskViT (downloads dataset, trains model, saves checkpoint)
   - `cloth_segmentation.py`: U2NET-based segmentation that extracts clothing items from images
   - `vit_infer.py`: Single-image inference using trained ViT model
   - `vit_llm_infer.py`: Batch inference + Gemini integration for outfit recommendations
   - `networks.py`: U2NET architecture implementation (RSU modules)
   - `ml_server.py`: Flask API server that wraps the ML backend for Node.js communication

2. **Node.js API Server (`server/`):**
   - `server.js`: Express server with two main endpoints:
     - `POST /api/analyze-images`: Analyzes clothing images (Gemini Vision API)
     - `POST /api/get-recommendation`: Gets style recommendations based on context
   - `gemini.js`: Gemini API integration with automatic demo mode fallback
   - `mock.js`: Mock data for testing without API keys
   - Demo mode activates if `GEMINI_API_KEY` is missing or invalid

3. **Frontend (`frontend/`):**
   - Modular ES6 architecture with view-based navigation
   - `app.js`: Main entry point, dependency injection, view callbacks
   - `upload.js`: Image upload handling
   - `analysis.js`: Communicates with backend APIs
   - `results.js`: Displays recommendations
   - `wardrobe.js`: Persistent wardrobe management (IndexedDB)
   - `history.js`: Recommendation history (IndexedDB)
   - `db.js`: IndexedDB wrapper
   - `navigation.js`: View routing system
   - `state.js`: Global state management

## Common Commands

### Setup and Installation

**Python ML Environment:**
```bash
pip install -r requirements.txt
```

**Node.js Server:**
```bash
cd server
npm install
```

**Environment Configuration:**
Create `.env` file in project root:
```bash
GEMINI_API_KEY=your_api_key_here
PORT=3000
```

### Running the Application

**Start Server (with demo mode if no API key):**
```bash
cd server
npm start
```

**Start Server with Auto-reload (development):**
```bash
cd server
npm run dev
```

**Access Frontend:**
Open `http://localhost:3000` in browser after starting server.

### ML Model Operations

**Download Pretrained Models (Recommended):**
```bash
cd custom_ml_model/model
bash download_models.sh
```
- Downloads `best_multitask_vit.pth` (~340 MB) - Required for ML backend
- Downloads `checkpoint_u2net.pth` (~176 MB) - Optional, for cloth segmentation
- See `custom_ml_model/model/README.md` for details

**Train Multi-task ViT Model (Optional - Only if you want to retrain):**
```bash
kaggle datasets download -d paramaggarwal/fashion-product-images-small
python custom_ml_model/train.py
```
- Trains on Kaggle fashion dataset (apparel category only)
- Saves best model to `best_multitask_vit.pth`
- Move trained model to `custom_ml_model/model/` for use with ML server
- Batch size: 32, Learning rate: 2e-5, Epochs: 10
- Uses AdamW optimizer with task-weighted loss
- **Note:** The ML server does NOT run training - it only loads pretrained models

**Run Single Image Inference:**
```bash
python custom_ml_model/vit_infer.py <image_path>
```

**Run Batch Inference with LLM Recommendations:**
```bash
python custom_ml_model/vit_llm_infer.py <event/context> <image_path1> [<image_path2> ...]
```
Example: `python custom_ml_model/vit_llm_infer.py "business meeting" img1.jpg img2.jpg`

**Run Cloth Segmentation:**
```bash
python custom_ml_model/cloth_segmentation.py
```
- Processes images from `./input/raw_images/`
- Outputs segmentation masks to `./input/masks/`
- Outputs segmented clothes to `./output/segmented_clothes/`
- Requires U2NET checkpoint at `custom_ml_model/model/checkpoint_u2net.pth`
- Creates 3 versions: black background, white background, transparent PNG

## Key Implementation Details

### Multi-Task Vision Transformer (model_architecture.py)

The `MultiTaskViT` class uses a pretrained ViT backbone (`google/vit-base-patch16-224-in21k`) with 5 separate classification heads:
- **gender**: Male/Female classification
- **articleType**: Specific clothing type (shirts, jeans, dresses, etc.)
- **baseColour**: Primary color
- **season**: Seasonal appropriateness
- **usage**: Occasion type (formal, casual, sports, etc.)

The model architecture is separated from training code in `model_architecture.py` so that:
- ML server (`ml_server.py`) can import the model without running training code
- Inference scripts (`vit_infer.py`, `vit_llm_infer.py`) can load pretrained models
- Training script (`train.py`) can use the same architecture definition

Task weights can be adjusted in training (default: articleType=1.5, others=1.0).

### U2NET Segmentation Architecture (networks.py)

Implements Residual U-blocks (RSU) at multiple scales (RSU-7, RSU-6, RSU-5, RSU-4, RSU-4F) for salient object detection. The network outputs 4-class segmentation:
- Class 0: Background
- Classes 1-3: Different clothing regions

### Gemini Integration Pattern (server/gemini.js)

The system uses a two-stage Gemini API approach:

**Stage 1 - Image Analysis:**
- Each image analyzed independently with parallel requests
- Returns structured JSON: `{isClothing, name, color, texture, category, confidence}`
- Includes retry logic for transient network errors
- Graceful fallback to mock data in demo mode

**Stage 2 - Recommendation:**
- Takes analyzed attributes + user context
- Returns: `{selectedItems: [indices], recommendation: "formatted text"}`
- Uses specific prompt engineering for Title Case item names
- Never uses "Item 0", "Item 1" format - always uses descriptive names

### Frontend State Management

State is managed through multiple mechanisms:
- **In-Memory:** `state.js` for current session (uploaded images, attributes)
- **IndexedDB:** Persistent storage for wardrobe items and recommendation history
- **URL Parameters:** View routing (`#view=upload|wardrobe|history|results`)

Migration from localStorage to IndexedDB happens automatically on app initialization.

### Garment Type Detection (vit_llm_infer.py)

Clothing items are categorized into three types:
- **TOPS**: Shirts, Tshirts, Jackets, Sweaters, Blazers, etc.
- **BOTTOMS**: Jeans, Trousers, Skirts, Shorts, Leggings, etc.
- **ONEPIECES**: Dresses, Jumpsuits, Rompers, Kurta Sets, etc.

The LLM recommendation logic handles both top+bottom combinations and standalone one-piece outfits.

## Development Patterns

### Adding New Clothing Attributes

When extending the multi-task model:
1. Update `num_classes_dict` in `MultiTaskViT.__init__`
2. Add new classifier head: `self.new_task_classifier = nn.Linear(hidden_size, num_classes_dict['new_task'])`
3. Update `forward()` return dictionary
4. Update `train_epoch()` and `evaluate()` loops to include new task
5. Add task to `prepare_data()` label encoding
6. Update `FashionDataset.__getitem__` labels dictionary

### Adding New API Endpoints

Follow the pattern in `server/server.js`:
1. Use multer for file uploads if needed
2. Check for demo mode early: `if (isDemoMode()) { ... }`
3. Return consistent JSON structure: `{success: true/false, ...data}`
4. Include comprehensive error handling with try-catch
5. Log operations for debugging

### Frontend Module Communication

To avoid circular dependencies, the codebase uses callback injection:
- `setGetRecommendationCallback()` in upload.js
- `setDisplayResultsCallback()` in analysis.js
- `registerViewCallback()` in navigation.js

When adding cross-module functionality, prefer callbacks over direct imports.

## Important File Locations

- **Model Checkpoints:** `./model/` (create if missing)
  - `best_multitask_vit.pth`: Trained ViT model
  - `checkpoint_u2net.pth`: U2NET segmentation weights
- **Dataset:** `./datasets/`
  - `styles.csv`: Fashion metadata
  - `images/`: Product images
- **Input/Output:** 
  - `./input/raw_images/`: Source images for segmentation
  - `./input/masks/`: Generated segmentation masks
  - `./output/segmented_clothes/`: Final extracted clothing images
- **Environment:** `.env` in project root (not in server/)
- **Server:** `server/` directory
- **Frontend:** `frontend/` directory with `index.html` and `js/` modules

## Data Requirements

The training script expects:
- Kaggle Fashion Product Images (small) dataset
- CSV format with columns: id, gender, masterCategory, subCategory, articleType, baseColour, season, usage, year, productDisplayName
- Filters to "Apparel" masterCategory only
- Excludes "Boys" and "Girls" gender categories
- Images must exist in `./datasets/images/{id}.jpg`

## Error Handling Patterns

**Python ML Code:**
- Use `torch.cuda.is_available()` for device selection
- Always provide CPU fallback: `map_location=torch.device("cpu")`
- Check file existence before loading checkpoints
- Use `on_bad_lines='skip'` when reading CSVs

**Node.js Server:**
- Demo mode for missing API keys (no hard failures)
- Retry logic for transient network errors (2 retries with exponential backoff)
- Always return valid JSON even on errors
- Return safe fallback data if individual image processing fails

**Frontend:**
- Graceful degradation if IndexedDB fails
- Toast notifications for user-facing errors
- Console warnings for non-critical failures
- View initialization proceeds even if storage fails

## Copyright and Usage

This project was created for academic purposes by Saad Satter (2025). Commercial use requires written permission. Contributors may modify for educational purposes only.
