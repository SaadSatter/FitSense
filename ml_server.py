#!/usr/bin/env python3
"""
Flask server wrapper for the ML backend (ViT model + Gemini LLM)
This allows the Node.js server to communicate with the Python ML pipeline.
"""

import os
import sys
import json
import tempfile
from typing import List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image

# Add custom_ml_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'custom_ml_model'))

from model_architecture import MultiTaskViT
from vit_llm_infer import (
    load_model, predict, detect_garment_type, build_vision_output,
    build_prompt, OutfitSelection
)
from transformers import ViTImageProcessor
from google import genai  # Using google-genai package

app = Flask(__name__)
CORS(app)

# Global model variables (loaded once at startup)
model = None
processor = None
label_encoders = None
device = None

def init_model():
    """Initialize the ML model at server startup"""
    global model, processor, label_encoders, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./custom_ml_model/model/best_multitask_vit.pth"
    model_name = "google/vit-base-patch16-224-in21k"
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Using device: {device}")
    
    model, processor, label_encoders = load_model(checkpoint_path, model_name, device)
    print("Model loaded successfully!")

def format_attribute_for_frontend(predictions: Dict[str, str], image_index: int) -> Dict:
    """
    Convert ML backend predictions to Gemini-like format for frontend compatibility
    """
    article_type = predictions.get('articleType', 'Unknown')
    base_colour = predictions.get('baseColour', 'Unknown')
    gender = predictions.get('gender', 'Unisex')
    usage = predictions.get('usage', 'Casual')
    
    # Create a name similar to Gemini's format
    name = f"{gender} {base_colour} {article_type}"
    
    return {
        "isClothing": True,
        "name": name.title(),
        "color": base_colour.title(),
        "texture": "Unknown",  # ViT doesn't predict texture
        "category": article_type.title(),
        "confidence": 0.85,  # Mock confidence since ViT doesn't provide it
        # Include original ML predictions for LLM
        "mlPredictions": predictions
    }

@app.route('/api/ml/analyze-images', methods=['POST'])
def analyze_images():
    """
    Analyze images using the ViT model
    Expected: multipart/form-data with 'images' field containing image files
    """
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({"error": "No images provided"}), 400
        
        print(f"Analyzing {len(files)} images with ML backend...")
        
        attributes = []
        temp_files = []
        
        try:
            # Save uploaded files temporarily and run inference
            for idx, file in enumerate(files):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    file.save(tmp.name)
                    temp_files.append(tmp.name)
                    
                    # Run inference
                    print(f"Processing image {idx + 1}/{len(files)}: {file.filename}")
                    predictions = predict(model, tmp.name, processor, label_encoders, device)
                    
                    # Format for frontend
                    attr = format_attribute_for_frontend(predictions, idx)
                    attributes.append(attr)
                    
                    print(f"  Predictions: {predictions}")
        
        finally:
            # Clean up temporary files
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except:
                    pass
        
        return jsonify({
            "success": True,
            "attributes": attributes,
            "count": len(attributes)
        })
        
    except Exception as e:
        print(f"Error in analyze-images: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to analyze images",
            "message": str(e)
        }), 500

@app.route('/api/ml/analyze-wardrobe', methods=['POST'])
def analyze_wardrobe():
    """
    Analyze wardrobe images using ViT model and get outfit recommendation
    Expected JSON: { "wardrobeItems": [{id, imageData}], "context": "..." }
    This uses vit_infer.py for analysis and vit_llm_infer.py for recommendations
    """
    try:
        data = request.json
        if not data or 'wardrobeItems' not in data or 'context' not in data:
            return jsonify({"error": "Missing wardrobeItems or context"}), 400
        
        wardrobe_items = data['wardrobeItems']
        context = data['context']
        
        if not wardrobe_items:
            return jsonify({"error": "No wardrobe items provided"}), 400
        
        print(f"Analyzing {len(wardrobe_items)} wardrobe items for context: {context}")
        
        # Step 1: Run ViT inference on each wardrobe image
        import base64
        import tempfile
        
        temp_files = []
        predictions_list = []
        
        try:
            for idx, item in enumerate(wardrobe_items):
                # Decode base64 image data
                image_data = item['imageData']
                if ',' in image_data:
                    # Remove data:image/...;base64, prefix
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(image_bytes)
                    temp_path = tmp.name
                    temp_files.append(temp_path)
                
                # Run ViT inference
                print(f"  Analyzing wardrobe item {idx + 1}/{len(wardrobe_items)}...")
                preds = predict(model, temp_path, processor, label_encoders, device)
                preds['wardrobe_item_id'] = item.get('id')
                preds['index'] = idx
                predictions_list.append(preds)
                
                print(f"    Predictions: {preds['articleType']}, {preds['baseColour']}, {preds['usage']}")
        
        finally:
            # Clean up temp files
            for tmp_file in temp_files:
                try:
                    os.unlink(tmp_file)
                except:
                    pass
        
        # Step 2: Use vit_llm_infer.py logic to get outfit recommendation
        from vit_llm_infer import (
            build_vision_output,
            build_prompt,
            OutfitSelection
        )
        
        # Build vision output (categorize by tops/bottoms/onepieces)
        vision_output = build_vision_output(predictions_list, context)
        print(f"\nVision output: {len(vision_output['tops'])} tops, "
              f"{len(vision_output['bottoms'])} bottoms, "
              f"{len(vision_output['onepieces'])} onepieces")
        
        # Get LLM recommendation if API key available
        if 'GOOGLE_API_KEY' in os.environ or 'GEMINI_API_KEY' in os.environ:
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            client = genai.Client(api_key=api_key)
            
            # Try models in order of preference (different rate limit pools)
            # Can be overridden with GEMINI_MODEL env variable
            preferred_model = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-exp')
            
            # Build fallback list starting with preferred model
            models_to_try = [preferred_model]
            # Add other models as fallbacks if not already in list
            # Note: These are verified model names from the API
            for model in ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-flash-latest"]:
                if model not in models_to_try:
                    models_to_try.append(model)
            
            recommendation_text = None
            selected_indices = None
            
            for model_name in models_to_try:
                try:
                    # Build Gemini-style prompt
                    gemini_prompt = build_gemini_style_prompt(predictions_list, context)
                    
                    print(f"  Trying {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=gemini_prompt
                    )
                    
                    # Parse response in Gemini format
                    recommendation_text, selected_indices = parse_gemini_response(
                        response.text, len(predictions_list)
                    )
                    
                    print(f"  ✓ Success with {model_name}")
                    break  # Success - exit the loop
                    
                except Exception as api_error:
                    error_str = str(api_error)
                    if 'RESOURCE_EXHAUSTED' in error_str or '429' in error_str or 'quota' in error_str.lower():
                        print(f"  ✗ {model_name} rate limited, trying next model...")
                        continue  # Try next model
                    else:
                        # Non-rate-limit error - re-raise
                        raise
            
            # If we got a response from any model, return it
            if recommendation_text is not None:
                return jsonify({
                    "success": True,
                    "recommendation": recommendation_text,
                    "selectedItems": selected_indices,
                    "visionOutput": vision_output,
                    "predictions": predictions_list
                })
            else:
                # All models hit rate limits
                print("  All models rate limited, falling back to basic recommendation")
        
        # No API key or rate limit hit - provide basic recommendation
        recommendation_text = format_wardrobe_basic_recommendation(predictions_list, context)
        selected_indices = list(range(min(2, len(predictions_list))))
        
        return jsonify({
            "success": True,
            "recommendation": recommendation_text,
            "selectedItems": selected_indices,
            "visionOutput": vision_output,
            "predictions": predictions_list
        })
    
    except Exception as e:
        print(f"Error in analyze-wardrobe: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to analyze wardrobe",
            "message": str(e)
        }), 500

@app.route('/api/ml/get-recommendation', methods=['POST'])
def get_recommendation():
    """
    Get outfit recommendation using ML predictions + Gemini LLM
    Expected JSON: { "attributes": [...], "context": "..." }
    """
    try:
        data = request.json
        if not data or 'attributes' not in data or 'context' not in data:
            return jsonify({"error": "Missing attributes or context"}), 400
        
        attributes = data['attributes']
        context = data['context']
        
        print(f"Getting recommendation for context: {context}")
        print(f"Number of items: {len(attributes)}")
        
        # Extract ML predictions from attributes
        items = []
        for attr in attributes:
            if 'mlPredictions' in attr:
                items.append(attr['mlPredictions'])
            else:
                # Fallback: construct from frontend attributes
                items.append({
                    'gender': attr.get('name', '').split()[0] if attr.get('name') else 'Unisex',
                    'articleType': attr.get('category', 'Unknown'),
                    'baseColour': attr.get('color', 'Unknown'),
                    'season': 'All',
                    'usage': 'Casual'
                })
        
        # Build vision output for LLM
        vision_output = build_vision_output(items, context)
        print(f"Vision output: tops={len(vision_output['tops'])}, "
              f"bottoms={len(vision_output['bottoms'])}, "
              f"onepieces={len(vision_output['onepieces'])}")
        
        # Get LLM recommendation if API key is available
        if 'GOOGLE_API_KEY' in os.environ or 'GEMINI_API_KEY' in os.environ:
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            client = genai.Client(api_key=api_key)
            
            # Try models in order of preference (different rate limit pools)
            # Can be overridden with GEMINI_MODEL env variable
            preferred_model = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-exp')
            
            # Build fallback list starting with preferred model
            models_to_try = [preferred_model]
            # Add other models as fallbacks if not already in list
            # Note: These are verified model names from the API
            for model in ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-flash-latest"]:
                if model not in models_to_try:
                    models_to_try.append(model)
            
            recommendation_text = None
            selected_items = None
            
            for model_name in models_to_try:
                try:
                    # Build Gemini-style prompt from attributes
                    gemini_prompt = build_gemini_style_prompt_from_attributes(attributes, context)
                    
                    print(f"  Trying {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=gemini_prompt
                    )
                    
                    # Parse response in Gemini format
                    recommendation_text, selected_items = parse_gemini_response(
                        response.text, len(attributes)
                    )
                    
                    print(f"  ✓ Success with {model_name}")
                    break  # Success - exit the loop
                    
                except Exception as api_error:
                    error_str = str(api_error)
                    if 'RESOURCE_EXHAUSTED' in error_str or '429' in error_str or 'quota' in error_str.lower():
                        print(f"  ✗ {model_name} rate limited, trying next model...")
                        continue  # Try next model
                    else:
                        # Non-rate-limit error - re-raise
                        raise
            
            # If we got a response from any model, return it
            if recommendation_text is not None:
                return jsonify({
                    "success": True,
                    "recommendation": recommendation_text,
                    "selectedItems": selected_items
                })
            else:
                # All models hit rate limits
                print("  All models rate limited, falling back to basic recommendation")
        
        # No API key or rate limit hit - provide basic recommendation
        recommendation_text = f"Based on ML analysis for {context}:\n\n"
        recommendation_text += format_basic_recommendation(items, context)
        
        return jsonify({
            "success": True,
            "recommendation": recommendation_text,
            "selectedItems": list(range(min(2, len(attributes))))
        })
        
    except Exception as e:
        print(f"Error in get-recommendation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to get recommendation",
            "message": str(e)
        }), 500

def format_llm_response(analysis: OutfitSelection, attributes: List[Dict]) -> str:
    """Format the LLM analysis into a readable recommendation"""
    text = f"**Style:** {analysis.style.replace('_', ' ').title()}\n\n"
    
    if analysis.outfit_type == "onepiece":
        text += f"**Selected Outfit:**\n- {analysis.selected_onepiece}\n\n"
    else:
        text += f"**Selected Outfit:**\n"
        text += f"- Top: {analysis.selected_top}\n"
        text += f"- Bottom: {analysis.selected_bottom}\n\n"
    
    text += f"**Fit Assessment:** {analysis.fit.replace('_', ' ').title()}\n\n"
    text += f"**Reasoning:**\n{analysis.reason}"
    
    return text

def get_selected_indices(analysis: OutfitSelection, attributes: List[Dict]) -> List[int]:
    """Determine which item indices were selected by the LLM"""
    selected = []
    
    if analysis.outfit_type == "onepiece":
        # Find the onepiece item
        for idx, attr in enumerate(attributes):
            if analysis.selected_onepiece and analysis.selected_onepiece.lower() in attr.get('name', '').lower():
                selected.append(idx)
                break
    else:
        # Find top and bottom
        for idx, attr in enumerate(attributes):
            name_lower = attr.get('name', '').lower()
            if analysis.selected_top and analysis.selected_top.lower() in name_lower:
                selected.append(idx)
            elif analysis.selected_bottom and analysis.selected_bottom.lower() in name_lower:
                selected.append(idx)
    
    # If we couldn't match, just return first 2 items
    if not selected:
        selected = list(range(min(2, len(attributes))))
    
    return selected

def format_basic_recommendation(items: List[Dict], context: str) -> str:
    """Provide a basic recommendation without LLM"""
    text = "Items detected:\n"
    for idx, item in enumerate(items, 1):
        text += f"{idx}. {item.get('gender', '')} {item.get('baseColour', '')} "
        text += f"{item.get('articleType', '')} - {item.get('usage', '')}\n"
    
    text += f"\nSuggestion: Consider the formality required for '{context}' "
    text += "when selecting your outfit. Match colors and styles appropriately."
    
    return text

def format_wardrobe_llm_response(analysis: 'OutfitSelection', predictions_list: List[Dict]) -> str:
    """Format LLM analysis for wardrobe recommendations"""
    text = f"**Style:** {analysis.style.replace('_', ' ').title()}\n\n"
    
    if analysis.outfit_type == "onepiece":
        text += f"**Selected Outfit:**\n- {analysis.selected_onepiece}\n\n"
    else:
        text += f"**Selected Outfit:**\n"
        text += f"- Top: {analysis.selected_top}\n"
        text += f"- Bottom: {analysis.selected_bottom}\n\n"
    
    text += f"**Fit Assessment:** {analysis.fit.replace('_', ' ').title()}\n\n"
    text += f"**Reasoning:**\n{analysis.reason}"
    
    return text

def get_wardrobe_selected_indices(analysis: 'OutfitSelection', predictions_list: List[Dict]) -> List[int]:
    """Determine which wardrobe item indices were selected by the LLM"""
    from vit_llm_infer import detect_garment_type
    
    selected = []
    
    if analysis.outfit_type == "onepiece":
        # Find the onepiece item by matching article type
        for idx, pred in enumerate(predictions_list):
            garment_type = detect_garment_type(pred.get('articleType', ''))
            if garment_type == 'onepiece':
                selected.append(idx)
                break
    else:
        # Find top and bottom by matching article types and attributes
        top_found = False
        bottom_found = False
        
        for idx, pred in enumerate(predictions_list):
            garment_type = detect_garment_type(pred.get('articleType', ''))
            
            # Try to match based on LLM description
            article_lower = pred.get('articleType', '').lower()
            color_lower = pred.get('baseColour', '').lower()
            gender_lower = pred.get('gender', '').lower()
            
            # Check if this item matches the selected top
            if not top_found and garment_type == 'top':
                if (analysis.selected_top and 
                    (article_lower in analysis.selected_top.lower() or 
                     color_lower in analysis.selected_top.lower() or
                     gender_lower in analysis.selected_top.lower())):
                    selected.append(idx)
                    top_found = True
                    continue
            
            # Check if this item matches the selected bottom
            if not bottom_found and garment_type == 'bottom':
                if (analysis.selected_bottom and 
                    (article_lower in analysis.selected_bottom.lower() or 
                     color_lower in analysis.selected_bottom.lower() or
                     gender_lower in analysis.selected_bottom.lower())):
                    selected.append(idx)
                    bottom_found = True
                    continue
        
        # Fallback: if we couldn't match, select first top and first bottom
        if not selected:
            for idx, pred in enumerate(predictions_list):
                garment_type = detect_garment_type(pred.get('articleType', ''))
                if garment_type == 'top' and not top_found:
                    selected.append(idx)
                    top_found = True
                elif garment_type == 'bottom' and not bottom_found:
                    selected.append(idx)
                    bottom_found = True
                if top_found and bottom_found:
                    break
    
    # If still nothing selected, select first 2 items
    if not selected:
        selected = list(range(min(2, len(predictions_list))))
    
    return selected

def format_wardrobe_basic_recommendation(predictions_list: List[Dict], context: str) -> str:
    """Provide a basic wardrobe recommendation without LLM"""
    from vit_llm_infer import detect_garment_type
    
    text = f"**Wardrobe Analysis for {context}:**\n\n"
    text += "Items in your selection:\n"
    
    for idx, pred in enumerate(predictions_list, 1):
        garment_type = detect_garment_type(pred.get('articleType', ''))
        text += f"{idx}. {pred.get('gender', '')} {pred.get('baseColour', '')} "
        text += f"{pred.get('articleType', '')} ({garment_type or 'other'}) - {pred.get('usage', '')}\n"
    
    text += f"\n**Suggestion:**\nConsider pairing complementary items based on the formality "
    text += f"required for '{context}'. Match colors and styles for a cohesive look."
    
    return text

def build_gemini_style_prompt(predictions_list: List[Dict], context: str) -> str:
    """Build Gemini-compatible prompt from ML predictions"""
    # Format items like Gemini does: "[idx] Name - Color: X, Texture: Y, Category: Z"
    attributes_text = ""
    for idx, pred in enumerate(predictions_list):
        name = f"{pred.get('gender', '')} {pred.get('baseColour', '')} {pred.get('articleType', '')}".strip().title()
        color = pred.get('baseColour', 'Unknown').title()
        texture = "Unknown"  # ViT doesn't predict texture
        category = pred.get('usage', 'Casual').title()
        
        attributes_text += f"[{idx}] {name} - Color: {color}, Texture: {texture}, Category: {category}\n"
    
    prompt = f"""You are a professional fashion stylist. I have the following clothing items:
{attributes_text}

The occasion/context is: {context}

Please analyze and provide a response in TWO parts:

PART 1 - Selected Items (JSON format):
Return a JSON array of item indices (the numbers in brackets like [0], [1], etc.) that are appropriate for this occasion.
Example: [0, 2, 3]

PART 2 - Style Recommendation (text):
Provide a detailed style assessment organized into these EXACT subsections.

CRITICAL NAMING RULE: NEVER use "Item 0", "Item 1", "Item 2" etc. NEVER write formats like "Item 0 (description):" or "Item 1 (name):". Instead, ALWAYS refer to items ONLY by their descriptive names. ALWAYS use Title Case for item names (capitalize each word), like "Navy Straight-leg Trousers", "Double-breasted Plaid Blazer", or "Cropped Tweed Jacket".

**Overall Assessment**
A brief style assessment of the SELECTED items (2-3 sentences about the overall look and aesthetic)

**Why These Pieces Work**
For each selected item, explain why it works for this occasion. Format as bullet points with the item name in BOLD (in Title Case) followed by a colon, like:
- **Navy Straight-leg Trousers:** These are a foundational piece...
- **Double-breasted Plaid Blazer:** This adds sophistication...

**Outfit Combinations**
Provide 2-3 specific outfit combination suggestions. Give each outfit a CREATIVE NAME in bold, then describe what items to combine. Format like:
1. **The Polished Professional:** Pair the Navy Straight-leg Trousers with...
2. **Chic Business Casual:** Combine the Double-breasted Plaid Blazer with...
3. **Effortless Elegance:** Layer the Cropped Tweed Jacket over...

**Additional Styling Tips**
Organize tips by CATEGORY with bold headers. Format like:
- **Tops:** Suggestions for tops to pair with these items...
- **Footwear:** Shoe recommendations...
- **Accessories:** Belt, jewelry, bag suggestions...
- **Layering:** Tips for layering pieces...

Format your response exactly as:
SELECTED_ITEMS: [array of indices]
RECOMMENDATION: your text here

Keep the recommendation natural, friendly, and conversational. Make sure to include all four subsection headers in bold (wrapped in **)."""
    
    return prompt

def parse_gemini_response(text: str, total_items: int) -> tuple:
    """Parse Gemini-style response to extract selected items and recommendation"""
    import re
    
    selected_items = []
    recommendation = text
    
    # Try to extract selected items
    selected_match = re.search(r'SELECTED_ITEMS:\s*(\[[^\]]+\])', text)
    if selected_match:
        try:
            import json
            selected_items = json.loads(selected_match.group(1))
            # Validate indices
            selected_items = [idx for idx in selected_items if 0 <= idx < total_items]
        except:
            print('Could not parse selected items, selecting all')
    
    # Extract recommendation text
    rec_match = re.search(r'RECOMMENDATION:\s*([\s\S]*)', text)
    if rec_match:
        recommendation = rec_match.group(1).strip()
    
    # If no items selected, select all
    if not selected_items:
        selected_items = list(range(total_items))
    
    return recommendation, selected_items

def build_gemini_style_prompt_from_attributes(attributes: List[Dict], context: str) -> str:
    """Build Gemini-compatible prompt from frontend attributes"""
    # Format items like Gemini does
    attributes_text = ""
    for idx, attr in enumerate(attributes):
        name = attr.get('name', 'Unknown Item').title()
        color = attr.get('color', 'Unknown').title()
        texture = attr.get('texture', 'Unknown').title()
        category = attr.get('category', 'Unknown').title()
        
        attributes_text += f"[{idx}] {name} - Color: {color}, Texture: {texture}, Category: {category}\n"
    
    # Use same prompt as build_gemini_style_prompt
    prompt = f"""You are a professional fashion stylist. I have the following clothing items:
{attributes_text}

The occasion/context is: {context}

Please analyze and provide a response in TWO parts:

PART 1 - Selected Items (JSON format):
Return a JSON array of item indices (the numbers in brackets like [0], [1], etc.) that are appropriate for this occasion.
Example: [0, 2, 3]

PART 2 - Style Recommendation (text):
Provide a detailed style assessment organized into these EXACT subsections.

CRITICAL NAMING RULE: NEVER use "Item 0", "Item 1", "Item 2" etc. NEVER write formats like "Item 0 (description):" or "Item 1 (name):". Instead, ALWAYS refer to items ONLY by their descriptive names. ALWAYS use Title Case for item names (capitalize each word), like "Navy Straight-leg Trousers", "Double-breasted Plaid Blazer", or "Cropped Tweed Jacket".

**Overall Assessment**
A brief style assessment of the SELECTED items (2-3 sentences about the overall look and aesthetic)

**Why These Pieces Work**
For each selected item, explain why it works for this occasion. Format as bullet points with the item name in BOLD (in Title Case) followed by a colon, like:
- **Navy Straight-leg Trousers:** These are a foundational piece...
- **Double-breasted Plaid Blazer:** This adds sophistication...

**Outfit Combinations**
Provide 2-3 specific outfit combination suggestions. Give each outfit a CREATIVE NAME in bold, then describe what items to combine. Format like:
1. **The Polished Professional:** Pair the Navy Straight-leg Trousers with...
2. **Chic Business Casual:** Combine the Double-breasted Plaid Blazer with...
3. **Effortless Elegance:** Layer the Cropped Tweed Jacket over...

**Additional Styling Tips**
Organize tips by CATEGORY with bold headers. Format like:
- **Tops:** Suggestions for tops to pair with these items...
- **Footwear:** Shoe recommendations...
- **Accessories:** Belt, jewelry, bag suggestions...
- **Layering:** Tips for layering pieces...

Format your response exactly as:
SELECTED_ITEMS: [array of indices]
RECOMMENDATION: your text here

Keep the recommendation natural, friendly, and conversational. Make sure to include all four subsection headers in bold (wrapped in **)."""
    
    return prompt

@app.route('/api/ml/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "ML Backend is running",
        "device": str(device),
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    print("Initializing ML Backend Server...")
    init_model()
    
    port = int(os.environ.get('ML_PORT', 5001))  # Default to 5001 to avoid macOS AirPlay conflict
    print(f"\nML Backend Server starting on port {port}...")
    print(f"Health check: http://localhost:{port}/api/ml/health\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
