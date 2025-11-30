// API Calls and Recommendation Logic

import { state } from './state.js';
import { toast } from './toast.js';
import { toTitleCase, getErrorHint } from './utils.js';

// DOM Elements
let loadingState;

// Callbacks
let onDisplayResults = null;

export function setDisplayResultsCallback(callback) {
    onDisplayResults = callback;
}

export function initAnalysis() {
    loadingState = document.getElementById('loadingState');
}

export async function getRecommendation() {
    // Prepare form data
    const formData = new FormData();
    state.uploadedFiles.forEach((file, index) => {
        // Add the index into the filename so the backend can restore ordering
        const indexedFile = new File([file], `${index}__${file.name}`, { type: file.type });
        formData.append('images', indexedFile);
    });
    
    const context = toTitleCase(state.customContext || state.selectedContext);
    formData.append('context', context);

    // Show loading state (hide upload section if it exists - for regular upload flow)
    const uploadSection = document.querySelector('.upload-section');
    if (uploadSection) {
        uploadSection.style.display = 'none';
    }
    loadingState.classList.remove('hidden');

    try {
        // Step 1: Vision Model Processing
        updateLoadingMessage('Analyzing clothing attributes with vision model...');
        const visionResponse = await fetch('http://localhost:3000/api/analyze-images', {
            method: 'POST',
            body: formData
        });

        if (!visionResponse.ok) {
            throw new Error('Vision model analysis failed');
        }

        const visionData = await visionResponse.json();

        // Check for non-clothing items
        const nonClothingItems = visionData.attributes.filter(attr => attr.isClothing === false);
        const clothingItems = visionData.attributes.filter(attr => attr.isClothing !== false);
        
        // If ALL items are non-clothing, show error and abort
        if (clothingItems.length === 0) {
            throw new Error('No clothing items detected. Please upload images of clothing, footwear, or accessories.');
        }

        // Step 2: LLM Recommendation
        updateLoadingMessage('Getting personalized recommendations from AI...');
        const llmResponse = await fetch('http://localhost:3000/api/get-recommendation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                attributes: visionData.attributes,
                context: context
            })
        });

        if (!llmResponse.ok) {
            const errorData = await llmResponse.json().catch(() => ({ error: 'Unknown error' }));
            console.error('LLM recommendation failed:', errorData);
            throw new Error(`LLM recommendation failed: ${errorData.message || errorData.error || llmResponse.statusText}`);
        }

        const llmData = await llmResponse.json();

        // Store result (with images for current display)
        state.currentResult = {
            context: context,
            attributes: visionData.attributes,
            recommendation: llmData.recommendation,
            selectedItems: llmData.selectedItems || [],
            images: [...state.uploadedImageURLs], // Store image URLs for current session
            timestamp: new Date().toISOString()
        };
        
        // Clear the isFromHistory flag since this is a new analysis
        state.isFromHistory = false;
        // analysisSource is already set by the caller (getRecommendation or getWardrobeRecommendation)

        // Show results
        if (onDisplayResults) {
            onDisplayResults();
        }

    } catch (error) {
        console.error('Error:', error);
        
        // Show error toast
        const hint = getErrorHint(error);
        toast.error(
            'Analysis failed',
            hint || 'Check that the server is running and your API key is valid.',
            6000
        );
        
        // Reset view
        loadingState.classList.add('hidden');
        const uploadSection = document.querySelector('.upload-section');
        if (uploadSection) {
            uploadSection.style.display = 'block';
        }
    }
}

export function updateLoadingMessage(message) {
    const substep = document.querySelector('.loading-substep');
    if (substep) {
        substep.textContent = message;
    }
}

export function hideLoading() {
    if (loadingState) {
        loadingState.classList.add('hidden');
    }
}

export function showLoading() {
    if (loadingState) {
        loadingState.classList.remove('hidden');
    }
}
