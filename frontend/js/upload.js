// File Upload Handling

import { state, MAX_ITEMS } from './state.js';
import { toast } from './toast.js';
import { getErrorHint } from './utils.js';
import { addToWardrobe, getDb } from './db.js';

// DOM Elements
let uploadArea;
let fileInput;
let imagePreviews;
let contextButtons;
let customContextInput;
let getRecommendationBtn;

// Callback for getting recommendation
let onGetRecommendation = null;

export function setGetRecommendationCallback(callback) {
    onGetRecommendation = callback;
}

export function initUpload() {
    uploadArea = document.getElementById('uploadArea');
    fileInput = document.getElementById('fileInput');
    imagePreviews = document.getElementById('imagePreviews');
    contextButtons = document.querySelectorAll('.context-btn');
    customContextInput = document.getElementById('customContext');
    getRecommendationBtn = document.getElementById('getRecommendation');

    // File Upload Handling
    uploadArea.addEventListener('click', () => {
        if (state.uploadedFiles.length >= MAX_ITEMS) {
            return; // Upload area already shows limit message
        }
        fileInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Context Selection
    contextButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            contextButtons.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            state.selectedContext = btn.dataset.context;
            customContextInput.value = '';
            state.customContext = '';
            updateSubmitButton();
        });
    });

    customContextInput.addEventListener('input', (e) => {
        state.customContext = e.target.value;
        if (state.customContext) {
            contextButtons.forEach(b => b.classList.remove('selected'));
            state.selectedContext = null;
        }
        updateSubmitButton();
    });

    // Get Recommendation button
    getRecommendationBtn.addEventListener('click', async () => {
        state.analysisSource = 'upload';
        if (onGetRecommendation) {
            await onGetRecommendation();
        }
    });

    // Clear All Uploads button
    const clearAllUploadsBtn = document.getElementById('clearAllUploads');
    if (clearAllUploadsBtn) {
        clearAllUploadsBtn.addEventListener('click', () => {
            state.uploadedFiles = [];
            state.uploadedImageURLs = [];
            renderImagePreviews();
            updateSubmitButton();
            updateItemCounter();
        });
    }

    // Remove image from previews
    imagePreviews.addEventListener('click', (e) => {
        if (e.target.classList.contains('remove-image')) {
            const index = parseInt(e.target.dataset.index);
            state.uploadedFiles.splice(index, 1);
            state.uploadedImageURLs.splice(index, 1);
            renderImagePreviews();
            updateSubmitButton();
            updateItemCounter();
        }
    });

    // Save to Wardrobe button
    const saveToWardrobeBtn = document.getElementById('saveToWardrobe');
    if (saveToWardrobeBtn) {
        saveToWardrobeBtn.addEventListener('click', async () => {
            const db = getDb();
            if (!db) {
                toast.error('Not ready', 'Wardrobe is still loading. Please wait a moment.');
                return;
            }

            if (state.uploadedImageURLs.length === 0) return;
            
            const saveBtn = document.getElementById('saveToWardrobe');
            const originalHTML = saveBtn.innerHTML;
            saveBtn.disabled = true;
            saveBtn.innerHTML = '<span>Saving...</span>';
            
            try {
                let savedCount = 0;
                for (let i = 0; i < state.uploadedImageURLs.length; i++) {
                    const imageURL = state.uploadedImageURLs[i];
                    const fileName = state.uploadedFiles[i]?.name || `item-${Date.now()}-${i}.jpg`;
                    await addToWardrobe(imageURL, fileName);
                    savedCount++;
                }
                
                // Success feedback
                saveBtn.innerHTML = `<span>✓ Saved ${savedCount} item${savedCount !== 1 ? 's' : ''} to wardrobe!</span>`;
                saveBtn.style.backgroundColor = '#10b981';
                saveBtn.style.borderColor = '#10b981';
                saveBtn.style.color = 'white';
                
                setTimeout(() => {
                    saveBtn.innerHTML = originalHTML;
                    saveBtn.style.backgroundColor = '';
                    saveBtn.style.borderColor = '';
                    saveBtn.style.color = '';
                    saveBtn.disabled = false;
                }, 2000);
                
            } catch (error) {
                console.error('Error saving to wardrobe:', error);
                const hint = getErrorHint(error);
                toast.error('Save failed', `Couldn't save to wardrobe.${hint ? ' ' + hint : ''}`);
                saveBtn.innerHTML = originalHTML;
                saveBtn.disabled = false;
            }
        });
    }

    // Initial state
    updateSubmitButton();
    updateItemCounter();
}

function handleFiles(files) {
    const fileArray = Array.from(files);
    const imageFiles = fileArray.filter(file => file.type.startsWith('image/'));

    // Check current count and calculate how many we can add
    const currentCount = state.uploadedFiles.length;
    const remainingSlots = MAX_ITEMS - currentCount;
    
    if (remainingSlots <= 0) {
        return; // Upload area already shows limit message
    }

    // Only take as many files as we have slots for
    const filesToAdd = imageFiles.slice(0, remainingSlots);

    // Remember where these new files will start in the arrays
    const startIndex = state.uploadedFiles.length;

    // Store files in order
    state.uploadedFiles = [...state.uploadedFiles, ...filesToAdd];

    // Read each file and store its data URL at the correct index
    filesToAdd.forEach((file, i) => {
        const reader = new FileReader();
        const targetIndex = startIndex + i;  // exact position in the global arrays

        reader.onload = (e) => {
            state.uploadedImageURLs[targetIndex] = e.target.result;
        };

        reader.readAsDataURL(file);
    });

    renderImagePreviews();
    updateSubmitButton();
    updateItemCounter();
}

function renderImagePreviews() {
    imagePreviews.innerHTML = '';
    
    state.uploadedFiles.forEach((file, index) => {
        const div = document.createElement('div');
        div.className = 'image-preview';
        
        // Use the already-stored data URL if available, otherwise read the file
        if (state.uploadedImageURLs[index]) {
            div.innerHTML = `
                <img src="${state.uploadedImageURLs[index]}" alt="Preview">
                <button class="remove-image" data-index="${index}">&times;</button>
            `;
            imagePreviews.appendChild(div);
        } else {
            // Fallback: read the file if URL not yet available
            const reader = new FileReader();
            reader.onload = (e) => {
                div.innerHTML = `
                    <img src="${e.target.result}" alt="Preview">
                    <button class="remove-image" data-index="${index}">&times;</button>
                `;
                imagePreviews.appendChild(div);
            };
            reader.readAsDataURL(file);
        }
    });
}

export function updateSubmitButton() {
    if (!getRecommendationBtn) return;
    const hasImages = state.uploadedFiles.length > 0;
    const hasContext = state.selectedContext || state.customContext;
    getRecommendationBtn.disabled = !(hasImages && hasContext);
}

export function updateItemCounter() {
    const itemCounter = document.getElementById('itemCounter');
    const itemCount = document.getElementById('itemCount');
    const itemCountText = document.getElementById('itemCountText');
    const count = state.uploadedFiles.length;
    
    if (itemCounter && itemCount && itemCountText) {
        itemCount.textContent = count;
        
        // Handle singular/plural
        itemCountText.textContent = count === 1 ? 'item selected' : 'items selected';
        
        // Update styling based on count
        if (count >= MAX_ITEMS) {
            itemCounter.classList.add('limit-reached');
            uploadArea.classList.add('limit-reached');
            showUploadLimitMessage();
        } else {
            itemCounter.classList.remove('limit-reached');
            uploadArea.classList.remove('limit-reached');
            restoreUploadPlaceholder();
        }
    }
    
    // Show/hide save to wardrobe button
    const saveToWardrobeSection = document.getElementById('saveToWardrobeSection');
    if (saveToWardrobeSection) {
        if (count > 0) {
            saveToWardrobeSection.classList.remove('hidden');
        } else {
            saveToWardrobeSection.classList.add('hidden');
        }
    }
    
    // Show/hide clear all button
    const clearAllBtn = document.getElementById('clearAllUploads');
    if (clearAllBtn) {
        if (count > 0) {
            clearAllBtn.classList.remove('hidden');
        } else {
            clearAllBtn.classList.add('hidden');
        }
    }
}

// Update upload area to show limit message or restore normal state
function showUploadLimitMessage() {
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    if (placeholder) {
        placeholder.innerHTML = `
            <h3>Maximum of ${MAX_ITEMS} items reached</h3>
            <p>To upload more, remove one of the selected items.</p>
        `;
    }
}

function restoreUploadPlaceholder() {
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    if (placeholder) {
        placeholder.innerHTML = `
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <h3>Drag & drop images here</h3>
            <p>or click to browse files</p>
            <span class="upload-hint">Supports: JPG, PNG, WEBP</span>
        `;
    }
}

export function resetUploadForm() {
    state.uploadedFiles = [];
    state.uploadedImageURLs = [];
    state.selectedContext = null;
    state.customContext = '';
    
    if (imagePreviews) imagePreviews.innerHTML = '';
    if (customContextInput) customContextInput.value = '';
    if (contextButtons) {
        contextButtons.forEach(b => b.classList.remove('selected'));
    }
    
    updateSubmitButton();
    updateItemCounter();
}
