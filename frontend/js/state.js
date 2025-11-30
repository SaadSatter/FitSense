// Application State Management

export const MAX_ITEMS = 10;

export const state = {
    uploadedFiles: [],
    uploadedImageURLs: [], // Store data URLs for display
    selectedContext: null,
    customContext: '',
    currentResult: null,
    selectedWardrobeItems: [], // IDs of selected wardrobe items
    wardrobeContext: null,
    wardrobeCustomContext: '',
    isFromHistory: false, // Track if current result is from history
    analysisSource: 'upload', // Track where analysis came from: 'upload', 'wardrobe', or 'history'
    usedSelectAll: false // Track if Select All was used (for hiding styling options when >10)
};
