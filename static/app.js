// State management
let currentFolder = '';
let targetFolder = ''; // For transfer functionality
let images = [];

// Stitch mode state
let stitchMode = false;
let stitchSelectedIndices = [];       // current round selection (indices)
let stitchUsedFilenames = new Set();  // all filenames consumed across rounds this session
let currentIndex = 0;
let overlayActive = false;
let opacityValue = 50; // Default 50%
// Per-file cache busting: only files actually modified this session get &t=
// dirtyFiles: Map<filename, timestamp> — individual edits/crops
// folderBuster: string — bumped for bulk ops (reshuffle/compress/fit) that change all files
const dirtyFiles = new Map();
let folderBuster = '';

function fileBuster(filename) {
    if (dirtyFiles.has(filename)) return `&t=${dirtyFiles.get(filename)}`;
    if (folderBuster) return `&t=${folderBuster}`;
    return '';
}

function markDirty(filename) {
    dirtyFiles.set(filename, Date.now());
}

function bumpFolderBuster() {
    folderBuster = String(Date.now());
    dirtyFiles.clear();
}
let allFolders = []; // Store all folders for target selection
let activeControlView = null; // Which control is shown in full preview (null = original image)
let comparisonControlView = null; // Which control is shown in comparison view (null = hidden)
let linkedDataset = null; // Linked dataset for synchronized operations
let mainEditor = null; // Main editor instance
let comparisonEditor = null; // Comparison editor instance

// DOM elements
const folderSelect = document.getElementById('folder-select');
const imageGrid = document.getElementById('image-grid');
const imageCount = document.getElementById('image-count');
const modal = document.getElementById('preview-modal');
const previewImg = document.getElementById('preview-img');
const previewControl = document.getElementById('preview-control');
const comparisonContainer = document.getElementById('comparison-container');
const comparisonImg = document.getElementById('comparison-img');
const currentFilename = document.getElementById('current-filename');
const closeBtn = document.getElementById('close-modal');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const toggleBtn = document.getElementById('toggle-overlay');
const duplicateBtn = document.getElementById('duplicate-btn');
const pixelmatorBtn = document.getElementById('pixelmator-btn');
const deleteBtn = document.getElementById('delete-btn');
const reshuffleBtn = document.getElementById('reshuffle-btn');
const compressBtn = document.getElementById('compress-btn');
const fitBtn = document.getElementById('fit-btn');
const exportBtn = document.getElementById('export-btn');
const opacitySlider = document.getElementById('opacity-slider');
const opacityValueDisplay = document.getElementById('opacity-value');
const targetDatasetSelect = document.getElementById('target-dataset-select');
const transferBtn = document.getElementById('transfer-btn');
const captionText = document.getElementById('caption-text');
const saveCaptionBtn = document.getElementById('save-caption-btn');
const controlThumbs = {
    Control1: document.getElementById('control1-thumb'),
    Control2: document.getElementById('control2-thumb'),
    Control3: document.getElementById('control3-thumb')
};

// Link dataset elements
const linkBtn = document.getElementById('link-btn');
const linkSelect = document.getElementById('link-select');
const linkedIndicator = document.getElementById('linked-indicator');
const unlinkBtn = document.getElementById('unlink-btn');

// Initialize
// ── Session persistence ────────────────────────────────────────────────────
const SESSION_KEY = 'qdm_session';
let _saveTimer = null;

function saveAppState() {
    clearTimeout(_saveTimer);
    _saveTimer = setTimeout(() => {
        try {
            localStorage.setItem(SESSION_KEY, JSON.stringify({
                folder:               currentFolder,
                filename:             images[currentIndex] || null,
                modalOpen:            modal.classList.contains('active'),
                opacityValue,
                activeControlView,
                comparisonControlView,
            }));
        } catch (e) { /* localStorage unavailable */ }
    }, 200);
}

async function restoreAppState() {
    try {
        const raw = localStorage.getItem(SESSION_KEY);
        if (!raw) return;
        const s = JSON.parse(raw);
        if (!s.folder) return;

        // Only restore if folder still exists
        if (!allFolders.some(f => f.path === s.folder)) return;

        folderSelect.value = s.folder;
        await loadImages(s.folder);

        // Restore position by filename (robust against list reorder)
        if (s.filename) {
            const idx = images.indexOf(s.filename);
            if (idx !== -1) currentIndex = idx;
        }

        // Restore opacity
        if (s.opacityValue != null) {
            opacityValue = s.opacityValue;
            opacitySlider.value = opacityValue;
            if (opacityValueDisplay) opacityValueDisplay.textContent = `${opacityValue}%`;
        }

        // Restore open modal — bypass openPreview() to avoid resetting control views
        if (s.modalOpen && images.length > 0) {
            activeControlView     = s.activeControlView     || null;
            comparisonControlView = s.comparisonControlView || null;
            updateTargetDatasetSelect();
            updatePreview();
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    } catch (e) {
        console.warn('Failed to restore session state:', e);
    }
}
// ──────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    await loadFolders();
    setupEventListeners();
    await restoreAppState();
});




// Sync Save and Undo buttons based on both editors
function syncEditorButtons() {
    const undoBtn  = document.getElementById('undo-btn');
    const resetBtn = document.getElementById('reset-edit-btn');
    const saveBtn  = document.getElementById('save-edit-btn');

    const hasChanges = (mainEditor && mainEditor.history.length > 1) ||
        (comparisonEditor && comparisonEditor.history.length > 1);

    [undoBtn, resetBtn, saveBtn].forEach(btn => {
        if (btn) btn.classList.toggle('hidden', !hasChanges);
    });
}

// Global callback for editor to notify when image is saved
window.onImageSaved = function (reloadList = false) {
    const filename = images[currentIndex];
    markDirty(filename);
    updatePreview();

    if (reloadList) {
        // Reload the entire grid (for new images)
        loadImages(currentFolder);
    } else {
        // Just refresh the grid thumbnail for this image
        const gridImg = document.querySelector(`.image-item[data-index="${currentIndex}"] img`);
        if (gridImg) {
            const src = gridImg.src.split('?')[0];
            gridImg.src = `${src}?folder=${encodeURIComponent(currentFolder)}${fileBuster(filename)}`;
        }
    }
};

// Load available folders
async function loadFolders() {
    try {
        const response = await fetch('/api/folders');
        const data = await response.json();

        if (data.error) {
            console.error('Error loading folders:', data.error);
            return;
        }

        allFolders = data.folders; // Store for later use
        folderSelect.innerHTML = '<option value="">-- Select a folder --</option>';

        // Add Create New Dataset option
        const createOption = document.createElement('option');
        createOption.value = '__create_new__';
        createOption.textContent = '➕ Create New Dataset';
        createOption.style.fontWeight = 'bold';
        folderSelect.appendChild(createOption);

        data.folders.forEach(folder => {
            const option = document.createElement('option');
            option.value = folder.path;
            option.textContent = folder.name;
            folderSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load folders:', error);
    }
}

// Create new dataset
async function createNewDataset() {
    const name = prompt('Enter new dataset name (letters, numbers, underscores, hyphens only):');

    if (!name) {
        folderSelect.value = '';
        return;
    }

    try {
        const response = await fetch('/api/create-dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name.trim() })
        });

        const data = await response.json();

        if (data.success) {
            alert(`Dataset "${data.name}" created successfully!`);
            await loadFolders(); // Reload folder list
            folderSelect.value = data.path; // Select new dataset
            loadImages(data.path);
        } else {
            alert(`Failed to create dataset: ${data.error}`);
            folderSelect.value = '';
        }
    } catch (error) {
        console.error('Failed to create dataset:', error);
        alert('Failed to create dataset. Check console for details.');
        folderSelect.value = '';
    }
}

// Link Dataset Functions
function showLinkSelector() {
    // Populate link select with other folders
    linkSelect.innerHTML = '<option value="">-- Select linked dataset --</option>';
    allFolders.forEach(folder => {
        if (folder.path !== currentFolder) {
            const option = document.createElement('option');
            option.value = folder.path;
            option.textContent = folder.name;
            linkSelect.appendChild(option);
        }
    });

    linkBtn.classList.add('hidden');
    linkSelect.classList.remove('hidden');
}

async function linkDataset(folderPath) {
    if (!folderPath) {
        linkSelect.classList.add('hidden');
        linkBtn.classList.remove('hidden');
        return;
    }

    linkedDataset = folderPath;
    const folderName = allFolders.find(f => f.path === folderPath)?.name || folderPath;

    // Update UI
    linkSelect.classList.add('hidden');
    linkBtn.classList.add('hidden');
    linkedIndicator.textContent = folderName;
    linkedIndicator.classList.remove('hidden');
    unlinkBtn.classList.remove('hidden');

    // Check for orphan files
    await checkOrphanFiles();
}

function unlinkDataset() {
    linkedDataset = null;
    linkedIndicator.classList.add('hidden');
    unlinkBtn.classList.add('hidden');
    linkBtn.classList.remove('hidden');
}

async function checkOrphanFiles() {
    if (!currentFolder || !linkedDataset) return;

    try {
        const response = await fetch('/api/compare-datasets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                primaryFolder: currentFolder,
                linkedFolder: linkedDataset
            })
        });

        const data = await response.json();

        if (data.orphans && data.orphans.length > 0) {
            const deleteOrphans = confirm(
                `Found ${data.orphans.length} orphan file(s) in linked dataset ` +
                `that don't exist in primary dataset.\n\n` +
                `Examples: ${data.orphans.slice(0, 3).join(', ')}${data.orphans.length > 3 ? '...' : ''}\n\n` +
                `Delete these orphan files?`
            );

            if (deleteOrphans) {
                await deleteOrphanFiles(data.orphans);
            }
        }
    } catch (error) {
        console.error('Failed to check orphan files:', error);
    }
}

async function deleteOrphanFiles(orphans) {
    let deleted = 0;
    for (const filename of orphans) {
        try {
            const response = await fetch(
                `/api/delete/${encodeURIComponent(filename)}?folder=${encodeURIComponent(linkedDataset)}`,
                { method: 'DELETE' }
            );
            if (response.ok) deleted++;
        } catch (error) {
            console.error(`Failed to delete orphan ${filename}:`, error);
        }
    }
    alert(`Deleted ${deleted} orphan file(s) from linked dataset.`);
}

// Load images from selected folder
async function loadImages(folder) {
    if (!folder) {
        imageGrid.innerHTML = '<div class="empty-state"><p>📁 Select a dataset folder to view images</p></div>';
        imageCount.textContent = '';
        return;
    }

    // Capture current filename to restore position after reload
    let currentFilename = null;
    if (images.length > 0 && currentIndex >= 0 && currentIndex < images.length) {
        currentFilename = images[currentIndex];
    }

    try {
        const response = await fetch(`/api/images?folder=${encodeURIComponent(folder)}`);
        const data = await response.json();

        if (data.error) {
            console.error('Error loading images:', data.error);
            imageGrid.innerHTML = `<div class="empty-state"><p>❌ Error: ${data.error}</p></div>`;
            return;
        }

        images = data.images;
        currentFolder = folder;

        // Restore currentIndex to keep user on the same image even if list order changed
        if (currentFilename) {
            const newIndex = images.indexOf(currentFilename);
            if (newIndex !== -1) {
                currentIndex = newIndex;
                console.log(`Restored position: ${currentFilename} is at index ${currentIndex}`);
            } else {
                // If file is gone, stay at roughly same position or 0
                if (currentIndex >= images.length) {
                    currentIndex = Math.max(0, images.length - 1);
                }
            }
        }

        renderImageGrid();
        updateImageCount();
        saveAppState();
    } catch (error) {
        console.error('Failed to load images:', error);
        imageGrid.innerHTML = '<div class="empty-state"><p>❌ Failed to load images</p></div>';
    }
}

// Render image grid
function renderImageGrid() {
    if (images.length === 0) {
        imageGrid.innerHTML = '<div class="empty-state"><p>📷 No images found in this folder</p></div>';
        return;
    }

    const fragment = document.createDocumentFragment();
    images.forEach((filename, index) => {
        const item = document.createElement('div');
        item.className = 'image-item';
        item.dataset.index = index;

        const img = document.createElement('img');
        img.src = `/api/image/img/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}${fileBuster(filename)}`;
        img.alt = filename;
        img.loading = 'lazy';
        img.addEventListener('load', () => {
            const w = img.naturalWidth, h = img.naturalHeight;
            if (h > w * 1.3) {
                // tall: span as many rows as the ratio requires, min 2 max 5
                const span = Math.min(5, Math.max(2, Math.round(h / w)));
                item.style.gridRow = `span ${span}`;
            } else if (w > h * 1.3) {
                // wide: span as many columns as the ratio requires, min 2 max 4
                const span = Math.min(4, Math.max(2, Math.round(w / h)));
                item.style.gridColumn = `span ${span}`;
            }
        }, { once: true });

        const filenameSpan = document.createElement('span');
        filenameSpan.className = 'filename';
        filenameSpan.textContent = filename;

        if (stitchMode && stitchUsedFilenames.has(filename)) {
            item.classList.add('stitch-picked');
        }

        item.appendChild(img);
        item.appendChild(filenameSpan);
        fragment.appendChild(item);
    });

    imageGrid.innerHTML = '';
    imageGrid.appendChild(fragment);
}

// Update image count display
function updateImageCount() {
    imageCount.textContent = `${images.length} image${images.length !== 1 ? 's' : ''}`;
}

// Open preview modal
function openPreview(index) {
    currentIndex = index;
    overlayActive = false;
    targetFolder = ''; // Reset target folder
    activeControlView = null; // Reset to show original image
    comparisonControlView = null; // Reset comparison view
    updateTargetDatasetSelect(); // Update dropdown options
    updatePreview();
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    saveAppState();
}

// Update target dataset dropdown (exclude current folder)
function updateTargetDatasetSelect() {
    targetDatasetSelect.innerHTML = '<option value="">-- Select target dataset --</option>';
    allFolders.forEach(folder => {
        if (folder.path !== currentFolder) {
            const option = document.createElement('option');
            option.value = folder.path;
            option.textContent = folder.name;
            targetDatasetSelect.appendChild(option);
        }
    });
    targetDatasetSelect.value = '';
    transferBtn.style.display = 'none';
}

// Handle target dataset selection
function onTargetDatasetChange(value) {
    targetFolder = value;
    transferBtn.style.display = value ? 'flex' : 'none';
}

// Close preview modal
function closePreview() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
    overlayActive = false;
    previewImg.classList.remove('overlay-active');
    previewControl.classList.remove('active');
    toggleBtn.classList.remove('active');
    saveAppState();
}

// Update preview with current image
function updatePreview() {
    if (images.length === 0) return;

    const filename = images[currentIndex];
    const baseUrl = `/api/image`;
    const folderParam = `?folder=${encodeURIComponent(currentFolder)}${fileBuster(filename)}`;
    const imageContainer = document.querySelector('.image-container');

    if (!imageContainer) {
        console.error('imageContainer not found');
        return;
    }

    // Force-hide preview-img — it's only a data source for the canvas,
    // never shown directly. Inline style ensures it works even with cached CSS.
    previewImg.style.display = 'none';
    comparisonImg.style.display = 'none';

    // Update comparison view FIRST - before changing src
    if (comparisonControlView) {
        comparisonContainer.classList.add('active');
        imageContainer.classList.add('comparison-mode');
        console.log('Added comparison-mode class');
    } else {
        comparisonContainer.classList.remove('active');
        imageContainer.classList.remove('comparison-mode');
        console.log('Removed comparison-mode class');
    }

    // Update preview images
    if (activeControlView) {
        previewImg.src = `${baseUrl}/${activeControlView}/${encodeURIComponent(filename)}${folderParam}`;
    } else {
        previewImg.src = `${baseUrl}/img/${encodeURIComponent(filename)}${folderParam}`;
    }
    previewControl.src = `${baseUrl}/Control1/${encodeURIComponent(filename)}${folderParam}`;
    currentFilename.textContent = filename;

    // Load caption
    loadCaption(filename);

    // Update comparison image src
    if (comparisonControlView) {
        comparisonImg.src = `${baseUrl}/${comparisonControlView}/${encodeURIComponent(filename)}${folderParam}`;
    } else {
        comparisonImg.src = '';
    }

    // Force canvas update if image is already loaded (e.g. toggling comparison)
    if (mainEditor && previewImg.complete) {
        setTimeout(() => mainEditor.fitToScreen(), 50);
    }
    if (comparisonEditor && comparisonImg.complete) {
        setTimeout(() => comparisonEditor.fitToScreen(), 50);
    }

    // Initialize main canvas editor
    previewImg.onload = () => {
        const controlImg = new Image();
        controlImg.crossOrigin = 'anonymous';
        controlImg.src = `${baseUrl}/Control1/${encodeURIComponent(filename)}${folderParam}`;
        controlImg.onload = () => {
            const canvas = document.getElementById('edit-canvas');
            const subfolder = activeControlView || 'img';
            if (!mainEditor) {
                mainEditor = new ImageEditor(canvas, previewImg, controlImg, previewControl, subfolder);
                mainEditor.onStateChange = syncEditorButtons;
            } else {
                mainEditor.targetImageElement = previewImg;
                mainEditor.controlImageElement = controlImg;
                mainEditor.overlayElement = previewControl;
                mainEditor.subfolder = subfolder;
                mainEditor.onStateChange = syncEditorButtons;
                mainEditor.setupCanvas();
                mainEditor.history = [];
                mainEditor.saveState();
            }
            mainEditor.currentFilename = filename;
            mainEditor.currentFolder = currentFolder;
            mainEditor.updateSaveButton();
        };
    };

    // Initialize comparison canvas editor if comparison mode is active
    if (comparisonControlView) {
        comparisonImg.onload = () => {
            const comparisonCanvas = document.getElementById('comparison-canvas');
            // For comparison editor, we don't have an overlay or control image to erase from (usually)
            // But we can use the main img as the control source for stamp/eraser if we want? 
            // For now, use comparisonImg itself as its own control.

            if (!comparisonEditor) {
                comparisonEditor = new ImageEditor(comparisonCanvas, comparisonImg, comparisonImg, null, comparisonControlView);
                comparisonEditor.onStateChange = syncEditorButtons;
            } else {
                comparisonEditor.targetImageElement = comparisonImg;
                comparisonEditor.controlImageElement = comparisonImg;
                comparisonEditor.subfolder = comparisonControlView;
                comparisonEditor.onStateChange = syncEditorButtons;
                comparisonEditor.setupCanvas();
                comparisonEditor.history = [];
                comparisonEditor.saveState();
            }
            comparisonEditor.currentFilename = filename;
            comparisonEditor.currentFolder = currentFolder;
            comparisonEditor.updateSaveButton();
        };
    } else {
        // Clear comparison editor if not in comparison mode
        if (comparisonEditor) {
            comparisonEditor.reset(); // Or hide?
        }
    }

    // Load control thumbnails
    loadControlThumbnails(filename);

    // Update navigation buttons
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex === images.length - 1;

    // Reset overlay state
    if (!overlayActive) {
        previewImg.classList.remove('overlay-active');
        previewControl.classList.remove('active');
        toggleBtn.classList.remove('active');
    } else {
        previewImg.classList.add('overlay-active');
        previewControl.classList.add('active');
        toggleBtn.classList.add('active');
    }
}

// Load control thumbnails and check which exist
function loadControlThumbnails(filename) {
    const baseUrl = `/api/image`;
    const folderParam = `?folder=${encodeURIComponent(currentFolder)}${fileBuster(filename)}`;

    const controls = ['Control1', 'Control2', 'Control3'];

    controls.forEach(controlName => {
        const thumb = controlThumbs[controlName];
        const img = thumb.querySelector('img');
        const imgUrl = `${baseUrl}/${controlName}/${encodeURIComponent(filename)}${folderParam}`;

        // Reset state
        thumb.classList.remove('hidden', 'active', 'comparison-active');

        // Mark active if this control is shown in main preview
        if (activeControlView === controlName) {
            thumb.classList.add('active');
        }

        // Mark active if this control is shown in comparison
        if (comparisonControlView === controlName) {
            thumb.classList.add('comparison-active');
        }

        // Try to load image
        img.src = imgUrl;
        img.onerror = () => {
            thumb.classList.add('hidden');
        };
        img.onload = () => {
            thumb.classList.remove('hidden');
        };
    });
}

// Show control image in full preview or side-by-side comparison
function showControlFullPreview(controlName) {
    if (comparisonControlView === controlName) {
        // Toggle off - hide comparison
        comparisonControlView = null;
    } else {
        comparisonControlView = controlName;
    }
    updatePreview();
    saveAppState();
}

// Navigate to previous image
async function showPrevious() {
    if (currentIndex > 0) {
        if (!await checkUnsavedCaption()) return;
        currentIndex--;
        updatePreview();
        saveAppState();
    }
}

// Navigate to next image
async function showNext() {
    if (currentIndex < images.length - 1) {
        if (!await checkUnsavedCaption()) return;
        currentIndex++;
        updatePreview();
        saveAppState();
    }
}

// Toggle overlay
function toggleOverlay() {
    overlayActive = !overlayActive;
    const canvas = document.getElementById('edit-canvas');

    if (overlayActive) {
        canvas.style.opacity = opacityValue / 100;
        previewControl.classList.add('active');
        toggleBtn.classList.add('active');
    } else {
        canvas.style.opacity = 1;
        previewControl.classList.remove('active');
        toggleBtn.classList.remove('active');
    }
}

// Update opacity value
function updateOpacity(value) {
    opacityValue = parseInt(value);
    opacityValueDisplay.textContent = `${opacityValue}%`;
    saveAppState();

    // Update canvas opacity if overlay is active
    if (overlayActive) {
        const canvas = document.getElementById('edit-canvas');
        canvas.style.opacity = opacityValue / 100;
    }
}

// Transfer current image to target dataset
async function transferCurrentImage() {
    if (images.length === 0 || !targetFolder) return;

    const filename = images[currentIndex];

    try {
        transferBtn.disabled = true;
        transferBtn.textContent = 'Transferring...';

        // Build request body with optional linked folder
        const requestBody = { targetFolder: targetFolder };
        if (linkedDataset) {
            requestBody.linkedFolder = linkedDataset;
        }

        const response = await fetch(
            `/api/transfer/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            }
        );

        const data = await response.json();

        if (data.success) {
            // Remove from images array (file was moved)
            images.splice(currentIndex, 1);

            // Partial DOM update: remove the transferred element
            const gridItem = imageGrid.querySelector(`.image-item[data-index="${currentIndex}"]`);
            if (gridItem) {
                gridItem.remove();
                // Update indices for all subsequent items
                const items = imageGrid.querySelectorAll('.image-item');
                items.forEach((item, idx) => item.dataset.index = idx);
            }

            // Update UI
            if (images.length === 0) {
                closePreview();
                updateImageCount();
                imageGrid.innerHTML = '<div class="empty-state"><p>📷 No images found in this folder</p></div>';
            } else {
                if (currentIndex >= images.length) {
                    currentIndex = images.length - 1;
                }
                updatePreview();
                updateImageCount();
            }

            console.log('Transferred:', data.transferred);
        } else {
            alert(`Failed to transfer: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Failed to transfer image:', error);
        alert('Failed to transfer image. Check console for details.');
    } finally {
        transferBtn.disabled = false;
        transferBtn.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 19V5M5 12l7-7 7 7"></path>
            </svg>
            Transfer (↑)
        `;
    }
}

// Duplicate current image set
async function duplicateCurrentImage() {
    if (images.length === 0) return;

    const filename = images[currentIndex];

    try {
        duplicateBtn.disabled = true;
        duplicateBtn.textContent = '...';

        const response = await fetch('/api/augment/duplicate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                folder: currentFolder,
                filename: filename
            })
        });

        const data = await response.json();

        if (data.success) {
            // Add new image to list immediately after current
            const newIndex = currentIndex + 1;
            images.splice(newIndex, 0, data.newFilename);

            // Switch to it
            currentIndex = newIndex;

            // Render updated grid and preview
            renderImageGrid();
            updateImageCount();
            updatePreview();

            console.log('Duplicated to:', data.newFilename);
        } else {
            alert(`Failed to duplicate: ${data.error}`);
        }

    } catch (error) {
        console.error('Duplicate failed:', error);
        alert('Failed to duplicate. Check console.');
    } finally {
        duplicateBtn.disabled = false;
        duplicateBtn.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                Duplicate
            `;
    }
}

// Delete current image set
async function deleteCurrentImage() {
    if (images.length === 0) return;

    const filename = images[currentIndex];

    // Build URL with optional linked folder
    let deleteUrl = `/api/delete/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}`;
    if (linkedDataset) {
        deleteUrl += `&linkedFolder=${encodeURIComponent(linkedDataset)}`;
    }

    try {
        const response = await fetch(deleteUrl, { method: 'DELETE' });

        const data = await response.json();

        if (data.success) {
            // Remove from images array
            images.splice(currentIndex, 1);

            // Partial DOM update: remove the deleted element
            const gridItem = imageGrid.querySelector(`.image-item[data-index="${currentIndex}"]`);
            if (gridItem) {
                gridItem.remove();
                // Update indices for all subsequent items
                const items = imageGrid.querySelectorAll('.image-item');
                items.forEach((item, idx) => item.dataset.index = idx);
            }

            // Update UI
            if (images.length === 0) {
                closePreview();
                updateImageCount();
                imageGrid.innerHTML = '<div class="empty-state"><p>📷 No images found in this folder</p></div>';
            } else {
                // Adjust index if needed
                if (currentIndex >= images.length) {
                    currentIndex = images.length - 1;
                }
                updatePreview();
                updateImageCount();
            }

            console.log('Deleted:', data.deleted);
            if (data.errors && data.errors.length > 0) {
                console.warn('Warnings:', data.errors);
            }
        } else {
            alert(`Failed to delete: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Failed to delete image:', error);
        alert('Failed to delete image. Check console for details.');
    }
}

// Reshuffle dataset
async function reshuffleDataset() {
    if (!currentFolder) {
        alert('Please select a dataset folder first.');
        return;
    }

    const confirmed = confirm(
        'Are you sure you want to reshuffle all images?\n\n' +
        'This will randomize the filenames (image_00001, image_00002...) ' +
        'while keeping targets, controls, and captions synchronized.\n\n' +
        'This action cannot be undone.'
    );

    if (!confirmed) return;

    try {
        reshuffleBtn.disabled = true;
        reshuffleBtn.textContent = 'Shuffling...';

        const response = await fetch(`/api/reshuffle?folder=${encodeURIComponent(currentFolder)}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            bumpFolderBuster();
            alert(`Successfully reshuffled ${data.count} images!`);
            loadImages(currentFolder); // Reload grid
        } else {
            alert(`Failed to reshuffle: ${data.error}`);
        }
    } catch (error) {
        console.error('Reshuffle failed:', error);
        alert('Failed to reshuffle dataset. Check console for details.');
    } finally {
        reshuffleBtn.disabled = false;
        reshuffleBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5"></path>
            </svg>
            Reshuffle
        `;
    }
}

// Compress dataset
async function fitDataset() {
    if (!currentFolder) return;

    if (!confirm('Resize/Letterbox all control images (Control1-3) to match primary images (img folder)?\nThis will add black bars if aspect ratios differ.')) return;

    fitBtn.disabled = true;
    fitBtn.textContent = 'Processing...';

    try {
        const response = await fetch(`/api/dataset/fit?folder=${encodeURIComponent(currentFolder)}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            alert(`Success!\nProcessed: ${data.processed} image sets\nUpdated (resized): ${data.updated} control images`);
            bumpFolderBuster();
            updatePreview();
        } else {
            alert(`Failed: ${data.error}`);
        }
    } catch (error) {
        console.error('Fit dataset error:', error);
        alert('Failed to process dataset');
    } finally {
        fitBtn.disabled = false;
        fitBtn.textContent = 'Fit';
    }
}

async function compressDataset() {
    if (!currentFolder) {
        alert('Please select a dataset folder first.');
        return;
    }

    const confirmed = confirm(
        'Compress all PNG images in this dataset?\n\n' +
        'This will optimize PNG files for smaller size.\n' +
        'Original quality will be preserved as much as possible.'
    );

    if (!confirmed) return;

    try {
        compressBtn.disabled = true;
        compressBtn.textContent = 'Compressing...';

        const response = await fetch(`/api/compress?folder=${encodeURIComponent(currentFolder)}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            bumpFolderBuster();
            alert(
                `Compressed ${data.compressed} images!\n\n` +
                `Original: ${data.originalSizeMB} MB\n` +
                `New: ${data.newSizeMB} MB\n` +
                `Saved: ${data.savingsMB} MB (${data.savingsPercent}%)`
            );
            loadImages(currentFolder);
        } else {
            alert(`Failed to compress: ${data.error}`);
        }
    } catch (error) {
        console.error('Compress failed:', error);
        alert('Failed to compress dataset. Check console for details.');
    } finally {
        compressBtn.disabled = false;
        compressBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M4 14h6v6H4zM14 4h6v6h-6zM4 4h6v6H4zM14 14l6 6M17 14v7h-7"></path>
            </svg>
            Compress
        `;
    }
}

// Export to AI-Toolkit format
async function exportDataset() {
    if (!currentFolder) {
        alert('Please select a dataset folder first.');
        return;
    }

    const exportPath = prompt(
        'Enter the export path:\n\n' +
        'Folders will be created as:\n' +
        `• ${currentFolder}_img\n` +
        `• ${currentFolder}_ctr1\n` +
        `• ${currentFolder}_ctr2\n` +
        `• ${currentFolder}_ctr3\n\n` +
        '(Empty folders will be skipped)'
    );

    if (!exportPath) return;

    try {
        exportBtn.disabled = true;
        exportBtn.textContent = 'Exporting...';

        const response = await fetch(`/api/export?folder=${encodeURIComponent(currentFolder)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exportPath })
        });

        const data = await response.json();

        if (data.success) {
            const summary = Object.entries(data.exported)
                .map(([folder, info]) => `${folder}: ${info.files} files`)
                .join('\n');
            alert(`Export complete!\n\nPath: ${data.exportPath}\n\n${summary}`);
        } else {
            alert(`Failed to export: ${data.error}`);
        }
    } catch (error) {
        console.error('Export failed:', error);
        alert('Failed to export dataset. Check console for details.');
    } finally {
        exportBtn.disabled = false;
        exportBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"></path>
            </svg>
            Export
        `;
    }
}

// Load caption for current image
// Track unsaved caption changes
let captionDirty = false;

function autoresizeCaption() {
    captionText.style.height = 'auto';
    captionText.style.height = captionText.scrollHeight + 'px';
}

// Returns false if navigation should be cancelled (user chose not to discard)
async function checkUnsavedCaption() {
    if (!captionDirty) return true;
    const choice = confirm('Caption has unsaved changes.\n\nSave before continuing?');
    if (choice) await saveCurrentCaption();
    return true; // proceed either way after user acknowledged
}

async function loadCaption(filename) {
    try {
        captionText.value = 'Loading...';
        captionDirty = false;
        autoresizeCaption();
        const response = await fetch(`/api/caption/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}`);
        const data = await response.json();

        if (data.error) {
            console.error('Error loading caption:', data.error);
            captionText.value = '';
        } else {
            captionText.value = data.caption || '';
        }
        captionDirty = false;
        autoresizeCaption();
    } catch (error) {
        console.error('Failed to load caption:', error);
        captionText.value = '';
        captionDirty = false;
    }
}

// Save current caption
async function saveCurrentCaption() {
    if (images.length === 0) return;

    const filename = images[currentIndex];
    const caption = captionText.value;

    try {
        saveCaptionBtn.disabled = true;
        const response = await fetch(`/api/caption/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ caption })
        });

        const data = await response.json();

        if (data.success) {
            captionDirty = false;
            const originalBackground = saveCaptionBtn.style.background;
            saveCaptionBtn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
            setTimeout(() => { saveCaptionBtn.style.background = originalBackground; }, 1000);
        } else {
            alert(`Failed to save caption: ${data.error}`);
        }
    } catch (error) {
        console.error('Failed to save caption:', error);
        alert('Failed to save caption. Check console for details.');
    } finally {
        saveCaptionBtn.disabled = false;
    }
}

// Open current image pair in Pixelmator Pro
async function openInPixelmator() {
    if (images.length === 0) return;

    const filename = images[currentIndex];

    try {
        pixelmatorBtn.disabled = true;
        pixelmatorBtn.classList.add('loading');

        const response = await fetch('/api/open-in-pixelmator', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                folder: currentFolder,
                filename: filename
            })
        });

        const data = await response.json();

        if (data.success) {
            console.log('Opened in Pixelmator Pro');
        } else {
            alert(`Failed to open in Pixelmator: ${data.error}`);
        }
    } catch (error) {
        console.error('Pixelmator opening failed:', error);
        alert('Failed to call Pixelmator API. Check console.');
    } finally {
        pixelmatorBtn.disabled = false;
        pixelmatorBtn.classList.remove('loading');
    }
}

// Event listeners
function setupEventListeners() {
    // Folder selection
    folderSelect.addEventListener('change', (e) => {
        if (e.target.value === '__create_new__') {
            createNewDataset();
        } else {
            loadImages(e.target.value);
        }
    });

    // Image grid event delegation
    imageGrid.addEventListener('click', (e) => {
        const item = e.target.closest('.image-item');
        if (!item || item.dataset.index === undefined) return;
        const idx = parseInt(item.dataset.index);
        if (stitchMode) {
            toggleStitchSelection(idx);
        } else {
            openPreview(idx);
        }
    });

    // Modal controls
    closeBtn.addEventListener('click', closePreview);
    prevBtn.addEventListener('click', showPrevious);
    nextBtn.addEventListener('click', showNext);
    toggleBtn.addEventListener('click', toggleOverlay);
    duplicateBtn.addEventListener('click', duplicateCurrentImage);
    pixelmatorBtn.addEventListener('click', openInPixelmator);
    deleteBtn.addEventListener('click', deleteCurrentImage);
    transferBtn.addEventListener('click', transferCurrentImage);
    saveCaptionBtn.addEventListener('click', saveCurrentCaption);

    captionText.addEventListener('input', () => {
        captionDirty = true;
        autoresizeCaption();
    });
    reshuffleBtn.addEventListener('click', reshuffleDataset);
    compressBtn.addEventListener('click', compressDataset);
    fitBtn.addEventListener('click', fitDataset);
    exportBtn.addEventListener('click', exportDataset);
    document.getElementById('process-text-btn').addEventListener('click', openProcessTextModal);
    document.getElementById('stitch-btn').addEventListener('click', () => {
        if (stitchMode) deactivateStitchMode();
        else activateStitchMode();
    });
    document.getElementById('stitch-cancel-btn').addEventListener('click', deactivateStitchMode);
    document.getElementById('stitch-confirm-btn').addEventListener('click', performStitch);
    document.getElementById('stitch-error-close').addEventListener('click', hideStitchError);
    document.getElementById('stitch-error-modal').addEventListener('click', (e) => {
        if (e.target === document.getElementById('stitch-error-modal')) hideStitchError();
    });
    document.getElementById('stitch-count').addEventListener('change', updateStitchCounter);
    document.querySelectorAll('.stitch-dir-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.stitch-dir-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const is2x2 = btn.dataset.dir === 'grid2x2';
            const countEl = document.getElementById('stitch-count');
            const countLabel = document.getElementById('stitch-count-label');
            countEl.style.display = is2x2 ? 'none' : '';
            countLabel.style.display = is2x2 ? 'none' : '';
            if (is2x2) countEl.value = '4';
            updateStitchCounter();
        });
    });

    // Target dataset selection
    targetDatasetSelect.addEventListener('change', (e) => {
        onTargetDatasetChange(e.target.value);
    });

    // Control thumbnail clicks
    Object.entries(controlThumbs).forEach(([controlName, thumb]) => {
        thumb.addEventListener('click', () => {
            showControlFullPreview(controlName);
        });
    });

    // Link dataset controls
    linkBtn.addEventListener('click', showLinkSelector);
    linkSelect.addEventListener('change', (e) => {
        linkDataset(e.target.value);
    });
    unlinkBtn.addEventListener('click', unlinkDataset);

    // Opacity slider
    opacitySlider.addEventListener('input', (e) => {
        updateOpacity(e.target.value);
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (!modal.classList.contains('active')) return;

        // Handle Ctrl+S for saving caption
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            saveCurrentCaption();
            return;
        }

        // Don't trigger navigation if user is typing in the caption textarea
        if (document.activeElement === captionText) {
            return;
        }

        switch (e.key) {
            case 'Escape':
                closePreview();
                break;
            case 'ArrowLeft':
                showPrevious();
                break;
            case 'ArrowRight':
                showNext();
                break;
            case 'Backspace':
            case 'Delete':
                e.preventDefault();
                deleteCurrentImage();
                break;
            case ' ':
                e.preventDefault();
                toggleOverlay();
                break;
            case 'ArrowUp':
                if (targetFolder) {
                    transferCurrentImage();
                }
                break;
            case 'p':
            case 'P':
                openInPixelmator();
                break;
        }
    });

    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closePreview();
        }
    });

    // Process Text modal: close on backdrop click
    document.getElementById('process-text-modal').addEventListener('click', (e) => {
        if (e.target === document.getElementById('process-text-modal')) {
            closeProcessTextModal();
        }
    });
}

// ─── Process Text ──────────────────────────────────────────────────────────

const ptModal       = document.getElementById('process-text-modal');
const ptCloseBtn    = document.getElementById('pt-close-btn');
const ptDropsEl     = document.getElementById('pt-drops');
const ptSlotsBody   = document.getElementById('pt-slots-body');
const ptTemplateEl  = document.getElementById('pt-template');
const ptPreviewList = document.getElementById('pt-preview-list');
const ptAddSlotBtn  = document.getElementById('pt-add-slot-btn');
const ptRefreshBtn  = document.getElementById('pt-refresh-btn');
const ptPreviewBtn  = document.getElementById('pt-preview-btn');
const ptApplyBtn    = document.getElementById('pt-apply-btn');
const ptSaveBtn     = document.getElementById('pt-save-config-btn');
const ptLoadBtn     = document.getElementById('pt-load-config-btn');

let ptPreviewDebounce = null;

// ── Open / Close ─────────────────────────────────────────────────────────────

async function openProcessTextModal() {
    if (!currentFolder) {
        alert('Please select a dataset folder first.');
        return;
    }
    ptModal.classList.add('active');
    await ptLoadConfig();
}

function closeProcessTextModal() {
    ptModal.classList.remove('active');
}

ptCloseBtn.addEventListener('click', closeProcessTextModal);

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && ptModal.classList.contains('active')) {
        closeProcessTextModal();
    }
});

// ── Config helpers ────────────────────────────────────────────────────────────

function ptGetConfig() {
    const drops = ptDropsEl.value
        .split('\n')
        .map(l => l.trim())
        .filter(Boolean);

    const slots = [];
    ptSlotsBody.querySelectorAll('.pt-slot-row').forEach(row => {
        const [nameEl, patternEl, transformEl] = row.querySelectorAll('input');
        const name    = nameEl.value.trim();
        const pattern = patternEl.value.trim();
        const transform = transformEl.value.trim();
        if (name && pattern) {
            slots.push({ name, pattern, transform });
        }
    });

    return {
        drops,
        slots,
        template: ptTemplateEl.value
    };
}

function ptSetConfig(cfg) {
    ptDropsEl.value = (cfg.drops || []).join('\n');
    ptTemplateEl.value = cfg.template || '';

    ptSlotsBody.innerHTML = '';
    (cfg.slots || []).forEach(slot => ptAddSlotRow(slot));
}

// ── Slot rows ─────────────────────────────────────────────────────────────────

function ptAddSlotRow(slot = {}) {
    const row = document.createElement('div');
    row.className = 'pt-slot-row';
    row.innerHTML = `
        <input type="text"  class="pt-input pt-input-name"      placeholder="age"            value="${escHtml(slot.name      || '')}">
        <input type="text"  class="pt-input pt-input-pattern"   placeholder="^(\\d+)yo$"     value="${escHtml(slot.pattern   || '')}">
        <input type="text"  class="pt-input pt-input-transform" placeholder="$1yo"           value="${escHtml(slot.transform || '')}">
        <button class="pt-del-slot-btn" title="Remove slot">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
        </button>`;
    row.querySelector('.pt-del-slot-btn').addEventListener('click', () => row.remove());

    // Auto-preview on change (debounced)
    row.querySelectorAll('input').forEach(inp => {
        inp.addEventListener('input', ptSchedulePreview);
    });

    ptSlotsBody.appendChild(row);
}

function escHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

ptAddSlotBtn.addEventListener('click', () => ptAddSlotRow());

// ── Save / Load config ────────────────────────────────────────────────────────

async function ptSaveConfig() {
    if (!currentFolder) return;
    try {
        const res = await fetch(`/api/process-text/config?folder=${encodeURIComponent(currentFolder)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(ptGetConfig())
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error);
        ptShowStatus('Config saved.', 'success');
    } catch (e) {
        ptShowStatus('Save failed: ' + e.message, 'error');
    }
}

async function ptLoadConfig() {
    if (!currentFolder) return;
    try {
        const res  = await fetch(`/api/process-text/config?folder=${encodeURIComponent(currentFolder)}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        ptSetConfig(data);
        ptSchedulePreview();
    } catch (e) {
        ptShowStatus('Load failed: ' + e.message, 'error');
    }
}

ptSaveBtn.addEventListener('click', ptSaveConfig);
ptLoadBtn.addEventListener('click', ptLoadConfig);

// ── Preview ───────────────────────────────────────────────────────────────────

function ptSchedulePreview() {
    clearTimeout(ptPreviewDebounce);
    ptPreviewDebounce = setTimeout(ptRunPreview, 600);
}

async function ptRunPreview() {
    if (!currentFolder) return;
    ptPreviewList.innerHTML = '<div class="pt-preview-empty pt-loading">Generating preview…</div>';
    try {
        const res  = await fetch('/api/process-text/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder: currentFolder, config: ptGetConfig(), count: 5 })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        if (!data.results.length) {
            ptPreviewList.innerHTML = '<div class="pt-preview-empty">No caption files found in this dataset.</div>';
            return;
        }

        const countEl = ptModal.querySelector('.pt-preview-count');
        if (countEl) countEl.textContent = `(${data.results.length} samples)`;

        ptPreviewList.innerHTML = data.results.map(r => `
            <div class="pt-preview-item">
                <div class="pt-preview-filename">${escHtml(r.filename)}</div>
                <div class="pt-preview-row">
                    <div class="pt-preview-col">
                        <div class="pt-preview-label">Before</div>
                        <div class="pt-preview-text pt-preview-before">${escHtml(r.original)}</div>
                    </div>
                    <div class="pt-preview-arrow">→</div>
                    <div class="pt-preview-col">
                        <div class="pt-preview-label">After</div>
                        <div class="pt-preview-text pt-preview-after">${escHtml(r.processed)}</div>
                    </div>
                </div>
            </div>`).join('');
    } catch (e) {
        ptPreviewList.innerHTML = `<div class="pt-preview-empty pt-error">Preview error: ${escHtml(e.message)}</div>`;
    }
}

ptRefreshBtn.addEventListener('click', ptRunPreview);
ptPreviewBtn.addEventListener('click', ptRunPreview);
ptDropsEl.addEventListener('input', ptSchedulePreview);
ptTemplateEl.addEventListener('input', ptSchedulePreview);

// ── Apply ─────────────────────────────────────────────────────────────────────

ptApplyBtn.addEventListener('click', async () => {
    if (!currentFolder) return;
    const ok = confirm(
        `Apply processing to ALL captions in "${currentFolder}"?\n\nThis will overwrite every .txt file. The operation cannot be undone.`
    );
    if (!ok) return;

    ptApplyBtn.disabled = true;
    ptApplyBtn.textContent = 'Applying…';

    try {
        const res  = await fetch('/api/process-text/apply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder: currentFolder, config: ptGetConfig() })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        const errMsg = data.errors?.length
            ? `\n${data.errors.length} error(s): ${data.errors.map(e => e.file).join(', ')}`
            : '';
        ptShowStatus(`Done — ${data.processed} captions updated.${errMsg}`, data.errors?.length ? 'warning' : 'success');
        ptRunPreview();
    } catch (e) {
        ptShowStatus('Apply failed: ' + e.message, 'error');
    } finally {
        ptApplyBtn.disabled = false;
        ptApplyBtn.textContent = 'Apply to All';
    }
});

// ── Status bar ────────────────────────────────────────────────────────────────

function ptShowStatus(msg, type = 'info') {
    let bar = ptModal.querySelector('.pt-status-bar');
    if (!bar) {
        bar = document.createElement('div');
        bar.className = 'pt-status-bar';
        ptModal.querySelector('.pt-footer').prepend(bar);
    }
    bar.textContent = msg;
    bar.dataset.type = type;
    bar.classList.add('visible');
    setTimeout(() => bar.classList.remove('visible'), 4000);
}

// ─── Stitch Mode ──────────────────────────────────────────────────────────────

function activateStitchMode() {
    if (!currentFolder) {
        alert('Please select a dataset folder first.');
        return;
    }
    stitchMode = true;
    stitchSelectedIndices = [];
    document.getElementById('stitch-bar').classList.remove('hidden');
    document.getElementById('stitch-btn').classList.add('active');
    updateStitchCounter();
    // Add selection-mode class to grid for cursor change
    imageGrid.classList.add('stitch-select-mode');
}

function deactivateStitchMode() {
    stitchMode = false;
    stitchSelectedIndices = [];
    stitchUsedFilenames.clear();
    document.getElementById('stitch-bar').classList.add('hidden');
    document.getElementById('stitch-btn').classList.remove('active');
    imageGrid.classList.remove('stitch-select-mode');
    // Restore all hidden/highlighted items
    document.querySelectorAll('.image-item.stitch-picked, .image-item.stitch-invalid').forEach(el => {
        el.classList.remove('stitch-picked', 'stitch-invalid');
    });
    // Clear thumbs row
    document.getElementById('stitch-thumbs-row').innerHTML = '';
}

function showStitchError(msg, invalidItems = []) {
    document.getElementById('stitch-error-msg').textContent = msg;
    document.getElementById('stitch-error-modal').classList.remove('hidden');
    invalidItems.forEach(el => {
        el.classList.add('stitch-invalid');
        // Auto-remove highlight after 4s
        setTimeout(() => el.classList.remove('stitch-invalid'), 4000);
    });
}

function hideStitchError() {
    document.getElementById('stitch-error-modal').classList.add('hidden');
}

function getStitchCount() {
    return parseInt(document.getElementById('stitch-count').value) || 3;
}

function updateStitchCounter() {
    const count = getStitchCount();
    const selected = stitchSelectedIndices.length;
    document.getElementById('stitch-counter').textContent = `Click images to select (${selected} / ${count})`;
    const confirmBtn = document.getElementById('stitch-confirm-btn');
    confirmBtn.disabled = selected < count;
}

function toggleStitchSelection(index) {
    const count = getStitchCount();
    const item = imageGrid.querySelector(`.image-item[data-index="${index}"]`);
    if (!item) return;

    {
        if (stitchSelectedIndices.length >= count) return; // already at max

        // 2×2 mode: validate that image is square
        const direction = document.querySelector('.stitch-dir-btn.active')?.dataset.dir;
        if (direction === 'grid2x2') {
            const img = item.querySelector('img');
            if (img && img.naturalWidth && img.naturalHeight) {
                const ratio = img.naturalWidth / img.naturalHeight;
                if (ratio < 0.9 || ratio > 1.1) {
                    const w = img.naturalWidth, h = img.naturalHeight;
                    showStitchError(
                        `Изображение «${images[index]}» не квадратное (${w}×${h}).\nРежим 2×2 требует квадратные изображения.`,
                        [item]
                    );
                    return;
                }
            }
        }

        stitchSelectedIndices.push(index);
        stitchUsedFilenames.add(images[index]);
        // Hide the item from grid so user sees only remaining choices
        item.classList.add('stitch-picked');
        // Add thumbnail to stitch bar
        addStitchThumb(index, stitchSelectedIndices.length);
    }
    updateStitchCounter();
}

function addStitchThumb(index, order) {
    const filename = images[index];
    const thumbRow = document.getElementById('stitch-thumbs-row');

    const wrap = document.createElement('div');
    wrap.className = 'stitch-thumb';
    wrap.dataset.index = index;

    const img = document.createElement('img');
    img.src = `/api/image/img/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}`;
    img.alt = filename;
    img.title = filename;

    const orderBadge = document.createElement('span');
    orderBadge.className = 'stitch-thumb-order';
    orderBadge.textContent = order;

    const removeBtn = document.createElement('button');
    removeBtn.className = 'stitch-thumb-remove';
    removeBtn.title = 'Remove from selection';
    removeBtn.textContent = '✕';
    removeBtn.addEventListener('click', () => removeStitchThumb(index));

    wrap.appendChild(img);
    wrap.appendChild(orderBadge);
    wrap.appendChild(removeBtn);
    thumbRow.appendChild(wrap);
}

function removeStitchThumb(index) {
    const pos = stitchSelectedIndices.indexOf(index);
    if (pos === -1) return;

    // Restore grid item and remove from used set
    const gridItem = imageGrid.querySelector(`.image-item[data-index="${index}"]`);
    if (gridItem) gridItem.classList.remove('stitch-picked');
    stitchUsedFilenames.delete(images[index]);

    // Remove from selection
    stitchSelectedIndices.splice(pos, 1);

    // Remove thumb element
    const thumbRow = document.getElementById('stitch-thumbs-row');
    const thumb = thumbRow.querySelector(`.stitch-thumb[data-index="${index}"]`);
    if (thumb) thumb.remove();

    // Re-number remaining thumbs
    thumbRow.querySelectorAll('.stitch-thumb').forEach((el, i) => {
        const badge = el.querySelector('.stitch-thumb-order');
        if (badge) badge.textContent = i + 1;
    });

    updateStitchCounter();
}

async function performStitch() {
    const count = getStitchCount();
    if (stitchSelectedIndices.length < count) return;

    const direction = document.querySelector('.stitch-dir-btn.active')?.dataset.dir || 'horizontal';
    const asIsControl = document.getElementById('stitch-asis-control').value;
    const filenames = stitchSelectedIndices.map(i => images[i]);

    const confirmBtn = document.getElementById('stitch-confirm-btn');
    confirmBtn.disabled = true;
    confirmBtn.textContent = 'Stitching...';

    try {
        const response = await fetch('/api/stitch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                folder: currentFolder,
                filenames,
                direction,
                asIsControl
            })
        });

        const data = await response.json();

        if (data.success) {
            // Stay in stitch mode — just reset the current round selection
            stitchSelectedIndices = [];
            document.getElementById('stitch-thumbs-row').innerHTML = '';
            updateStitchCounter();

            bumpFolderBuster();
            await loadImages(currentFolder);
            // renderImageGrid already re-applies stitch-picked for stitchUsedFilenames

            // Highlight the new stitched image
            const newIdx = images.indexOf(data.newFilename);
            if (newIdx !== -1) {
                const newItem = imageGrid.querySelector(`.image-item[data-index="${newIdx}"]`);
                if (newItem) {
                    newItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    newItem.classList.add('stitch-new-flash');
                    setTimeout(() => newItem.classList.remove('stitch-new-flash'), 1500);
                }
            }
        } else {
            alert(`Stitch failed: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Stitch failed:', error);
        alert('Stitch failed. Check console for details.');
    } finally {
        confirmBtn.disabled = false;
        confirmBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Create Stitch
        `;
    }
}
