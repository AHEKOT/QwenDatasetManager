// State management
let currentFolder = '';
let targetFolder = ''; // For transfer functionality
let images = [];
let currentIndex = 0;
let overlayActive = false;
let opacityValue = 50; // Default 50%
let cacheBuster = Date.now(); // For cache busting after reshuffle
let allFolders = []; // Store all folders for target selection
let activeControlView = null; // Which control is shown in full preview (null = original image)
let comparisonControlView = null; // Which control is shown in comparison view (null = hidden)
let linkedDataset = null; // Linked dataset for synchronized operations
let imageEditor = null; // Image editor instance

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
const deleteBtn = document.getElementById('delete-btn');
const reshuffleBtn = document.getElementById('reshuffle-btn');
const compressBtn = document.getElementById('compress-btn');
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
document.addEventListener('DOMContentLoaded', () => {
    loadFolders();
    setupEventListeners();
});

// Global callback for editor to notify when image is saved
// Global callback for editor to notify when image is saved
window.onImageSaved = function (reloadList = false) {
    cacheBuster = Date.now();
    updatePreview();

    if (reloadList) {
        // Reload the entire grid (for new images)
        loadImages(currentFolder);
    } else {
        // Just refresh the grid thumbnail for this image
        const gridImg = document.querySelector(`.image-item:nth-child(${currentIndex + 1}) img`);
        if (gridImg) {
            const src = gridImg.src.split('?')[0];
            gridImg.src = `${src}?folder=${encodeURIComponent(currentFolder)}&t=${cacheBuster}`;
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

    imageGrid.innerHTML = '';
    images.forEach((filename, index) => {
        const item = document.createElement('div');
        item.className = 'image-item';
        item.onclick = () => openPreview(index);

        const img = document.createElement('img');
        img.src = `/api/image/img/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}&t=${cacheBuster}`;
        img.alt = filename;
        img.loading = 'lazy';

        const filenameSpan = document.createElement('span');
        filenameSpan.className = 'filename';
        filenameSpan.textContent = filename;

        item.appendChild(img);
        item.appendChild(filenameSpan);
        imageGrid.appendChild(item);
    });
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
}

// Update preview with current image
function updatePreview() {
    if (images.length === 0) return;

    const filename = images[currentIndex];
    const baseUrl = `/api/image`;
    const folderParam = `?folder=${encodeURIComponent(currentFolder)}&t=${cacheBuster}`;
    const imageContainer = document.querySelector('.image-container');

    if (!imageContainer) {
        console.error('imageContainer not found');
        return;
    }

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
    if (imageEditor && previewImg.complete) {
        setTimeout(() => imageEditor.updateCanvasSize(), 50);
    }

    // Initialize canvas editor
    previewImg.onload = () => {

        const controlImg = new Image();
        controlImg.crossOrigin = 'anonymous';
        controlImg.src = `${baseUrl}/Control1/${encodeURIComponent(filename)}${folderParam}`;
        controlImg.onload = () => {
            const canvas = document.getElementById('edit-canvas');
            if (!imageEditor) {
                imageEditor = new ImageEditor(canvas, previewImg, controlImg, previewControl);
            } else {
                imageEditor.targetImageElement = previewImg;
                imageEditor.controlImageElement = controlImg;
                imageEditor.overlayElement = previewControl;
                imageEditor.setupCanvas();
                imageEditor.history = [];
                imageEditor.saveState();
            }
            imageEditor.currentFilename = filename;
            imageEditor.currentFolder = currentFolder;
            imageEditor.updateSaveButton();
        };
    };

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
    const folderParam = `?folder=${encodeURIComponent(currentFolder)}&t=${cacheBuster}`;

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
}

// Navigate to previous image
function showPrevious() {
    if (currentIndex > 0) {
        currentIndex--;
        updatePreview();
    }
}

// Navigate to next image
function showNext() {
    if (currentIndex < images.length - 1) {
        currentIndex++;
        updatePreview();
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

            // Update UI
            if (images.length === 0) {
                closePreview();
                renderImageGrid();
                updateImageCount();
            } else {
                if (currentIndex >= images.length) {
                    currentIndex = images.length - 1;
                }
                updatePreview();
                renderImageGrid();
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

            // Update UI
            if (images.length === 0) {
                closePreview();
                renderImageGrid();
                updateImageCount();
            } else {
                // Adjust index if needed
                if (currentIndex >= images.length) {
                    currentIndex = images.length - 1;
                }
                updatePreview();
                renderImageGrid();
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
            // Update cache buster to force reload of images
            cacheBuster = Date.now();
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
            cacheBuster = Date.now();
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
async function loadCaption(filename) {
    try {
        captionText.value = 'Loading...';
        const response = await fetch(`/api/caption/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}`);
        const data = await response.json();

        if (data.error) {
            console.error('Error loading caption:', data.error);
            captionText.value = '';
            return;
        }

        captionText.value = data.caption || '';
    } catch (error) {
        console.error('Failed to load caption:', error);
        captionText.value = '';
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
            console.log('Caption saved successfully');
            // Visual feedback
            const originalBackground = saveCaptionBtn.style.background;
            saveCaptionBtn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
            setTimeout(() => {
                saveCaptionBtn.style.background = originalBackground;
            }, 1000);
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

    // Modal controls
    closeBtn.addEventListener('click', closePreview);
    prevBtn.addEventListener('click', showPrevious);
    nextBtn.addEventListener('click', showNext);
    toggleBtn.addEventListener('click', toggleOverlay);
    deleteBtn.addEventListener('click', deleteCurrentImage);
    transferBtn.addEventListener('click', transferCurrentImage);
    saveCaptionBtn.addEventListener('click', saveCurrentCaption);
    reshuffleBtn.addEventListener('click', reshuffleDataset);
    compressBtn.addEventListener('click', compressDataset);
    exportBtn.addEventListener('click', exportDataset);

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
                e.preventDefault();
                if (targetFolder) {
                    transferCurrentImage();
                }
                break;
        }
    });

    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closePreview();
        }
    });
}
