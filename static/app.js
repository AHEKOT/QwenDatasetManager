// State management
let currentFolder = '';
let targetFolder = ''; // For transfer/copy functionality
let transferMode = 'transfer';
let images = [];
let blurPreviewFilename = '';
let mirrorPreviewFilename = '';
let duplicateReviewState = {
    pairs: [],
    index: 0,
    threshold: 8
};
let importScanResults = [];
let currentToolsView = 'blur';
let activeImportJobId = null;
let activeImportButton = null;
let activeImportCard = null;
let importPollTimer = null;
let datasetNameModalState = null;

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
let captionRequestController = null;
let captionLoadToken = 0;
let captionLoadedKey = '';

// DOM elements
const folderSelect = document.getElementById('folder-select');
const refreshFoldersBtn = document.getElementById('refresh-folders-btn');
const renameDatasetBtn = document.getElementById('rename-dataset-btn');
const datasetNameModal = document.getElementById('dataset-name-modal');
const datasetNameModalKicker = document.getElementById('dataset-name-modal-kicker');
const datasetNameModalTitle = document.getElementById('dataset-name-modal-title');
const datasetNameModalDescription = document.getElementById('dataset-name-modal-description');
const datasetNameInput = document.getElementById('dataset-name-input');
const datasetNameModalError = document.getElementById('dataset-name-modal-error');
const datasetNameCloseBtn = document.getElementById('dataset-name-close-btn');
const datasetNameCancelBtn = document.getElementById('dataset-name-cancel-btn');
const datasetNameSubmitBtn = document.getElementById('dataset-name-submit-btn');
const imageGrid = document.getElementById('image-grid');
const imageCount = document.getElementById('image-count');
const toolsBtn = document.getElementById('tools-btn');
const toolsModal = document.getElementById('tools-modal');
const toolsCloseBtn = document.getElementById('tools-close-btn');
const toolsCurrentFolder = document.getElementById('tools-current-folder');
const toolsCurrentCount = document.getElementById('tools-current-count');
const toolsNavButtons = Array.from(document.querySelectorAll('.tools-nav-btn'));
const toolsDetailPanels = Array.from(document.querySelectorAll('.tools-detail-panel'));
const toolProgressCards = new Map(
    Array.from(document.querySelectorAll('[data-tool-progress]')).map((card) => [
        card.dataset.toolProgress,
        {
            card,
            title: card.querySelector('[data-tool-progress-title]'),
            percent: card.querySelector('[data-tool-progress-percent]'),
            fill: card.querySelector('[data-tool-progress-fill]'),
            summary: card.querySelector('[data-tool-progress-summary]'),
            details: card.querySelector('[data-tool-progress-details]')
        }
    ])
);
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
const duplicateThresholdInput = document.getElementById('duplicate-threshold');
const duplicateThresholdValue = document.getElementById('duplicate-threshold-value');
const duplicateScanBtn = document.getElementById('duplicate-scan-btn');
const duplicateReview = document.getElementById('duplicate-review');
const duplicateEmpty = document.getElementById('duplicate-empty');
const duplicateStage = document.getElementById('duplicate-stage');
const duplicateProgress = document.getElementById('duplicate-progress');
const duplicateDistance = document.getElementById('duplicate-distance');
const duplicateSkipBtn = document.getElementById('duplicate-skip-btn');
const duplicateLeftCard = document.getElementById('duplicate-left-card');
const duplicateRightCard = document.getElementById('duplicate-right-card');
const duplicateLeftImg = document.getElementById('duplicate-left-img');
const duplicateRightImg = document.getElementById('duplicate-right-img');
const duplicateLeftName = document.getElementById('duplicate-left-name');
const duplicateRightName = document.getElementById('duplicate-right-name');
const duplicateLeftCaption = document.getElementById('duplicate-left-caption');
const duplicateRightCaption = document.getElementById('duplicate-right-caption');
const blurStrengthInput = document.getElementById('blur-strength');
const blurStrengthValue = document.getElementById('blur-strength-value');
const blurPreviewImage = document.getElementById('blur-preview-image');
const blurPreviewEmpty = document.getElementById('blur-preview-empty');
const blurPreviewName = document.getElementById('blur-preview-name');
const blurRerollBtn = document.getElementById('blur-reroll-btn');
const blurApplyBtn = document.getElementById('blur-apply-btn');
const mirrorHorizontalInput = document.getElementById('mirror-horizontal');
const mirrorVerticalInput = document.getElementById('mirror-vertical');
const mirrorExcludeControl1Input = document.getElementById('mirror-exclude-control1');
const mirrorExcludeControl2Input = document.getElementById('mirror-exclude-control2');
const mirrorExcludeControl3Input = document.getElementById('mirror-exclude-control3');
const mirrorPreviewImage = document.getElementById('mirror-preview-image');
const mirrorPreviewEmpty = document.getElementById('mirror-preview-empty');
const mirrorPreviewName = document.getElementById('mirror-preview-name');
const mirrorRerollBtn = document.getElementById('mirror-reroll-btn');
const mirrorApplyBtn = document.getElementById('mirror-apply-btn');
const mergePrimaryName = document.getElementById('merge-primary-name');
const mergeSecondarySelect = document.getElementById('merge-secondary-select');
const mergeTargetNameInput = document.getElementById('merge-target-name');
const mergeApplyBtn = document.getElementById('merge-apply-btn');
const importPathInput = document.getElementById('import-path-input');
const importScanBtn = document.getElementById('import-scan-btn');
const importStatus = document.getElementById('import-status');
const importProgressCard = document.getElementById('import-progress-card');
const importProgressTitle = document.getElementById('import-progress-title');
const importProgressSummary = document.getElementById('import-progress-summary');
const importProgressPercent = document.getElementById('import-progress-percent');
const importProgressFill = document.getElementById('import-progress-fill');
const importProgressFolders = document.getElementById('import-progress-folders');
const importResults = document.getElementById('import-results');
const exportBtn = document.getElementById('export-btn');
const exportModal = document.getElementById('export-modal');
const exportPathInput = document.getElementById('export-path-input');
const exportModalError = document.getElementById('export-modal-error');
const exportCloseBtn = document.getElementById('export-close-btn');
const exportCancelBtn = document.getElementById('export-cancel-btn');
const exportSubmitBtn = document.getElementById('export-submit-btn');
const exportProgressCard = document.getElementById('export-progress-card');
const exportProgressTitle = document.getElementById('export-progress-title');
const exportProgressPercent = document.getElementById('export-progress-percent');
const exportProgressFill = document.getElementById('export-progress-fill');
const exportProgressSummary = document.getElementById('export-progress-summary');
const exportProgressDetails = document.getElementById('export-progress-details');
const opacitySlider = document.getElementById('opacity-slider');
const opacityValueDisplay = document.getElementById('opacity-value');
const transferModeSelect = document.getElementById('transfer-mode-select');
const transferModeHelp = document.getElementById('transfer-mode-help');
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
const IMPORT_CACHE_KEY = 'qdm_import_cache';
const EXPORT_PATH_KEY = 'qdm_export_path';
let _saveTimer = null;
let lastExportPath = loadLocalExportPath();
const toolPollTimers = new Map();
const toolJobContexts = new Map();

function loadLocalExportPath() {
    try {
        return localStorage.getItem(EXPORT_PATH_KEY) || '';
    } catch (e) {
        return '';
    }
}

function persistExportPath(path) {
    lastExportPath = path.trim();
    try {
        localStorage.setItem(EXPORT_PATH_KEY, lastExportPath);
    } catch (e) {
        console.warn('Failed to persist export path:', e);
    }
}

function loadImportCache() {
    try {
        const raw = localStorage.getItem(IMPORT_CACHE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return null;
        return parsed;
    } catch (e) {
        return null;
    }
}

function saveImportCache(cache) {
    try {
        localStorage.setItem(IMPORT_CACHE_KEY, JSON.stringify(cache));
    } catch (e) {
        /* localStorage unavailable */
    }
}

function persistImportPath(path) {
    const cache = loadImportCache() || {};
    cache.path = path;
    if (cache.cachedPath && cache.cachedPath !== path) {
        delete cache.cachedPath;
        delete cache.results;
        delete cache.cachedAt;
    }
    saveImportCache(cache);
}

function persistImportResults(path, results) {
    const cache = loadImportCache() || {};
    cache.path = path;
    cache.cachedPath = path;
    cache.results = results;
    cache.cachedAt = Date.now();
    saveImportCache(cache);
}

function clearPersistedImportResults(path) {
    const cache = loadImportCache() || {};
    cache.path = path;
    delete cache.cachedPath;
    delete cache.results;
    delete cache.cachedAt;
    saveImportCache(cache);
}

function restoreImportState() {
    const cache = loadImportCache();
    if (!cache) {
        renderImportResults();
        return;
    }

    if (typeof cache.path === 'string') {
        importPathInput.value = cache.path;
    }

    if (cache.cachedPath && cache.cachedPath === importPathInput.value.trim() && Array.isArray(cache.results)) {
        importScanResults = cache.results;
        renderImportResults();
        const cachedDate = cache.cachedAt ? new Date(cache.cachedAt) : null;
        const cachedLabel = cachedDate && !Number.isNaN(cachedDate.getTime())
            ? `Loaded cached scan from ${cachedDate.toLocaleString()}. Press Scan to refresh.`
            : 'Loaded cached scan. Press Scan to refresh.';
        setImportStatus(cachedLabel, importScanResults.length ? 'success' : 'warning');
        return;
    }

    renderImportResults();
}

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
    restoreImportState();
    await restoreAppState();
});




// Sync Save and Undo buttons based on both editors
function syncEditorButtons() {
    const undoBtn  = document.getElementById('undo-btn');
    const resetBtn = document.getElementById('reset-edit-btn');
    const saveBtn  = document.getElementById('save-edit-btn');

    const activeEditor = ImageEditor.activeEditor || mainEditor;
    const hasChanges = Boolean(activeEditor && activeEditor.history.length > 1);

    [undoBtn, resetBtn, saveBtn].forEach(btn => {
        if (btn) btn.classList.toggle('hidden', !hasChanges);
    });
    if (undoBtn) undoBtn.disabled = !hasChanges;
    if (saveBtn) saveBtn.disabled = !hasChanges;
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

async function refreshDatasetsView() {
    const selectedFolder = currentFolder || folderSelect.value;

    try {
        refreshFoldersBtn.disabled = true;
        await loadFolders();

        if (selectedFolder && allFolders.some(folder => folder.path === selectedFolder)) {
            folderSelect.value = selectedFolder;
            await loadImages(selectedFolder);
        } else if (selectedFolder) {
            folderSelect.value = '';
            currentFolder = '';
            images = [];
            imageGrid.innerHTML = '<div class="empty-state"><p>📁 Select a dataset folder to view images</p></div>';
            updateImageCount();
            updateToolsContext();
        }
    } catch (error) {
        console.error('Failed to refresh datasets view:', error);
    } finally {
        refreshFoldersBtn.disabled = false;
    }
}

function closeDatasetNameModal(result = null) {
    if (!datasetNameModalState) return;

    const { resolve } = datasetNameModalState;
    datasetNameModalState = null;
    datasetNameModal.classList.add('hidden');
    datasetNameModalError.textContent = '';
    datasetNameModalError.classList.add('hidden');
    document.body.style.overflow = '';
    resolve(result);
}

function showDatasetNameError(message) {
    datasetNameModalError.textContent = message;
    datasetNameModalError.classList.remove('hidden');
}

function closeExportModal() {
    exportModal.classList.add('hidden');
    exportModalError.textContent = '';
    exportModalError.classList.add('hidden');
    document.body.style.overflow = '';
}

function showExportModalError(message) {
    exportModalError.textContent = message;
    exportModalError.classList.remove('hidden');
}

function renderExportProgress({
    state = 'running',
    title = 'Preparing export',
    summary = '',
    percent = 0,
    details = [],
    indeterminate = false,
    visible = true
} = {}) {
    if (!visible) {
        exportProgressCard.classList.add('hidden');
        return;
    }

    exportProgressCard.classList.remove('hidden');
    exportProgressCard.classList.toggle('is-running', state === 'running');
    exportProgressCard.classList.toggle('is-success', state === 'success');
    exportProgressCard.classList.toggle('is-error', state === 'error');
    exportProgressTitle.textContent = title;
    exportProgressSummary.textContent = summary;

    const clampedPercent = Math.max(0, Math.min(100, Number(percent) || 0));
    exportProgressPercent.textContent = indeterminate ? '...' : `${clampedPercent.toFixed(0)}%`;
    exportProgressFill.classList.toggle('is-indeterminate', indeterminate);
    exportProgressFill.style.width = indeterminate ? '42%' : `${clampedPercent}%`;
    exportProgressDetails.innerHTML = (details || [])
        .filter((item) => item && item.label && item.value !== undefined && item.value !== null && item.value !== '')
        .map((item) => `
            <div class="tools-task-detail">
                <span>${escHtml(item.label)}</span>
                <strong>${escHtml(item.value)}</strong>
            </div>
        `)
        .join('');
}

function openExportModal() {
    if (!currentFolder) {
        return;
    }

    exportPathInput.value = lastExportPath;
    exportModalError.textContent = '';
    exportModalError.classList.add('hidden');
    renderExportProgress({ visible: false });
    exportModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    requestAnimationFrame(() => {
        exportPathInput.focus();
        exportPathInput.select();
    });
}

function openDatasetNameModal({ kicker, title, description, submitLabel, initialValue = '' }) {
    if (datasetNameModalState) {
        closeDatasetNameModal(null);
    }

    datasetNameModalKicker.textContent = kicker;
    datasetNameModalTitle.textContent = title;
    datasetNameModalDescription.textContent = description;
    datasetNameSubmitBtn.textContent = submitLabel;
    datasetNameInput.value = initialValue;
    datasetNameModalError.textContent = '';
    datasetNameModalError.classList.add('hidden');
    datasetNameModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    const promise = new Promise((resolve) => {
        datasetNameModalState = { resolve };
    });

    requestAnimationFrame(() => {
        datasetNameInput.focus();
        datasetNameInput.select();
    });

    return promise;
}

function submitDatasetNameModal() {
    if (!datasetNameModalState) return;

    const value = datasetNameInput.value.trim();
    if (!value) {
        showDatasetNameError('Dataset name is required.');
        datasetNameInput.focus();
        return;
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(value)) {
        showDatasetNameError('Use only letters, numbers, underscores, and hyphens.');
        datasetNameInput.focus();
        return;
    }

    closeDatasetNameModal(value);
}

async function renameCurrentDataset() {
    const selectedFolder = currentFolder || folderSelect.value;
    if (!selectedFolder || selectedFolder === '__create_new__') {
        alert('Please select a dataset folder first.');
        return;
    }

    const currentName = selectedFolder.split('/').pop();
    const newName = await openDatasetNameModal({
        kicker: 'Dataset',
        title: 'Rename Dataset',
        description: 'Choose a clean dataset name. The manager will rename the dataset and its root folder without reloading the page.',
        submitLabel: 'Rename Dataset',
        initialValue: currentName
    });
    if (!newName) return;

    const trimmedName = newName.trim();
    if (!trimmedName) return;

    try {
        renameDatasetBtn.disabled = true;

        const response = await fetch('/api/rename-dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ oldName: selectedFolder, newName: trimmedName })
        });

        const data = await response.json();
        if (data.success) {
            currentFolder = data.path;
            if (linkedDataset === selectedFolder) {
                linkedDataset = data.path;
            }
            await loadFolders();
            folderSelect.value = data.path;
            await loadImages(data.path);
            saveAppState();
        } else {
            alert(`Failed to rename dataset: ${data.error}`);
        }
    } catch (error) {
        console.error('Failed to rename dataset:', error);
        alert('Failed to rename dataset. Check console for details.');
    } finally {
        renameDatasetBtn.disabled = false;
    }
}

// Create new dataset
async function createNewDataset() {
    const name = await openDatasetNameModal({
        kicker: 'Dataset',
        title: 'Create Dataset',
        description: 'Create a new dataset with the standard Qwen folder layout. Use letters, numbers, underscores, and hyphens only.',
        submitLabel: 'Create Dataset',
        initialValue: ''
    });

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
            imageGrid.innerHTML = `<div class="empty-state"><p>❌ Error: ${escHtml(data.error)}</p></div>`;
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
        updateToolsContext();
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
        item.tabIndex = 0;
        item.setAttribute('role', 'button');
        item.setAttribute('aria-label', `Open ${filename}`);

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

function setActionButtonLabel(button, label) {
    const labelEl = button?.querySelector('[data-button-label]');
    if (labelEl) {
        labelEl.textContent = label;
    } else if (button) {
        button.textContent = label;
    }
}

function setActionButtonBusy(button, busyLabel, isBusy) {
    if (!button) return;
    button.disabled = isBusy;
    setActionButtonLabel(button, isBusy ? busyLabel : button.dataset.defaultLabel || 'Action');
}

function renderToolProgress(toolName, {
    state = 'running',
    title = 'Working',
    summary = '',
    percent = 0,
    details = [],
    indeterminate = false,
    visible = true
} = {}) {
    const parts = toolProgressCards.get(toolName);
    if (!parts) return;

    const { card, fill } = parts;

    if (!visible) {
        card.classList.add('hidden');
        card.dataset.autoRevealed = '';
        return;
    }

    card.classList.remove('hidden');
    card.classList.toggle('is-running', state === 'running');
    card.classList.toggle('is-success', state === 'success');
    card.classList.toggle('is-error', state === 'error');
    parts.title.textContent = title;
    parts.summary.textContent = summary;

    const clampedPercent = Math.max(0, Math.min(100, Number(percent) || 0));
    parts.percent.textContent = indeterminate ? '...' : `${clampedPercent.toFixed(0)}%`;
    fill.classList.toggle('is-indeterminate', indeterminate);
    fill.style.width = indeterminate ? '42%' : `${clampedPercent}%`;

    parts.details.innerHTML = (details || [])
        .filter((item) => item && item.label && item.value !== undefined && item.value !== null && item.value !== '')
        .map((item) => `
            <div class="tools-task-detail">
                <span>${escHtml(item.label)}</span>
                <strong>${escHtml(item.value)}</strong>
            </div>
        `)
        .join('');

    if (state === 'running' && card.dataset.autoRevealed !== 'true') {
        card.dataset.autoRevealed = 'true';
        card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    if (state !== 'running') {
        card.dataset.autoRevealed = '';
    }
}

function showToolValidation(toolName, summary) {
    renderToolProgress(toolName, {
        state: 'error',
        title: 'Cannot start',
        summary,
        percent: 100
    });
}

function clearToolPolling(toolName) {
    const timer = toolPollTimers.get(toolName);
    if (timer) {
        clearTimeout(timer);
        toolPollTimers.delete(toolName);
    }
}

function renderToolJobState(toolName, job) {
    const context = toolJobContexts.get(toolName) || {};
    const total = Number(job.totalItems || 0);
    const processed = Number(job.processedItems || 0);
    const percent = Number(job.progressPercent || 0);
    const currentItem = job.currentItem || 'Preparing...';
    const result = job.result || {};

    if (job.status === 'error') {
        const baseDetails = {
            reshuffle: [{ label: 'Dataset', value: context.folderPath }],
            compress: [{ label: 'Dataset', value: context.folderPath }],
            fit: [{ label: 'Dataset', value: context.folderPath }],
            duplicates: [{ label: 'Dataset', value: context.folderPath }],
            blur: [{ label: 'Source Dataset', value: context.folderPath }],
            mirror: [{ label: 'Source Dataset', value: context.folderPath }],
            merge: [
                { label: 'Primary', value: context.primaryFolder },
                { label: 'Secondary', value: context.secondaryFolder },
                { label: 'Output Name', value: context.targetName }
            ]
        };

        renderToolProgress(toolName, {
            state: 'error',
            title: `${toolName[0].toUpperCase()}${toolName.slice(1)} failed`,
            summary: job.error || 'The background job failed.',
            percent: 100,
            details: baseDetails[toolName] || []
        });
        return;
    }

    if (job.status === 'completed') {
        const completedStates = {
            reshuffle: {
                title: 'Reshuffle complete',
                summary: `Finished renaming ${result.count} synchronized image sets in ${context.folderPath}.`,
                details: [
                    { label: 'Renamed Sets', value: result.count },
                    { label: 'Files Renamed', value: result.filesRenamed },
                    { label: 'Dataset', value: context.folderPath }
                ]
            },
            compress: {
                title: 'Compression complete',
                summary: `Optimized ${result.compressed} images and reduced dataset size by ${result.savingsMB} MB.`,
                details: [
                    { label: 'Compressed', value: result.compressed },
                    { label: 'Original Size', value: `${result.originalSizeMB} MB` },
                    { label: 'New Size', value: `${result.newSizeMB} MB` },
                    { label: 'Savings', value: `${result.savingsMB} MB (${result.savingsPercent}%)` }
                ]
            },
            fit: {
                title: 'Fit complete',
                summary: `Processed ${result.processed} image sets and updated ${result.updated} control images.`,
                details: [
                    { label: 'Processed Sets', value: result.processed },
                    { label: 'Updated Controls', value: result.updated }
                ]
            },
            duplicates: {
                title: result.count ? 'Duplicate pairs found' : 'No duplicates found',
                summary: result.count
                    ? `Found ${result.count} duplicate pair${result.count !== 1 ? 's' : ''}.`
                    : `No pairs matched at threshold ${result.threshold}.`,
                details: [
                    { label: 'Images Scanned', value: result.imageCount },
                    { label: 'Pairs', value: result.count },
                    { label: 'Threshold', value: result.threshold }
                ]
            },
            blur: {
                title: 'Blurred copy created',
                summary: `Created ${result.targetFolder} and processed ${result.processed} images with the backend blur pipeline.`,
                details: [
                    { label: 'Output Dataset', value: result.targetFolder },
                    { label: 'Strength', value: `${Number(result.strength).toFixed(1)} px` },
                    { label: 'Processed Images', value: result.processed }
                ]
            },
            mirror: {
                title: 'Mirrored copy created',
                summary: `Created ${result.targetFolder} and processed ${result.processed} images with the selected mirror settings.`,
                details: [
                    { label: 'Output Dataset', value: result.targetFolder },
                    { label: 'Processed Images', value: result.processed },
                    { label: 'Excluded Controls', value: (result.excludedControls || []).length ? result.excludedControls.join(', ') : 'None' }
                ]
            },
            merge: {
                title: 'Merge complete',
                summary: `Created ${result.targetName} from two source datasets without changing the originals.`,
                details: [
                    { label: 'Output Dataset', value: result.targetName },
                    { label: 'Primary Sets Copied', value: result.primaryCount },
                    { label: 'Secondary Sets Copied', value: result.secondaryCount },
                    { label: 'Total Sets', value: result.totalCount }
                ]
            }
        };

        const view = completedStates[toolName];
        renderToolProgress(toolName, {
            state: 'success',
            title: view.title,
            summary: view.summary,
            percent: 100,
            details: view.details
        });
        return;
    }

    const runningStates = {
        reshuffle: {
            title: 'Reshuffling dataset',
            summary: total ? `Renamed ${processed}/${total} image sets${job.currentItem ? ` • ${currentItem}` : ''}.` : 'Preparing filenames for reshuffle...',
            details: [
                { label: 'Dataset', value: context.folderPath },
                { label: 'Progress', value: total ? `${processed}/${total} sets` : 'Preparing' },
                { label: 'Files Renamed', value: job.metrics?.filesRenamed ?? 0 }
            ]
        },
        compress: {
            title: 'Compressing PNG files',
            summary: total ? `Compressed ${processed}/${total} PNG files${job.currentItem ? ` • ${currentItem}` : ''}.` : 'Collecting PNG files to compress...',
            details: [
                { label: 'Dataset', value: context.folderPath },
                { label: 'Progress', value: total ? `${processed}/${total} files` : 'Preparing' },
                { label: 'Original Size', value: job.metrics?.originalTotalSize ? `${(job.metrics.originalTotalSize / (1024 * 1024)).toFixed(2)} MB` : '0.00 MB' },
                { label: 'New Size', value: job.metrics?.newTotalSize ? `${(job.metrics.newTotalSize / (1024 * 1024)).toFixed(2)} MB` : '0.00 MB' }
            ]
        },
        fit: {
            title: 'Fitting controls to targets',
            summary: total ? `Processed ${processed}/${total} target image sets${job.currentItem ? ` • ${currentItem}` : ''}.` : 'Preparing target/control pairs...',
            details: [
                { label: 'Dataset', value: context.folderPath },
                { label: 'Progress', value: total ? `${processed}/${total} sets` : 'Preparing' },
                { label: 'Updated Controls', value: job.metrics?.updated ?? 0 }
            ]
        },
        duplicates: {
            title: 'Scanning duplicates',
            summary: total
                ? `Processed ${processed}/${total} fingerprint and comparison steps.`
                : 'Preparing image fingerprints...',
            details: [
                { label: 'Dataset', value: context.folderPath },
                { label: 'Progress', value: total ? `${processed}/${total}` : 'Preparing' },
                { label: 'Pairs Found', value: job.metrics?.pairsFound ?? 0 },
                { label: 'Threshold', value: context.threshold }
            ]
        },
        blur: {
            title: 'Generating blurred copy',
            summary: total ? `Blurred ${processed}/${total} images${job.currentItem ? ` • ${currentItem}` : ''}.` : 'Preparing dataset copy for blur...',
            details: [
                { label: 'Source Dataset', value: context.folderPath },
                { label: 'Blur Strength', value: `${Number(context.strength || 0).toFixed(1)} px` },
                { label: 'Progress', value: total ? `${processed}/${total} images` : 'Preparing' }
            ]
        },
        mirror: {
            title: 'Generating mirrored copy',
            summary: total ? `Mirrored ${processed}/${total} images${job.currentItem ? ` • ${currentItem}` : ''}.` : 'Preparing dataset copy for mirror...',
            details: [
                { label: 'Source Dataset', value: context.folderPath },
                { label: 'Directions', value: [context.horizontal ? 'Horizontal' : null, context.vertical ? 'Vertical' : null].filter(Boolean).join(' + ') },
                { label: 'Progress', value: total ? `${processed}/${total} images` : 'Preparing' },
                { label: 'Excluded Controls', value: (context.excludedControls || []).length ? context.excludedControls.join(', ') : 'None' }
            ]
        },
        merge: {
            title: 'Merging datasets',
            summary: total ? `Copied ${processed}/${total} synchronized image sets${job.currentItem ? ` • ${currentItem}` : ''}.` : 'Preparing merged dataset structure...',
            details: [
                { label: 'Primary', value: context.primaryFolder },
                { label: 'Secondary', value: context.secondaryFolder },
                { label: 'Progress', value: total ? `${processed}/${total} sets` : 'Preparing' },
                { label: 'Primary Copied', value: job.metrics?.primaryCount ?? 0 },
                { label: 'Secondary Copied', value: job.metrics?.secondaryCount ?? 0 },
                { label: 'Output Name', value: context.targetName }
            ]
        }
    };

    const view = runningStates[toolName];
    renderToolProgress(toolName, {
        state: 'running',
        title: view.title,
        summary: view.summary,
        percent,
        details: view.details,
        indeterminate: total === 0
    });
}

async function pollToolJob(toolName, jobId, { onComplete, onFinally } = {}) {
    try {
        const response = await fetch(`/api/tool-jobs/status/${encodeURIComponent(jobId)}`);
        const data = await response.json();

        if (!response.ok || data.error) {
            renderToolProgress(toolName, {
                state: 'error',
                title: `${toolName[0].toUpperCase()}${toolName.slice(1)} failed`,
                summary: data.error || 'Failed to read tool progress.',
                percent: 100
            });
            onFinally?.();
            return;
        }

        renderToolJobState(toolName, data);

        if (data.status === 'completed') {
            await onComplete?.(data);
            onFinally?.();
            return;
        }

        if (data.status === 'error') {
            onFinally?.();
            return;
        }

        clearToolPolling(toolName);
        await new Promise((resolve) => {
            toolPollTimers.set(toolName, setTimeout(resolve, 200));
        });
        toolPollTimers.delete(toolName);
        return await pollToolJob(toolName, jobId, { onComplete, onFinally });
    } catch (error) {
        console.error(`${toolName} progress polling failed:`, error);
        renderToolProgress(toolName, {
            state: 'error',
            title: `${toolName[0].toUpperCase()}${toolName.slice(1)} failed`,
            summary: 'Failed to update progress. Check console for details.',
            percent: 100
        });
        onFinally?.();
    }
}

async function startToolJob(toolName, payload, { onComplete, onFinally } = {}) {
    clearToolPolling(toolName);

    const response = await fetch('/api/tool-jobs/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool: toolName, payload })
    });

    const data = await response.json();
    if (!response.ok || data.error) {
        throw new Error(data.error || 'Failed to start background job.');
    }

    await pollToolJob(toolName, data.jobId, { onComplete, onFinally });
}

function updateToolsContext() {
    if (!toolsCurrentFolder || !toolsCurrentCount) return;

    toolsCurrentFolder.textContent = currentFolder || 'No dataset';
    toolsCurrentCount.textContent = `${images.length} image${images.length !== 1 ? 's' : ''}`;

    if (mergePrimaryName) {
        mergePrimaryName.textContent = currentFolder || 'No dataset selected';
    }

    updateMergeOptions();
}

function updateMergeOptions() {
    if (!mergeSecondarySelect) return;

    const previousValue = mergeSecondarySelect.value;
    mergeSecondarySelect.innerHTML = '<option value="">-- Select second dataset --</option>';

    allFolders
        .filter((folder) => folder.path !== currentFolder)
        .forEach((folder) => {
            const option = document.createElement('option');
            option.value = folder.path;
            option.textContent = folder.name;
            mergeSecondarySelect.appendChild(option);
        });

    if (previousValue && Array.from(mergeSecondarySelect.options).some((option) => option.value === previousValue)) {
        mergeSecondarySelect.value = previousValue;
    }
}

function setToolsView(viewName, options = {}) {
    currentToolsView = viewName;

    toolsNavButtons.forEach((button) => {
        const isActive = button.dataset.toolView === viewName;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', String(isActive));
    });

    toolsDetailPanels.forEach((panel) => {
        const isActive = panel.dataset.toolPanel === viewName;
        panel.classList.toggle('hidden', !isActive);
    });

    if (viewName === 'blur') {
        updateBlurPreview(Boolean(options.forceRefresh));
    }

    if (viewName === 'mirror') {
        updateMirrorPreview(Boolean(options.forceRefresh));
    }

    if (viewName === 'merge') {
        updateMergeOptions();
        if (!mergeTargetNameInput.value && currentFolder) {
            mergeTargetNameInput.value = `${currentFolder}_merged`;
        }
    }

    if (viewName === 'duplicates') {
        renderDuplicateReview();
    }
}

function pickBlurPreviewFilename(forceNew = false) {
    if (!images.length) {
        blurPreviewFilename = '';
        return '';
    }

    if (!blurPreviewFilename || forceNew || !images.includes(blurPreviewFilename)) {
        blurPreviewFilename = images[Math.floor(Math.random() * images.length)];
    }

    return blurPreviewFilename;
}

function pickMirrorPreviewFilename(forceNew = false) {
    if (!images.length) {
        mirrorPreviewFilename = '';
        return '';
    }

    if (!mirrorPreviewFilename || forceNew || !images.includes(mirrorPreviewFilename)) {
        mirrorPreviewFilename = images[Math.floor(Math.random() * images.length)];
    }

    return mirrorPreviewFilename;
}

function updateBlurPreview(forceNew = false) {
    const strength = Number(blurStrengthInput.value || 0);
    blurStrengthValue.textContent = `${strength.toFixed(1)} px`;

    const filename = pickBlurPreviewFilename(forceNew);
    if (!filename || !currentFolder) {
        blurPreviewName.textContent = 'No image selected';
        blurPreviewImage.classList.add('hidden');
        blurPreviewImage.removeAttribute('src');
        blurPreviewEmpty.classList.remove('hidden');
        blurApplyBtn.disabled = true;
        return;
    }

    blurPreviewName.textContent = filename;
    blurPreviewImage.src = `/api/dataset/blur-preview/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}&strength=${encodeURIComponent(strength)}&t=${Date.now()}`;
    blurPreviewImage.classList.remove('hidden');
    blurPreviewEmpty.classList.add('hidden');
    blurApplyBtn.disabled = false;
}

function updateMirrorPreview(forceNew = false) {
    const filename = pickMirrorPreviewFilename(forceNew);
    const horizontal = mirrorHorizontalInput.checked;
    const vertical = mirrorVerticalInput.checked;

    if (!filename || !currentFolder) {
        mirrorPreviewName.textContent = 'No image selected';
        mirrorPreviewImage.classList.add('hidden');
        mirrorPreviewImage.removeAttribute('src');
        mirrorPreviewEmpty.classList.remove('hidden');
        mirrorApplyBtn.disabled = true;
        return;
    }

    mirrorPreviewName.textContent = filename;
    mirrorPreviewImage.src = `/api/image/img/${encodeURIComponent(filename)}?folder=${encodeURIComponent(currentFolder)}${fileBuster(filename)}`;

    const transforms = [];
    if (horizontal) transforms.push('scaleX(-1)');
    if (vertical) transforms.push('scaleY(-1)');
    mirrorPreviewImage.style.transform = transforms.join(' ') || 'none';
    mirrorPreviewImage.classList.remove('hidden');
    mirrorPreviewEmpty.classList.add('hidden');
    mirrorApplyBtn.disabled = !horizontal && !vertical;
}

function getMirrorExcludedControls() {
    const excludedControls = [];
    if (mirrorExcludeControl1Input.checked) excludedControls.push('Control1');
    if (mirrorExcludeControl2Input.checked) excludedControls.push('Control2');
    if (mirrorExcludeControl3Input.checked) excludedControls.push('Control3');
    return excludedControls;
}

function resetDuplicateReview(message = 'Run a scan to review duplicate candidates.') {
    duplicateReviewState = {
        pairs: [],
        index: 0,
        threshold: Number(duplicateThresholdInput?.value || 8)
    };
    renderDuplicateReview(message);
}

function renderDuplicateReview(emptyMessage = 'Run a scan to review duplicate candidates.') {
    if (!duplicateReview || !duplicateEmpty || !duplicateStage) return;

    const { pairs, index } = duplicateReviewState;
    const currentPair = pairs[index];

    if (!currentPair) {
        duplicateReview.classList.add('is-empty');
        duplicateStage.classList.add('hidden');
        duplicateEmpty.classList.remove('hidden');
        duplicateEmpty.textContent = emptyMessage;
        return;
    }

    duplicateReview.classList.remove('is-empty');
    duplicateEmpty.classList.add('hidden');
    duplicateStage.classList.remove('hidden');

    duplicateProgress.textContent = `${index + 1} / ${pairs.length}`;
    duplicateDistance.textContent = `Distance ${currentPair.distance}`;

    duplicateLeftImg.src = `/api/image/img/${encodeURIComponent(currentPair.left.filename)}?folder=${encodeURIComponent(currentFolder)}${fileBuster(currentPair.left.filename)}`;
    duplicateRightImg.src = `/api/image/img/${encodeURIComponent(currentPair.right.filename)}?folder=${encodeURIComponent(currentFolder)}${fileBuster(currentPair.right.filename)}`;
    duplicateLeftName.textContent = currentPair.left.filename;
    duplicateRightName.textContent = currentPair.right.filename;
    duplicateLeftCaption.textContent = currentPair.left.caption || 'No caption';
    duplicateRightCaption.textContent = currentPair.right.caption || 'No caption';
}

async function scanDuplicateImages() {
    if (!currentFolder) {
        showToolValidation('duplicates', 'Select a dataset before searching duplicates.');
        return;
    }

    if (images.length < 2) {
        showToolValidation('duplicates', 'This dataset needs at least two images to search duplicates.');
        return;
    }

    const threshold = Number(duplicateThresholdInput.value || 8);
    duplicateThresholdValue.textContent = String(threshold);

    try {
        setActionButtonBusy(duplicateScanBtn, 'Scanning...', true);
        renderToolProgress('duplicates', {
            state: 'running',
            title: 'Scanning duplicates',
            summary: `Comparing ${images.length} images against each other.`,
            percent: 35,
            details: [
                { label: 'Images', value: images.length },
                { label: 'Threshold', value: threshold }
            ],
            indeterminate: true
        });

        toolJobContexts.set('duplicates', { folderPath: currentFolder, threshold });
        let data = null;
        await startToolJob('duplicates', { folderPath: currentFolder, threshold }, {
            onComplete: async (job) => {
                data = job.result;
            }
        });
        if (!data) throw new Error('Duplicate scan did not return a result.');

        duplicateReviewState = {
            pairs: data.pairs || [],
            index: 0,
            threshold: data.threshold
        };

        if (!duplicateReviewState.pairs.length) {
            renderDuplicateReview('No duplicate pairs found in this dataset.');
        } else {
            renderDuplicateReview();
        }

        renderToolProgress('duplicates', {
            state: 'success',
            title: duplicateReviewState.pairs.length ? 'Duplicate pairs found' : 'No duplicates found',
            summary: duplicateReviewState.pairs.length
                ? `Found ${duplicateReviewState.pairs.length} duplicate pair${duplicateReviewState.pairs.length !== 1 ? 's' : ''}. Click the image you want to keep.`
                : `No pairs matched at threshold ${data.threshold}.`,
            percent: 100,
            details: [
                { label: 'Images Scanned', value: data.imageCount },
                { label: 'Pairs', value: duplicateReviewState.pairs.length },
                { label: 'Threshold', value: data.threshold }
            ]
        });
    } catch (error) {
        console.error('Duplicate scan failed:', error);
        resetDuplicateReview('Duplicate scan failed. Check the status card for details.');
        renderToolProgress('duplicates', {
            state: 'error',
            title: 'Duplicate scan failed',
            summary: error.message || 'Failed to scan duplicate images.',
            percent: 100,
            details: [{ label: 'Dataset', value: currentFolder }]
        });
    } finally {
        setActionButtonBusy(duplicateScanBtn, 'Scanning...', false);
    }
}

function advanceDuplicatePair() {
    if (!duplicateReviewState.pairs.length) {
        renderDuplicateReview('Duplicate review is complete.');
        return;
    }

    duplicateReviewState.index += 1;
    if (duplicateReviewState.index >= duplicateReviewState.pairs.length) {
        duplicateReviewState.pairs = [];
        duplicateReviewState.index = 0;
        renderDuplicateReview('Duplicate review is complete.');
        return;
    }

    renderDuplicateReview();
}

async function keepDuplicateSide(side) {
    const { pairs, index } = duplicateReviewState;
    const currentPair = pairs[index];
    if (!currentPair) return;

    const keepFilename = side === 'left' ? currentPair.left.filename : currentPair.right.filename;
    const deleteFilename = side === 'left' ? currentPair.right.filename : currentPair.left.filename;

    duplicateLeftCard.disabled = true;
    duplicateRightCard.disabled = true;
    duplicateSkipBtn.disabled = true;

    try {
        const response = await fetch(
            `/api/delete/${encodeURIComponent(deleteFilename)}?folder=${encodeURIComponent(currentFolder)}`,
            { method: 'DELETE' }
        );
        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || 'Delete failed.');
        }

        images = images.filter((filename) => filename !== deleteFilename);
        if (currentIndex >= images.length) {
            currentIndex = Math.max(0, images.length - 1);
        }
        markDirty(keepFilename);

        duplicateReviewState.pairs = duplicateReviewState.pairs
            .filter((pair, pairIndex) => (
                pairIndex !== index
                && pair.left.filename !== deleteFilename
                && pair.right.filename !== deleteFilename
            ));
        duplicateReviewState.index = Math.min(index, Math.max(0, duplicateReviewState.pairs.length - 1));

        renderImageGrid();
        updateImageCount();
        updateToolsContext();

        if (!duplicateReviewState.pairs.length) {
            renderDuplicateReview('Duplicate review is complete.');
        } else {
            renderDuplicateReview();
        }

        renderToolProgress('duplicates', {
            state: 'success',
            title: 'Image set deleted',
            summary: `Kept ${keepFilename} and deleted ${deleteFilename} with its related files.`,
            percent: 100,
            details: [
                { label: 'Kept', value: keepFilename },
                { label: 'Deleted', value: deleteFilename },
                { label: 'Remaining Pairs', value: duplicateReviewState.pairs.length }
            ]
        });
    } catch (error) {
        console.error('Duplicate delete failed:', error);
        renderToolProgress('duplicates', {
            state: 'error',
            title: 'Delete failed',
            summary: error.message || 'Failed to delete duplicate image set.',
            percent: 100,
            details: [
                { label: 'Keep', value: keepFilename },
                { label: 'Delete', value: deleteFilename }
            ]
        });
    } finally {
        duplicateLeftCard.disabled = false;
        duplicateRightCard.disabled = false;
        duplicateSkipBtn.disabled = false;
    }
}

async function openToolsModal() {
    if (!currentFolder) {
        alert('Please select a dataset folder first.');
        return;
    }
    if (modal.classList.contains('active') && !await checkUnsavedWork()) return;

    updateToolsContext();
    setToolsView(currentToolsView, { forceRefresh: true });
    toolsModal.classList.add('active');
    document.body.style.overflow = 'hidden';
    toolsCloseBtn.focus();
}

function closeToolsModal() {
    toolsModal.classList.remove('active');
    document.body.style.overflow = '';
}

function setImportStatus(message, type = 'info') {
    importStatus.textContent = message;
    importStatus.dataset.type = type;
}

function clearImportPolling() {
    if (importPollTimer) {
        clearTimeout(importPollTimer);
        importPollTimer = null;
    }
}

function releaseImportButton() {
    if (!activeImportButton) return;
    activeImportButton.disabled = false;
    activeImportButton.textContent = 'Import Dataset';
    activeImportButton = null;
}

function renderActiveImportCardProgress(job) {
    if (!activeImportCard) return;

    let progressBlock = activeImportCard.querySelector('.import-card-progress');
    if (!job) {
        if (progressBlock) progressBlock.remove();
        activeImportCard.classList.remove('import-card-running', 'import-card-completed', 'import-card-error');
        return;
    }

    if (!progressBlock) {
        progressBlock = document.createElement('div');
        progressBlock.className = 'import-card-progress';
        activeImportCard.appendChild(progressBlock);
    }

    activeImportCard.classList.toggle('import-card-running', job.status !== 'completed' && job.status !== 'error');
    activeImportCard.classList.toggle('import-card-completed', job.status === 'completed');
    activeImportCard.classList.toggle('import-card-error', job.status === 'error');

    const percent = Number(job.progressPercent || 0);
    const summary = job.status === 'completed'
        ? `Imported ${job.copiedFiles}/${job.totalFiles} files successfully.`
        : job.status === 'error'
            ? (job.error || `Import failed after ${job.copiedFiles}/${job.totalFiles} files.`)
            : `Copying ${job.copiedFiles}/${job.totalFiles} files${job.currentFolder ? ` • ${job.currentFolder}` : ''}`;

    progressBlock.innerHTML = `
        <div class="import-card-progress-head">
            <strong>${escHtml(job.targetName)}</strong>
            <span>${percent.toFixed(1)}%</span>
        </div>
        <div class="import-card-progress-bar">
            <div class="import-card-progress-fill" style="width: ${Math.max(0, Math.min(100, percent))}%"></div>
        </div>
        <p class="import-card-progress-text">${escHtml(summary)}</p>
    `;
}

function renderImportProgress(job) {
    if (!job) {
        importProgressCard.classList.add('hidden');
        importProgressFill.style.width = '0%';
        importProgressFolders.innerHTML = '';
        renderActiveImportCardProgress(null);
        return;
    }

    importProgressCard.classList.remove('hidden');
    importProgressTitle.textContent = job.title || `${job.sourceName} -> ${job.targetName}`;
    importProgressSummary.textContent = job.summary || (
        job.status === 'completed'
            ? `Import completed. Copied ${job.copiedFiles} of ${job.totalFiles} files.`
            : job.status === 'error'
                ? `Import failed after copying ${job.copiedFiles} of ${job.totalFiles} files.`
                : `Copying ${job.copiedFiles} of ${job.totalFiles} files${job.currentFolder ? ` • ${job.currentFolder}` : ''}`
    );

    const percent = Number(job.progressPercent || 0);
    importProgressPercent.textContent = `${percent.toFixed(1)}%`;
    importProgressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;

    const folderProgress = job.folderProgress || {};
    importProgressFolders.innerHTML = Object.entries(folderProgress)
        .map(([folderName, folderJob]) => {
            const total = Number(folderJob.total) || 0;
            const copied = Number(folderJob.copied) || 0;
            const width = total ? Math.min(100, (copied / total) * 100) : 100;
            return `
                <div class="import-progress-folder">
                    <strong>${escHtml(folderName)}</strong>
                    <div class="import-progress-folder-bar">
                        <div class="import-progress-folder-fill" style="width: ${width}%"></div>
                    </div>
                    <span>${copied}/${total}</span>
                </div>
            `;
        })
        .join('');

    renderActiveImportCardProgress(job);
}

async function pollImportJob(jobId) {
    try {
        const response = await fetch(`/api/import/status/${encodeURIComponent(jobId)}`);
        const data = await response.json();

        if (!response.ok || data.error) {
            activeImportJobId = null;
            releaseImportButton();
            setImportStatus(data.error || 'Failed to read import progress.', 'error');
            return;
        }

        renderImportProgress(data);

        if (data.status === 'completed') {
            activeImportJobId = null;
            if (activeImportButton) {
                activeImportButton.disabled = false;
                activeImportButton.textContent = 'Imported';
            }
            await loadFolders();
            setImportStatus(`Imported ${data.sourceName} as ${data.targetName}. Copied ${data.copiedFiles}/${data.totalFiles} files.`, 'success');
            return;
        }

        if (data.status === 'error') {
            activeImportJobId = null;
            if (activeImportButton) {
                activeImportButton.disabled = false;
                activeImportButton.textContent = 'Retry Import';
            }
            setImportStatus(data.error || 'Import failed.', 'error');
            return;
        }

        importPollTimer = setTimeout(() => pollImportJob(jobId), 350);
    } catch (error) {
        console.error('Import progress polling failed:', error);
        activeImportJobId = null;
        releaseImportButton();
        setImportStatus('Failed to update import progress. Check console for details.', 'error');
    }
}

function renderImportResults() {
    if (!importScanResults.length) {
        importResults.classList.add('empty');
        importResults.innerHTML = '<div class="import-results-empty">No importable trainer datasets found at this path.</div>';
        return;
    }

    importResults.classList.remove('empty');
    importResults.innerHTML = '';

    importScanResults.forEach((dataset) => {
        const card = document.createElement('div');
        card.className = 'import-dataset-card';

        const folderRows = ['img', 'Control1', 'Control2', 'Control3']
            .filter((folderName) => dataset.folders[folderName])
            .map((folderName) => {
                const folder = dataset.folders[folderName];
                return `
                    <div class="import-folder-row">
                        <strong>${escHtml(folderName)}</strong>
                        <span>${escHtml(folder.imageCount)} images</span>
                        <code>${escHtml(folder.name)}</code>
                    </div>
                `;
            })
            .join('');

        card.innerHTML = `
            <div class="import-dataset-head">
                <div class="import-dataset-title">
                    <strong>${escHtml(dataset.sourceName)}</strong>
                    <div class="import-dataset-meta">Grouped as one ${escHtml(dataset.pairStyle)} dataset from trainer export folders</div>
                </div>
                <div class="import-dataset-badges">
                    <span class="import-dataset-badge">${escHtml(dataset.imageCount)} targets</span>
                    <span class="import-dataset-badge">${escHtml(dataset.controlCount)} controls</span>
                </div>
            </div>
            <div class="import-folder-list">${folderRows}</div>
            <div class="import-target-row">
                <input class="tools-text-input import-target-input" type="text" value="${escHtml(dataset.sourceName)}" aria-label="Target dataset name">
                <button class="action-btn export-btn import-dataset-btn">Import Dataset</button>
            </div>
        `;

        const targetInput = card.querySelector('.import-target-input');
        const importBtn = card.querySelector('.import-dataset-btn');
        importBtn.addEventListener('click', () => importTrainerDataset(dataset.sourceName, targetInput.value.trim(), importBtn));
        importResults.appendChild(card);
    });
}

async function scanImportDatasets() {
    const path = importPathInput.value.trim();
    if (!path) {
        setImportStatus('Enter a trainer export path first.', 'error');
        importPathInput.focus();
        return;
    }

    try {
        persistImportPath(path);
        clearPersistedImportResults(path);
        importScanBtn.disabled = true;
        importScanBtn.textContent = 'Scanning...';
        setImportStatus('Scanning trainer export folders...', 'info');
        renderImportProgress({
            title: 'Scanning trainer export folders',
            summary: `Inspecting ${path} for grouped trainer datasets. Results will appear below when the scan completes.`,
            sourceName: 'scan',
            targetName: path,
            copiedFiles: 0,
            totalFiles: 1,
            progressPercent: 24,
            currentFolder: null,
            folderProgress: {},
            status: 'scanning'
        });

        const response = await fetch('/api/import/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });

        const data = await response.json();
        if (data.success) {
            importScanResults = data.datasets || [];
            persistImportResults(path, importScanResults);
            renderImportResults();
            renderImportProgress({
                title: 'Trainer scan complete',
                summary: importScanResults.length
                    ? `Found ${importScanResults.length} grouped trainer dataset${importScanResults.length !== 1 ? 's' : ''} at ${path}.`
                    : `Scan finished for ${path}, but no grouped trainer datasets were detected.`,
                sourceName: 'scan',
                targetName: path,
                copiedFiles: importScanResults.length,
                totalFiles: Math.max(importScanResults.length, 1),
                progressPercent: 100,
                currentFolder: null,
                folderProgress: {},
                status: 'completed'
            });
            setImportStatus(
                importScanResults.length
                    ? `Found ${importScanResults.length} grouped trainer dataset${importScanResults.length !== 1 ? 's' : ''}.`
                    : 'Scan completed, but no grouped trainer datasets were found.',
                importScanResults.length ? 'success' : 'warning'
            );
        } else {
            importScanResults = [];
            clearPersistedImportResults(path);
            renderImportResults();
            renderImportProgress({
                title: 'Trainer scan failed',
                summary: data.error || 'Failed to scan the trainer export folder.',
                sourceName: 'scan',
                targetName: path,
                copiedFiles: 0,
                totalFiles: 1,
                progressPercent: 100,
                currentFolder: null,
                folderProgress: {},
                status: 'error'
            });
            setImportStatus(data.error || 'Failed to scan trainer folder.', 'error');
        }
    } catch (error) {
        console.error('Import scan failed:', error);
        importScanResults = [];
        renderImportResults();
        renderImportProgress({
            title: 'Trainer scan failed',
            summary: 'The request failed before the server returned a scan result.',
            sourceName: 'scan',
            targetName: path,
            copiedFiles: 0,
            totalFiles: 1,
            progressPercent: 100,
            currentFolder: null,
            folderProgress: {},
            status: 'error'
        });
        setImportStatus('Failed to scan trainer folder. Check console for details.', 'error');
    } finally {
        importScanBtn.disabled = false;
        importScanBtn.textContent = importScanBtn.dataset.defaultLabel || 'Scan';
    }
}

async function importTrainerDataset(sourceName, targetName, button) {
    const basePath = importPathInput.value.trim();
    if (!basePath) {
        setImportStatus('Trainer export path is empty.', 'error');
        return;
    }

    if (!targetName) {
        setImportStatus('Target dataset name is required.', 'error');
        return;
    }

    if (activeImportJobId) {
        setImportStatus('Wait for the current import to finish before starting another one.', 'warning');
        return;
    }

    try {
        clearImportPolling();
        activeImportButton = button;
        activeImportCard = button.closest('.import-dataset-card');
        button.disabled = true;
        button.textContent = 'Starting...';
        setImportStatus(`Starting import for ${sourceName} as ${targetName}...`, 'info');
        renderImportProgress({
            sourceName,
            targetName,
            copiedFiles: 0,
            totalFiles: 0,
            progressPercent: 0,
            currentFolder: null,
            folderProgress: {},
            status: 'queued'
        });
        activeImportCard?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

        const response = await fetch('/api/import/dataset/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                basePath,
                sourceName,
                targetName
            })
        });

        const data = await response.json();
        if (data.success) {
            activeImportJobId = data.jobId;
            button.textContent = 'Importing...';
            renderImportProgress({
                sourceName: data.sourceName,
                targetName: data.targetName,
                copiedFiles: 0,
                totalFiles: data.totalFiles,
                progressPercent: 0,
                currentFolder: null,
                folderProgress: data.folderProgress || {},
                status: 'queued'
            });
            setImportStatus(`Import job started for ${data.targetName}.`, 'info');
            pollImportJob(data.jobId);
        } else {
            activeImportJobId = null;
            releaseImportButton();
            renderImportProgress(null);
            if (activeImportButton) activeImportButton.textContent = 'Retry Import';
            setImportStatus(data.error || 'Failed to import dataset.', 'error');
        }
    } catch (error) {
        console.error('Import dataset failed:', error);
        activeImportJobId = null;
        releaseImportButton();
        renderImportProgress(null);
        setImportStatus('Failed to import dataset. Check console for details.', 'error');
    }
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
    closeBtn.focus();
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

function transferActionMarkup(mode) {
    if (mode === 'copy') {
        return `
            <svg aria-hidden="true" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy (↑)
        `;
    }
    return `
        <svg aria-hidden="true" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 19V5M5 12l7-7 7 7"></path>
        </svg>
        Transfer (↑)
    `;
}

function updateTransferMode(value) {
    transferMode = value === 'copy' ? 'copy' : 'transfer';
    const actionLabel = transferMode === 'copy' ? 'Copy' : 'Transfer';
    transferBtn.innerHTML = transferActionMarkup(transferMode);
    transferBtn.title = `${actionLabel} to target dataset (↑)`;
    transferModeHelp.textContent = transferMode === 'copy'
        ? 'Copy this triplet to the selected dataset and keep the original.'
        : 'Move this triplet to the selected dataset.';
}

// Handle target dataset selection
function onTargetDatasetChange(value) {
    targetFolder = value;
    transferBtn.style.display = value ? 'flex' : 'none';
}

// Close preview modal
async function closePreview() {
    if (modal.classList.contains('active') && !await checkUnsavedWork()) return false;
    captionRequestController?.abort();
    modal.classList.remove('active');
    document.body.style.overflow = '';
    overlayActive = false;
    previewImg.classList.remove('overlay-active');
    previewControl.classList.remove('active');
    toggleBtn.classList.remove('active');
    saveAppState();
    return true;
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
                mainEditor.history = [];
                mainEditor.hasChanges = false;
                mainEditor.setupCanvas();
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
                comparisonEditor.history = [];
                comparisonEditor.hasChanges = false;
                comparisonEditor.setupCanvas();
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
        if (mainEditor) ImageEditor.activeEditor = mainEditor;
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
async function showControlFullPreview(controlName) {
    if (!await checkUnsavedEditors()) return;
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
        if (!await checkUnsavedWork()) return;
        currentIndex--;
        updatePreview();
        saveAppState();
    }
}

// Navigate to next image
async function showNext() {
    if (currentIndex < images.length - 1) {
        if (!await checkUnsavedWork()) return;
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

// Transfer or copy the current image set to the target dataset
async function transferCurrentImage() {
    if (images.length === 0 || !targetFolder) return;
    if (!await checkUnsavedWork()) return;

    const filename = images[currentIndex];
    const activeMode = transferMode;
    const actionLabel = activeMode === 'copy' ? 'Copy' : 'Transfer';

    try {
        transferBtn.disabled = true;
        transferBtn.textContent = activeMode === 'copy' ? 'Copying...' : 'Transferring...';

        // Build request body with optional linked folder
        const requestBody = { targetFolder: targetFolder, operation: activeMode };
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
            if (activeMode === 'transfer') {
                // Remove from images array because the source files were moved.
                images.splice(currentIndex, 1);

                const gridItem = imageGrid.querySelector(`.image-item[data-index="${currentIndex}"]`);
                if (gridItem) {
                    gridItem.remove();
                    const items = imageGrid.querySelectorAll('.image-item');
                    items.forEach((item, idx) => item.dataset.index = idx);
                }

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
            }

            console.log(`${actionLabel} complete:`, activeMode === 'copy' ? data.copied : data.transferred);
        } else {
            alert(`Failed to ${activeMode}: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error(`Failed to ${activeMode} image:`, error);
        alert(`Failed to ${activeMode} image. Check console for details.`);
    } finally {
        transferBtn.disabled = false;
        updateTransferMode(transferMode);
    }
}

// Duplicate current image set
async function duplicateCurrentImage() {
    if (images.length === 0) return;
    if (!await checkUnsavedWork()) return;

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
            captionDirty = false;
            [mainEditor, comparisonEditor].forEach((editor) => {
                if (!editor) return;
                editor.hasChanges = false;
                if (editor.history.length) editor.history = [editor.history[editor.history.length - 1]];
            });
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
        showToolValidation('reshuffle', 'Select a dataset before running reshuffle.');
        return;
    }

    try {
        setActionButtonBusy(reshuffleBtn, 'Shuffling...', true);
        toolJobContexts.set('reshuffle', { folderPath: currentFolder });
        await startToolJob('reshuffle', { folderPath: currentFolder }, {
            onComplete: async () => {
                bumpFolderBuster();
                await loadImages(currentFolder);
            },
            onFinally: () => {
                setActionButtonBusy(reshuffleBtn, 'Shuffling...', false);
            }
        });
        return;
    } catch (error) {
        console.error('Reshuffle failed:', error);
        renderToolProgress('reshuffle', {
            state: 'error',
            title: 'Reshuffle failed',
            summary: error.message || 'The request failed before the server started the job.',
            percent: 100,
            details: [{ label: 'Dataset', value: currentFolder }]
        });
    } finally {
        if (reshuffleBtn.disabled) {
            setActionButtonBusy(reshuffleBtn, 'Shuffling...', false);
        }
    }
}

// Compress dataset
async function fitDataset() {
    if (!currentFolder) {
        showToolValidation('fit', 'Select a dataset before fitting controls to targets.');
        return;
    }

    setActionButtonBusy(fitBtn, 'Processing...', true);

    try {
        toolJobContexts.set('fit', { folderPath: currentFolder });
        await startToolJob('fit', { folderPath: currentFolder }, {
            onComplete: async () => {
                bumpFolderBuster();
                if (modal.classList.contains('active')) {
                    updatePreview();
                }
                await loadImages(currentFolder);
            },
            onFinally: () => {
                setActionButtonBusy(fitBtn, 'Processing...', false);
            }
        });
        return;
    } catch (error) {
        console.error('Fit dataset error:', error);
        renderToolProgress('fit', {
            state: 'error',
            title: 'Fit failed',
            summary: error.message || 'The request failed before the server started the job.',
            percent: 100,
            details: [{ label: 'Dataset', value: currentFolder }]
        });
    } finally {
        if (fitBtn.disabled) {
            setActionButtonBusy(fitBtn, 'Processing...', false);
        }
    }
}

async function compressDataset() {
    if (!currentFolder) {
        showToolValidation('compress', 'Select a dataset before running compression.');
        return;
    }

    try {
        setActionButtonBusy(compressBtn, 'Compressing...', true);
        toolJobContexts.set('compress', { folderPath: currentFolder });
        await startToolJob('compress', { folderPath: currentFolder }, {
            onComplete: async () => {
                bumpFolderBuster();
                await loadImages(currentFolder);
            },
            onFinally: () => {
                setActionButtonBusy(compressBtn, 'Compressing...', false);
            }
        });
        return;
    } catch (error) {
        console.error('Compress failed:', error);
        renderToolProgress('compress', {
            state: 'error',
            title: 'Compression failed',
            summary: error.message || 'The request failed before the server started the job.',
            percent: 100,
            details: [{ label: 'Dataset', value: currentFolder }]
        });
    } finally {
        if (compressBtn.disabled) {
            setActionButtonBusy(compressBtn, 'Compressing...', false);
        }
    }
}

async function createBlurredDatasetCopy() {
    if (!currentFolder) {
        showToolValidation('blur', 'Select a dataset before generating a blurred copy.');
        return;
    }

    if (!images.length) {
        showToolValidation('blur', 'This dataset has no target images to blur.');
        return;
    }

    const strength = Number(blurStrengthInput.value || 0);

    try {
        blurApplyBtn.disabled = true;
        blurRerollBtn.disabled = true;
        setActionButtonLabel(blurApplyBtn, 'Creating...');
        toolJobContexts.set('blur', { folderPath: currentFolder, strength });
        await startToolJob('blur', { folderPath: currentFolder, strength }, {
            onComplete: async () => {
                const selectedFolder = currentFolder;
                await loadFolders();
                folderSelect.value = selectedFolder;
            },
            onFinally: () => {
                blurApplyBtn.disabled = false;
                blurRerollBtn.disabled = false;
                setActionButtonLabel(blurApplyBtn, blurApplyBtn.dataset.defaultLabel || 'Create Blurred Copy');
            }
        });
        return;
    } catch (error) {
        console.error('Blur dataset failed:', error);
        renderToolProgress('blur', {
            state: 'error',
            title: 'Blur generation failed',
            summary: error.message || 'The request failed before the server started the job.',
            percent: 100,
            details: [{ label: 'Source Dataset', value: currentFolder }]
        });
    } finally {
        if (blurApplyBtn.disabled) {
            blurApplyBtn.disabled = false;
            blurRerollBtn.disabled = false;
            setActionButtonLabel(blurApplyBtn, blurApplyBtn.dataset.defaultLabel || 'Create Blurred Copy');
        }
    }
}

async function createMirroredDatasetCopy() {
    if (!currentFolder) {
        showToolValidation('mirror', 'Select a dataset before generating a mirrored copy.');
        return;
    }

    if (!images.length) {
        showToolValidation('mirror', 'This dataset has no target images to mirror.');
        return;
    }

    const horizontal = mirrorHorizontalInput.checked;
    const vertical = mirrorVerticalInput.checked;
    const excludedControls = getMirrorExcludedControls();
    if (!horizontal && !vertical) {
        showToolValidation('mirror', 'Select at least one mirror direction.');
        return;
    }

    try {
        mirrorApplyBtn.disabled = true;
        mirrorRerollBtn.disabled = true;
        setActionButtonLabel(mirrorApplyBtn, 'Creating...');
        toolJobContexts.set('mirror', { folderPath: currentFolder, horizontal, vertical, excludedControls });
        await startToolJob('mirror', { folderPath: currentFolder, horizontal, vertical, excludedControls }, {
            onComplete: async () => {
                const selectedFolder = currentFolder;
                await loadFolders();
                folderSelect.value = selectedFolder;
            },
            onFinally: () => {
                mirrorApplyBtn.disabled = false;
                mirrorRerollBtn.disabled = false;
                setActionButtonLabel(mirrorApplyBtn, mirrorApplyBtn.dataset.defaultLabel || 'Create Mirrored Copy');
                updateMirrorPreview(false);
            }
        });
        return;
    } catch (error) {
        console.error('Mirror dataset failed:', error);
        renderToolProgress('mirror', {
            state: 'error',
            title: 'Mirror generation failed',
            summary: error.message || 'The request failed before the server started the job.',
            percent: 100,
            details: [{ label: 'Source Dataset', value: currentFolder }]
        });
    } finally {
        if (mirrorApplyBtn.disabled) {
            mirrorApplyBtn.disabled = false;
            mirrorRerollBtn.disabled = false;
            setActionButtonLabel(mirrorApplyBtn, mirrorApplyBtn.dataset.defaultLabel || 'Create Mirrored Copy');
            updateMirrorPreview(false);
        }
    }
}

async function createMergedDataset() {
    if (!currentFolder) {
        showToolValidation('merge', 'Select the primary dataset before running merge.');
        return;
    }

    const secondaryFolder = mergeSecondarySelect.value;
    const targetName = mergeTargetNameInput.value.trim();

    if (!secondaryFolder) {
        showToolValidation('merge', 'Choose the second dataset to merge with the active one.');
        return;
    }

    if (!targetName) {
        showToolValidation('merge', 'Enter the name for the new merged dataset.');
        mergeTargetNameInput.focus();
        return;
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(targetName)) {
        showToolValidation('merge', 'The new dataset name can only contain letters, numbers, underscores, and hyphens.');
        mergeTargetNameInput.focus();
        return;
    }

    try {
        setActionButtonBusy(mergeApplyBtn, 'Merging...', true);
        toolJobContexts.set('merge', { primaryFolder: currentFolder, secondaryFolder, targetName });
        await startToolJob('merge', { primaryFolder: currentFolder, secondaryFolder, targetName }, {
            onComplete: async () => {
                await loadFolders();
            },
            onFinally: () => {
                setActionButtonBusy(mergeApplyBtn, 'Merging...', false);
            }
        });
        return;
    } catch (error) {
        console.error('Merge dataset failed:', error);
        renderToolProgress('merge', {
            state: 'error',
            title: 'Merge failed',
            summary: error.message || 'The request failed before the server started the job.',
            percent: 100,
            details: [
                { label: 'Primary', value: currentFolder },
                { label: 'Secondary', value: secondaryFolder },
                { label: 'Output Name', value: targetName }
            ]
        });
    } finally {
        if (mergeApplyBtn.disabled) {
            setActionButtonBusy(mergeApplyBtn, 'Merging...', false);
        }
    }
}

// Export to AI-Toolkit format
async function exportDataset() {
    if (!currentFolder) {
        showExportModalError('Select a dataset folder first.');
        return;
    }

    const exportPath = exportPathInput.value.trim();
    if (!exportPath) {
        showExportModalError('Export path is required.');
        exportPathInput.focus();
        return;
    }

    try {
        persistExportPath(exportPath);
        exportModalError.textContent = '';
        exportModalError.classList.add('hidden');
        exportBtn.disabled = true;
        exportSubmitBtn.disabled = true;
        setActionButtonLabel(exportBtn, 'Exporting...');
        setActionButtonLabel(exportSubmitBtn, 'Exporting...');
        renderExportProgress({
            state: 'running',
            title: 'Exporting dataset',
            summary: `Creating trainer export folders for ${currentFolder} inside ${exportPath}.`,
            percent: 22,
            indeterminate: true,
            details: [
                { label: 'Dataset', value: currentFolder },
                { label: 'Export Root', value: exportPath }
            ]
        });

        const response = await fetch(`/api/export?folder=${encodeURIComponent(currentFolder)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exportPath })
        });

        const data = await response.json();

        if (data.success) {
            renderExportProgress({
                state: 'success',
                title: 'Export complete',
                summary: `Finished exporting ${currentFolder}. The trainer folders are ready at the selected root path.`,
                percent: 100,
                details: [
                    { label: 'Export Path', value: data.exportPath },
                    ...Object.entries(data.exported).map(([folder, info]) => ({
                        label: folder,
                        value: `${info.files} files`
                    }))
                ]
            });
        } else {
            renderExportProgress({
                state: 'error',
                title: 'Export failed',
                summary: data.error || 'The server could not export this dataset.',
                percent: 100,
                details: [
                    { label: 'Dataset', value: currentFolder },
                    { label: 'Export Root', value: exportPath }
                ]
            });
        }
    } catch (error) {
        console.error('Export failed:', error);
        renderExportProgress({
            state: 'error',
            title: 'Export failed',
            summary: 'The request failed before the server returned an export result.',
            percent: 100,
            details: [
                { label: 'Dataset', value: currentFolder },
                { label: 'Export Root', value: exportPath }
            ]
        });
    } finally {
        exportBtn.disabled = false;
        exportSubmitBtn.disabled = false;
        setActionButtonLabel(exportBtn, 'Export');
        setActionButtonLabel(exportSubmitBtn, 'Export Dataset');
    }
}

// Load caption for current image
// Track unsaved caption changes
let captionDirty = false;

function autoresizeCaption() {
    captionText.style.height = 'auto';
    captionText.style.height = captionText.scrollHeight + 'px';
}

// Save pending work without interrupting navigation with modal prompts.
async function checkUnsavedCaption() {
    if (!captionDirty) return true;
    return await saveCurrentCaption();
}

async function checkUnsavedEditors(editors = [mainEditor, comparisonEditor]) {
    const dirtyEditors = editors.filter((editor) => editor?.hasChanges);
    if (!dirtyEditors.length) return true;
    for (const editor of dirtyEditors) {
        if (!await editor.save()) return false;
    }
    return true;
}

async function checkUnsavedWork() {
    if (!await checkUnsavedCaption()) return false;
    return await checkUnsavedEditors();
}

async function loadCaption(filename) {
    const requestedFolder = currentFolder;
    const requestedKey = `${requestedFolder}\u0000${filename}`;
    if (captionDirty && captionLoadedKey === requestedKey) return;
    captionRequestController?.abort();
    const controller = new AbortController();
    captionRequestController = controller;
    const token = ++captionLoadToken;
    try {
        captionText.value = 'Loading...';
        captionDirty = false;
        autoresizeCaption();
        const response = await fetch(`/api/caption/${encodeURIComponent(filename)}?folder=${encodeURIComponent(requestedFolder)}`, {
            signal: controller.signal
        });
        const data = await response.json();

        if (
            token !== captionLoadToken ||
            requestedFolder !== currentFolder ||
            images[currentIndex] !== filename ||
            captionDirty
        ) return;

        if (data.error) {
            console.error('Error loading caption:', data.error);
            captionText.value = '';
        } else {
            captionText.value = data.caption || '';
        }
        captionLoadedKey = requestedKey;
        captionDirty = false;
        autoresizeCaption();
    } catch (error) {
        if (error.name === 'AbortError') return;
        console.error('Failed to load caption:', error);
        if (token === captionLoadToken && requestedFolder === currentFolder && images[currentIndex] === filename) {
            captionText.value = '';
            captionDirty = false;
        }
    }
}

// Save current caption
async function saveCurrentCaption() {
    if (images.length === 0) return false;

    const filename = images[currentIndex];
    const folder = currentFolder;
    const caption = captionText.value;

    try {
        saveCaptionBtn.disabled = true;
        const response = await fetch(`/api/caption/${encodeURIComponent(filename)}?folder=${encodeURIComponent(folder)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ caption })
        });

        const data = await response.json();

        if (data.success) {
            const savedCurrentCaption = (
                images[currentIndex] === filename &&
                currentFolder === folder &&
                captionText.value === caption
            );
            if (savedCurrentCaption) {
                captionDirty = false;
                captionLoadedKey = `${folder}\u0000${filename}`;
            }
            const originalBackground = saveCaptionBtn.style.background;
            saveCaptionBtn.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
            setTimeout(() => { saveCaptionBtn.style.background = originalBackground; }, 1000);
            return savedCurrentCaption;
        } else {
            alert(`Failed to save caption: ${data.error}`);
            return false;
        }
    } catch (error) {
        console.error('Failed to save caption:', error);
        alert('Failed to save caption. Check console for details.');
        return false;
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
    folderSelect.addEventListener('change', async (e) => {
        if (modal.classList.contains('active') && !await checkUnsavedWork()) {
            folderSelect.value = currentFolder;
            return;
        }
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
    imageGrid.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const item = e.target.closest('.image-item');
        if (!item || item.dataset.index === undefined) return;
        e.preventDefault();
        const idx = parseInt(item.dataset.index);
        if (stitchMode) toggleStitchSelection(idx);
        else openPreview(idx);
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
    toolsBtn.addEventListener('click', openToolsModal);
    toolsCloseBtn.addEventListener('click', closeToolsModal);
    refreshFoldersBtn.addEventListener('click', refreshDatasetsView);
    renameDatasetBtn.addEventListener('click', renameCurrentDataset);
    datasetNameCloseBtn.addEventListener('click', () => closeDatasetNameModal(null));
    datasetNameCancelBtn.addEventListener('click', () => closeDatasetNameModal(null));
    datasetNameSubmitBtn.addEventListener('click', submitDatasetNameModal);
    datasetNameInput.addEventListener('input', () => {
        datasetNameModalError.textContent = '';
        datasetNameModalError.classList.add('hidden');
    });
    datasetNameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            submitDatasetNameModal();
        }
    });
    toolsNavButtons.forEach((button) => {
        button.addEventListener('click', () => {
            setToolsView(button.dataset.toolView, { forceRefresh: button.dataset.toolView === 'blur' || button.dataset.toolView === 'mirror' });
        });
    });
    reshuffleBtn.addEventListener('click', reshuffleDataset);
    compressBtn.addEventListener('click', compressDataset);
    fitBtn.addEventListener('click', fitDataset);
    duplicateThresholdInput.addEventListener('input', () => {
        duplicateThresholdValue.textContent = duplicateThresholdInput.value;
    });
    duplicateScanBtn.addEventListener('click', scanDuplicateImages);
    duplicateSkipBtn.addEventListener('click', advanceDuplicatePair);
    duplicateLeftCard.addEventListener('click', () => keepDuplicateSide('left'));
    duplicateRightCard.addEventListener('click', () => keepDuplicateSide('right'));
    blurStrengthInput.addEventListener('input', () => updateBlurPreview(false));
    blurRerollBtn.addEventListener('click', () => updateBlurPreview(true));
    blurApplyBtn.addEventListener('click', createBlurredDatasetCopy);
    mirrorHorizontalInput.addEventListener('change', () => updateMirrorPreview(false));
    mirrorVerticalInput.addEventListener('change', () => updateMirrorPreview(false));
    mirrorExcludeControl1Input.addEventListener('change', () => updateMirrorPreview(false));
    mirrorExcludeControl2Input.addEventListener('change', () => updateMirrorPreview(false));
    mirrorExcludeControl3Input.addEventListener('change', () => updateMirrorPreview(false));
    mirrorRerollBtn.addEventListener('click', () => updateMirrorPreview(true));
    mirrorApplyBtn.addEventListener('click', createMirroredDatasetCopy);
    mergeApplyBtn.addEventListener('click', createMergedDataset);
    mergeSecondarySelect.addEventListener('change', () => {
        if (!mergeTargetNameInput.value.trim() && currentFolder && mergeSecondarySelect.value) {
            mergeTargetNameInput.value = `${currentFolder}_merged`;
        }
    });
    mergeTargetNameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            createMergedDataset();
        }
    });
    importScanBtn.addEventListener('click', scanImportDatasets);
    importPathInput.addEventListener('input', () => {
        const nextPath = importPathInput.value.trim();
        persistImportPath(nextPath);

        const cache = loadImportCache();
        if (!cache?.cachedPath || cache.cachedPath !== nextPath) {
            importScanResults = [];
            renderImportResults();
            if (nextPath) {
                setImportStatus('Path updated. Press Scan to build a fresh cache for this folder.', 'info');
            }
        }
    });
    importPathInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            scanImportDatasets();
        }
    });
    exportBtn.addEventListener('click', openExportModal);
    exportCloseBtn.addEventListener('click', closeExportModal);
    exportCancelBtn.addEventListener('click', closeExportModal);
    exportSubmitBtn.addEventListener('click', exportDataset);
    exportPathInput.addEventListener('input', () => {
        persistExportPath(exportPathInput.value);
        exportModalError.textContent = '';
        exportModalError.classList.add('hidden');
    });
    exportPathInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            exportDataset();
        }
    });
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
    transferModeSelect.addEventListener('change', (e) => {
        updateTransferMode(e.target.value);
    });

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

        // Let text and select controls handle their own keyboard input.
        if (document.activeElement === captionText || document.activeElement?.tagName === 'SELECT') {
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
        }
        if (e.shiftKey && e.key.toLowerCase() === 'p') openInPixelmator();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && toolsModal.classList.contains('active')) {
            closeToolsModal();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !datasetNameModal.classList.contains('hidden')) {
            closeDatasetNameModal(null);
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !exportModal.classList.contains('hidden')) {
            closeExportModal();
        }
    });

    toolsModal.addEventListener('click', (e) => {
        if (e.target === toolsModal) {
            closeToolsModal();
        }
    });

    datasetNameModal.addEventListener('click', (e) => {
        if (e.target === datasetNameModal) {
            closeDatasetNameModal(null);
        }
    });

    exportModal.addEventListener('click', (e) => {
        if (e.target === exportModal) {
            closeExportModal();
        }
    });

    updateToolsContext();
    setToolsView(currentToolsView);

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
    ptCloseBtn.focus();
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
    return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
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
        `Apply processing to ALL captions in "${currentFolder}"?\n\nA backup will be created inside the dataset before any file is changed.`
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
        const backupMsg = data.backup ? ` Backup: ${data.backup}.` : '';
        ptShowStatus(`Done — ${data.processed} captions updated.${backupMsg}${errMsg}`, data.errors?.length ? 'warning' : 'success');
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
