// Image Editor Module
class ImageEditor {
    constructor(canvas, targetImageElement, controlImageElement, overlayElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { willReadFrequently: true });
        this.targetImageElement = targetImageElement;
        this.controlImageElement = controlImageElement;
        this.overlayElement = overlayElement;
        
        this.currentTool = null;
        this.brushSize = 10;
        this.brushOpacity = 1;
        this.brushColor = '#ff0000';
        
        this.isDrawing = false;
        this.isPanning = false;
        
        // Transform state
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        
        this.cloneSource = null;
        this.stampOffset = { x: 0, y: 0 };

        this.history = [];
        this.maxHistory = 50;
        this.hasChanges = false;
        this.currentFilename = null;
        this.currentFolder = null;
        
        this.lastDrawX = 0;
        this.lastDrawY = 0;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        
        this.setupCanvas();
        this.setupEventListeners();
    }
    
    setupCanvas() {
        const img = this.targetImageElement;
        if (img.complete) {
            this.initializeCanvas();
        } else {
            img.onload = () => this.initializeCanvas();
        }
    }
    
    initializeCanvas() {
        const img = this.targetImageElement;
        this.canvas.width = img.naturalWidth || img.width;
        this.canvas.height = img.naturalHeight || img.height;
        
        // Draw initial image
        this.ctx.drawImage(img, 0, 0);
        this.saveState();
        this.canvas.classList.add('active');
        
        // Fit to screen initially
        this.fitToScreen();
    }
    
    fitToScreen() {
        const container = this.canvas.parentElement;
        if (!container) return;
        
        const padding = 40;
        const availWidth = container.clientWidth - padding;
        const availHeight = container.clientHeight - padding;
        
        const scaleX = availWidth / this.canvas.width;
        const scaleY = availHeight / this.canvas.height;
        
        this.zoom = Math.min(scaleX, scaleY, 1); // Don't zoom in by default, only out
        
        // Center
        this.panX = (container.clientWidth - this.canvas.width * this.zoom) / 2;
        this.panY = (container.clientHeight - this.canvas.height * this.zoom) / 2;
        
        this.applyTransform();
    }
    
    applyTransform() {
        const transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
        this.canvas.style.transform = transform;
        
        if (this.overlayElement) {
            this.overlayElement.style.transform = transform;
            this.overlayElement.style.transformOrigin = '0 0';
            this.overlayElement.style.width = `${this.canvas.width}px`;
            this.overlayElement.style.height = `${this.canvas.height}px`;
            this.overlayElement.style.left = '0';
            this.overlayElement.style.top = '0';
        }
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.parentElement.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        window.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        
        // Wheel zoom
        this.canvas.parentElement.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });
        
        // Tools
        const toolButtons = ['brush', 'picker', 'stamp', 'eraser'];
        toolButtons.forEach(tool => {
            const btn = document.getElementById(`${tool}-tool`);
            if (btn) btn.addEventListener('click', () => this.selectTool(tool));
        });
        
        // Settings
        const sizeSlider = document.getElementById('brush-size');
        if (sizeSlider) {
            sizeSlider.addEventListener('input', (e) => {
                this.brushSize = parseInt(e.target.value);
                document.getElementById('size-value').textContent = this.brushSize;
            });
        }
        
        const opacitySlider = document.getElementById('brush-opacity');
        if (opacitySlider) {
            opacitySlider.addEventListener('input', (e) => {
                this.brushOpacity = parseInt(e.target.value) / 100;
                document.getElementById('opacity-value-brush').textContent = e.target.value;
            });
        }
        
        const colorPicker = document.getElementById('brush-color');
        if (colorPicker) {
            colorPicker.addEventListener('change', (e) => {
                this.brushColor = e.target.value;
            });
        }
        
        // Actions
        const undoBtn = document.getElementById('undo-btn');
        if (undoBtn) undoBtn.addEventListener('click', () => this.undo());
        
        const resetBtn = document.getElementById('reset-edit-btn');
        if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
        
        const saveBtn = document.getElementById('save-edit-btn');
        if (saveBtn) saveBtn.addEventListener('click', () => this.save());

        document.addEventListener('keydown', (e) => {
            // Only handle shortcuts if canvas is visible
            if (this.canvas.offsetParent === null) return;

            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                this.undo();
            } else if (e.key === 'Enter') {
                const saveBtn = document.getElementById('save-edit-btn');
                if (saveBtn && !saveBtn.disabled) {
                    e.preventDefault();
                    this.save();
                }
            } else if (e.key === '[' || e.key === ']') {
                const step = 2;
                let newSize = this.brushSize;
                
                if (e.key === '[') {
                    newSize = Math.max(1, this.brushSize - step);
                } else {
                    newSize = Math.min(100, this.brushSize + step);
                }
                
                if (newSize !== this.brushSize) {
                    this.brushSize = newSize;
                    const sizeSlider = document.getElementById('brush-size');
                    const sizeValue = document.getElementById('size-value');
                    
                    if (sizeSlider) sizeSlider.value = newSize;
                    if (sizeValue) sizeValue.textContent = newSize;
                }
            } else if (e.key === '1') {
                this.selectTool('brush');
            } else if (e.key === '2') {
                this.selectTool('picker');
            } else if (e.key === '3') {
                this.selectTool('stamp');
            } else if (e.key === '4') {
                this.selectTool('eraser');
            }
        });
    }
    
    selectTool(tool) {
        this.currentTool = tool;
        document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
        const btn = document.querySelector(`[data-tool="${tool}"]`);
        if (btn) btn.classList.add('active');
    }
    
    // Robust coordinate calculation using getBoundingClientRect
    getCanvasCoords(e) {
        const rect = this.canvas.getBoundingClientRect();
        
        // Calculate scale factor between visual size and actual size
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        
        // Calculate position relative to the visual top-left, then scale to canvas coords
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        return { x, y };
    }
    
    handleMouseDown(e) {
        // Middle mouse or Space+Click for panning
        if (e.button === 1 || (e.button === 0 && e.code === 'Space')) {
            e.preventDefault();
            this.isPanning = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
            return;
        }
        
        if (e.button === 0) {
            const coords = this.getCanvasCoords(e);
            
            // Check if click is within canvas bounds
            if (coords.x >= 0 && coords.x <= this.canvas.width && 
                coords.y >= 0 && coords.y <= this.canvas.height) {
                
                if (!this.currentTool) return;

                if (this.currentTool === 'picker') {
                    this.pickColor(coords);
                } else if (this.currentTool === 'stamp') {
                    if (e.metaKey || e.ctrlKey) {
                        this.cloneSource = { x: coords.x, y: coords.y };
                        return;
                    }
                    
                    if (!this.cloneSource) return;

                    this.isDrawing = true;
                    this.lastDrawX = coords.x;
                    this.lastDrawY = coords.y;
                    this.stampOffset = { 
                        x: this.cloneSource.x - coords.x, 
                        y: this.cloneSource.y - coords.y 
                    };
                    this.stampLine(coords.x, coords.y, coords.x, coords.y);
                } else if (this.currentTool === 'brush') {
                    this.isDrawing = true;
                    this.lastDrawX = coords.x;
                    this.lastDrawY = coords.y;
                    this.drawPoint(coords.x, coords.y);
                } else if (this.currentTool === 'eraser') {
                    this.isDrawing = true;
                    this.lastDrawX = coords.x;
                    this.lastDrawY = coords.y;
                    this.eraseLine(coords.x, coords.y, coords.x, coords.y);
                }
            }
        }
    }
    
    handleMouseMove(e) {
        if (this.isPanning) {
            e.preventDefault();
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            
            this.panX += deltaX;
            this.panY += deltaY;
            
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            
            this.applyTransform();
            return;
        }
        
        if (this.isDrawing) {
            const coords = this.getCanvasCoords(e);
            
            if (this.currentTool === 'brush') {
                this.drawLine(this.lastDrawX, this.lastDrawY, coords.x, coords.y);
            } else if (this.currentTool === 'eraser') {
                this.eraseLine(this.lastDrawX, this.lastDrawY, coords.x, coords.y);
            } else if (this.currentTool === 'stamp' && this.cloneSource) {
                this.stampLine(this.lastDrawX, this.lastDrawY, coords.x, coords.y);
            }
            
            this.lastDrawX = coords.x;
            this.lastDrawY = coords.y;
        }
    }
    
    handleMouseUp(e) {
        if (this.isPanning) {
            this.isPanning = false;
            this.canvas.style.cursor = 'crosshair';
        }
        
        if (this.isDrawing) {
            this.isDrawing = false;
            this.saveState();
        }
    }
    
    handleWheel(e) {
        e.preventDefault();
        
        const rect = this.canvas.getBoundingClientRect();
        const containerRect = this.canvas.parentElement.getBoundingClientRect();
        
        // Mouse position relative to the container
        const mouseInContainerX = e.clientX - containerRect.left;
        const mouseInContainerY = e.clientY - containerRect.top;
        
        // Mouse position relative to the canvas (visual pixels)
        const mouseInCanvasX = e.clientX - rect.left;
        const mouseInCanvasY = e.clientY - rect.top;
        
        // Calculate the point in actual canvas coordinates (0 to width)
        const canvasPointX = mouseInCanvasX * (this.canvas.width / rect.width);
        const canvasPointY = mouseInCanvasY * (this.canvas.height / rect.height);
        
        // Calculate new zoom
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(0.1, Math.min(10, this.zoom * zoomFactor));
        
        // Calculate new pan positions
        // We want: newPanX + canvasPointX * newZoom = mouseInContainerX
        // So: newPanX = mouseInContainerX - canvasPointX * newZoom
        
        this.panX = mouseInContainerX - canvasPointX * newZoom;
        this.panY = mouseInContainerY - canvasPointY * newZoom;
        this.zoom = newZoom;
        
        this.applyTransform();
    }
    
    drawPoint(x, y) {
        this.ctx.fillStyle = this.brushColor;
        this.ctx.globalAlpha = this.brushOpacity;
        this.ctx.beginPath();
        this.ctx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.globalAlpha = 1;
    }
    
    drawLine(x1, y1, x2, y2) {
        this.ctx.strokeStyle = this.brushColor;
        this.ctx.globalAlpha = this.brushOpacity;
        this.ctx.lineWidth = this.brushSize;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
        this.ctx.globalAlpha = 1;
    }
    
    eraseLine(x1, y1, x2, y2) {
        const steps = Math.ceil(Math.hypot(x2 - x1, y2 - y1));
        for (let i = 0; i <= steps; i++) {
            const t = steps > 0 ? i / steps : 0;
            const x = x1 + (x2 - x1) * t;
            const y = y1 + (y2 - y1) * t;
            
            this.ctx.save();
            this.ctx.globalAlpha = this.brushOpacity;
            this.ctx.beginPath();
            this.ctx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
            this.ctx.clip();
            
            this.ctx.drawImage(this.controlImageElement, 0, 0);
            this.ctx.restore();
        }
    }
    
    stampLine(x1, y1, x2, y2) {
        const steps = Math.ceil(Math.hypot(x2 - x1, y2 - y1));
        for (let i = 0; i <= steps; i++) {
            const t = steps > 0 ? i / steps : 0;
            const x = x1 + (x2 - x1) * t;
            const y = y1 + (y2 - y1) * t;
            
            const sourceX = x + this.stampOffset.x;
            const sourceY = y + this.stampOffset.y;
            
            this.ctx.globalAlpha = this.brushOpacity;
            this.ctx.drawImage(
                this.controlImageElement,
                sourceX - this.brushSize / 2,
                sourceY - this.brushSize / 2,
                this.brushSize,
                this.brushSize,
                x - this.brushSize / 2,
                y - this.brushSize / 2,
                this.brushSize,
                this.brushSize
            );
        }
    }
    
    pickColor(coords) {
        try {
            const imageData = this.ctx.getImageData(Math.floor(coords.x), Math.floor(coords.y), 1, 1);
            const [r, g, b] = imageData.data;
            const hex = '#' + [r, g, b].map(x => ('0' + x.toString(16)).slice(-2)).join('').toUpperCase();
            
            this.brushColor = hex;
            const colorPicker = document.getElementById('brush-color');
            if (colorPicker) colorPicker.value = hex;
            
            this.selectTool('brush');
        } catch (e) {
            console.error('Color picker error:', e);
        }
    }
    
    saveState() {
        try {
            const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
            this.history.push(imageData);
            
            if (this.history.length > this.maxHistory) {
                this.history.shift();
            }
            
            // If we have more than 1 state (initial state + changes), we have changes
            if (this.history.length > 1) {
                this.hasChanges = true;
            }
            
            this.updateUndoButton();
            this.updateSaveButton();
        } catch (e) {
            console.error('Save state error:', e);
        }
    }
    
    undo() {
        if (this.history.length > 1) {
            this.history.pop();
            const imageData = this.history[this.history.length - 1];
            this.ctx.putImageData(imageData, 0, 0);
            
            // If we are back to 1 state, no changes (or we could track index)
            // For simplicity, if history > 1 we assume changes, but strictly speaking 
            // we might have undone all changes. 
            // Let's keep it simple: if history > 1, enable save.
            // Actually, better logic: if we undo, we still might want to save the previous state?
            // No, undo means revert.
            
            this.updateUndoButton();
            this.updateSaveButton();
        }
    }
    
    reset() {
        this.history = [];
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.targetImageElement, 0, 0);
        this.hasChanges = false;
        this.saveState();
        this.fitToScreen();
        this.updateSaveButton();
    }
    
    updateUndoButton() {
        const undoBtn = document.getElementById('undo-btn');
        if (undoBtn) undoBtn.disabled = this.history.length <= 1;
    }

    updateSaveButton() {
        const saveBtn = document.getElementById('save-edit-btn');
        if (saveBtn) saveBtn.disabled = this.history.length <= 1;
    }

    async save() {
        if (!this.currentFilename || !this.currentFolder) {
            console.error('No filename or folder set for saving');
            return;
        }
        
        const saveBtn = document.getElementById('save-edit-btn');
        const originalText = saveBtn.innerHTML;
        saveBtn.disabled = true;
        saveBtn.textContent = 'Saving...';
        
        try {
            const blob = await new Promise(resolve => this.canvas.toBlob(resolve));
            const formData = new FormData();
            formData.append('file', blob, this.currentFilename);
            
            const response = await fetch(`/api/save/${encodeURIComponent(this.currentFilename)}?folder=${encodeURIComponent(this.currentFolder)}`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                this.hasChanges = false;
                
                // Notify app to update cache and UI
                if (window.onImageSaved) {
                    window.onImageSaved();
                }
                
                alert('Image saved successfully');
            } else {
                alert('Failed to save image');
            }
        } catch (e) {
            console.error('Save error:', e);
            alert('Error saving image');
        } finally {
            saveBtn.disabled = this.history.length <= 1;
            saveBtn.innerHTML = originalText;
        }
    }
}
