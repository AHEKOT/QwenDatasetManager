// Image Editor Module
class ImageEditor {
    constructor(canvas, targetImageElement, controlImageElement, overlayElement, subfolder = 'img') {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { willReadFrequently: true });
        this.targetImageElement = targetImageElement;
        this.controlImageElement = controlImageElement;
        this.overlayElement = overlayElement;
        this.subfolder = subfolder;
        this.onStateChange = null;

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

        // Spray state
        this.sprayInterval = null;
        this.sprayDensity = 0.5; // Dots per pixel area per tick approx

        // Crop state
        this.selectionRect = null; // {x, y, w, h} in canvas coordinates
        this.isSelecting = false;
        this.selectionStart = null;

        this.history = [];
        this.maxHistory = 20;
        this.hasChanges = false;
        this.currentFilename = null;
        this.currentFolder = null;

        this.lastDrawX = 0;
        this.lastDrawY = 0;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        // Optimized Blur Buffers
        this.blurBuffer = document.createElement('canvas'); // For grabbing a chunk
        this.blurBufferCtx = this.blurBuffer.getContext('2d');

        this.blurSmallBuffer = document.createElement('canvas'); // For downscaling
        this.blurSmallBuffer.width = 32;
        this.blurSmallBuffer.height = 32;
        this.blurSmallBufferCtx = this.blurSmallBuffer.getContext('2d');
        this.blurSmallBufferCtx.imageSmoothingEnabled = true;

        this.blurMaskBuffer = document.createElement('canvas'); // For circular mask
        this.blurMaskCtx = this.blurMaskBuffer.getContext('2d');

        this.setupCanvas();
        this.setupEventListeners();
    }

    // ... (rest of methods)

    // optimized drawSelectionLayer
    drawSelectionLayer() {
        if (!this.selectionRect) return;

        const { x, y, w, h } = this.selectionRect;

        // Draw standard overlay with "hole" using evenodd rule
        // This is much faster than putImageData or complex clipping
        this.ctx.save();
        this.ctx.beginPath();
        // Outer rectangle (canvas bounds)
        this.ctx.rect(0, 0, this.canvas.width, this.canvas.height);
        // Inner rectangle (selection) - drawn in opposite direction effectively for evenodd, 
        // but even simply defining two rects works with 'evenodd'
        this.ctx.rect(x, y, w, h);

        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.ctx.fill('evenodd');
        this.ctx.restore();

        // Border
        this.ctx.save();
        this.ctx.strokeStyle = '#fff';
        this.ctx.lineWidth = 2 / this.zoom; // Scale line width
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(x, y, w, h);

        // Handles
        this.ctx.setLineDash([]);
        this.ctx.fillStyle = '#fff';
        const handleSize = 8 / this.zoom;

        // Corners
        this.ctx.fillRect(x - handleSize / 2, y - handleSize / 2, handleSize, handleSize); // TL
        this.ctx.fillRect(x + w - handleSize / 2, y - handleSize / 2, handleSize, handleSize); // TR
        this.ctx.fillRect(x + w - handleSize / 2, y + h - handleSize / 2, handleSize, handleSize); // BR
        this.ctx.fillRect(x - handleSize / 2, y + h - handleSize / 2, handleSize, handleSize); // BL
        this.ctx.restore();
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
        const bytesPerState = Math.max(1, this.canvas.width * this.canvas.height * 4);
        this.maxHistory = Math.max(2, Math.min(20, Math.floor((96 * 1024 * 1024) / bytesPerState)));

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
        this.canvas.style.transformOrigin = '0 0'; // Ensure origin is top-left

        if (this.overlayElement) {
            this.overlayElement.style.transform = transform;
            this.overlayElement.style.transformOrigin = '0 0';
            this.overlayElement.style.width = `${this.canvas.width}px`;
            this.overlayElement.style.height = `${this.canvas.height}px`;
            this.overlayElement.style.left = '0';
            this.overlayElement.style.top = '0';
        }

        if (this.onStateChange) {
            this.onStateChange();
        }
    }

    setupEventListeners() {
        const activate = () => {
            ImageEditor.activeEditor = this;
            const sizeSlider = document.getElementById('brush-size');
            const opacitySlider = document.getElementById('brush-opacity');
            const colorPicker = document.getElementById('brush-color');
            if (sizeSlider) sizeSlider.value = this.brushSize;
            if (opacitySlider) opacitySlider.value = Math.round(this.brushOpacity * 100);
            if (colorPicker) colorPicker.value = this.brushColor;
            const sizeValue = document.getElementById('size-value');
            const opacityValue = document.getElementById('opacity-value-brush');
            if (sizeValue) sizeValue.textContent = this.brushSize;
            if (opacityValue) opacityValue.textContent = Math.round(this.brushOpacity * 100);
            this.onStateChange?.();
        };

        // Canvas-specific events remain bound to their own editor.
        this.canvas.parentElement.addEventListener('mousedown', (e) => {
            activate();
            this.handleMouseDown(e);
        });
        window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        window.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.parentElement.addEventListener('wheel', (e) => {
            activate();
            this.handleWheel(e);
        }, { passive: false });

        if (!ImageEditor.activeEditor) ImageEditor.activeEditor = this;
        if (ImageEditor.globalControlsBound) return;
        ImageEditor.globalControlsBound = true;

        const active = () => ImageEditor.activeEditor;
        const toolButtons = ['brush', 'picker', 'spray', 'blur', 'distort', 'stamp', 'eraser', 'crop'];
        toolButtons.forEach(tool => {
            const btn = document.getElementById(`${tool}-tool`);
            if (btn) btn.addEventListener('click', () => active()?.selectTool(tool));
        });

        const sizeSlider = document.getElementById('brush-size');
        if (sizeSlider) {
            sizeSlider.addEventListener('input', (e) => {
                const editor = active();
                if (!editor) return;
                editor.brushSize = parseInt(e.target.value);
                document.getElementById('size-value').textContent = editor.brushSize;
            });
        }

        const opacitySlider = document.getElementById('brush-opacity');
        if (opacitySlider) {
            opacitySlider.addEventListener('input', (e) => {
                const editor = active();
                if (!editor) return;
                editor.brushOpacity = parseInt(e.target.value) / 100;
                document.getElementById('opacity-value-brush').textContent = e.target.value;
            });
        }

        const colorPicker = document.getElementById('brush-color');
        if (colorPicker) {
            colorPicker.addEventListener('change', (e) => {
                const editor = active();
                if (editor) editor.brushColor = e.target.value;
            });
        }

        const undoBtn = document.getElementById('undo-btn');
        if (undoBtn) undoBtn.addEventListener('click', () => active()?.undo());

        const resetBtn = document.getElementById('reset-edit-btn');
        if (resetBtn) resetBtn.addEventListener('click', () => active()?.reset());

        const saveBtn = document.getElementById('save-edit-btn');
        if (saveBtn) saveBtn.addEventListener('click', () => active()?.save());

        const applyCropBtn = document.getElementById('apply-crop-btn');
        if (applyCropBtn) applyCropBtn.addEventListener('click', () => active()?.applyCrop());

        document.addEventListener('keydown', (e) => {
            const editor = active();
            if (!editor || editor.canvas.offsetParent === null) return;
            // Don't steal keys when user is typing in an input or textarea
            if (document.activeElement && (document.activeElement.tagName === 'TEXTAREA' || document.activeElement.tagName === 'INPUT')) return;

            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
                e.preventDefault();
                editor.undo();
            } else if (e.key === 'Enter') {
                if (editor.currentTool === 'crop' && editor.selectionRect) {
                    e.preventDefault();
                    editor.applyCrop();
                } else {
                    const saveBtn = document.getElementById('save-edit-btn');
                    if (saveBtn && !saveBtn.disabled) {
                        e.preventDefault();
                        editor.save();
                    }
                }
            } else if (e.key === '[' || e.key === ']') {
                const step = 2;
                let newSize = editor.brushSize;

                if (e.key === '[') {
                    newSize = Math.max(1, editor.brushSize - step);
                } else {
                    newSize = Math.min(100, editor.brushSize + step);
                }

                if (newSize !== editor.brushSize) {
                    editor.brushSize = newSize;
                    const sizeSlider = document.getElementById('brush-size');
                    const sizeValue = document.getElementById('size-value');

                    if (sizeSlider) sizeSlider.value = newSize;
                    if (sizeValue) sizeValue.textContent = newSize;
                }
            } else if (e.key === '1') {
                editor.selectTool('brush');
            } else if (e.key === '2' || (!e.shiftKey && e.key.toLowerCase() === 'p')) {
                editor.selectTool('picker');
            } else if (e.key === '3' || e.key.toLowerCase() === 's') {
                editor.selectTool('stamp');
            } else if (e.key === '4' || e.key.toLowerCase() === 'e') {
                editor.selectTool('eraser');
            } else if (e.key === '5' || e.key === 'c') {
                editor.selectTool('crop');
            } else if (e.key === 'a' || e.key === '6') {
                editor.selectTool('spray');
            } else if (e.key === 'b' || e.key === '7') {
                editor.selectTool('blur');
            } else if (e.key === 'd' || e.key === '8') {
                editor.selectTool('distort');
            }
        });
    }

    selectTool(tool) {
        this.currentTool = tool;
        document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
        const btn = document.querySelector(`[data-tool="${tool}"]`);
        if (btn) btn.classList.add('active');

        // Show/Hide relevant panels
        const brushPanel = document.getElementById('brush-settings-panel');
        const cropPanel = document.getElementById('crop-settings-panel');

        if (tool === 'crop') {
            if (brushPanel) brushPanel.style.display = 'none';
            if (cropPanel) cropPanel.style.display = 'block';
            this.canvas.style.cursor = 'crosshair';
            // Clear any existing selection when entering tool? No, keeps state.
            this.redrawCanvas(); // Redraw to clear handles if needed
        } else {
            if (brushPanel) brushPanel.style.display = 'block';
            if (cropPanel) cropPanel.style.display = 'none';
            this.canvas.style.cursor = 'crosshair';
            // If leaving crop tool, clear selection visualization
            this.selectionRect = null;
            this.redrawCanvas();
        }
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
        // Middle mouse or Alt+Click for panning
        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            e.preventDefault();
            this.isPanning = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
            return;
        }

        if (e.button === 0) {
            const coords = this.getCanvasCoords(e);

            // Check if click is within canvas bounds (mostly)
            // Allow starting slightly outside for easier edge selection

            if (this.currentTool === 'crop') {
                this.isSelecting = true;
                this.selectionStart = coords;
                this.selectionRect = { x: coords.x, y: coords.y, w: 0, h: 0 };
                // Hide apply button while selecting
                const applyBtn = document.getElementById('apply-crop-btn');
                if (applyBtn) applyBtn.disabled = true;

                this.drawSelection();
                return;
            }

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
                } else if (this.currentTool === 'spray') {
                    this.isDrawing = true;
                    this.lastDrawX = coords.x;
                    this.lastDrawY = coords.y;
                    this.startSpray(coords.x, coords.y);
                } else if (this.currentTool === 'blur') {
                    this.isDrawing = true;
                    this.lastDrawX = coords.x;
                    this.lastDrawY = coords.y;
                    this.applyLocalBlur(coords.x, coords.y);
                } else if (this.currentTool === 'distort') {
                    this.isDrawing = true;
                    this.lastDrawX = coords.x;
                    this.lastDrawY = coords.y;
                    this.applyLocalDistort(coords.x, coords.y);
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

        if (this.isSelecting && this.currentTool === 'crop') {
            const coords = this.getCanvasCoords(e);

            let w = coords.x - this.selectionStart.x;
            let h;

            const cropModeSelect = document.getElementById('crop-mode');
            const cropMode = cropModeSelect ? cropModeSelect.value : 'keep';

            if (cropMode === '1:1') {
                // 1:1 logic, consider dragging direction
                const rawH = coords.y - this.selectionStart.y;
                // Calculate height magnitude based on width magnitude to keep it square
                h = Math.abs(w) * Math.sign(rawH || 1);
            } else {
                // Constrain aspect ratio to image aspect ratio
                const imgAspect = this.canvas.width / this.canvas.height;
                // Force height based on width to maintain aspect
                h = w / imgAspect;
            }

            this.selectionRect = {
                x: this.selectionStart.x,
                y: this.selectionStart.y,
                w: w,
                h: h
            };

            this.drawSelection();
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
            } else if (this.currentTool === 'spray') {
                // Just update position, the interval handles the drawing
                // But we optionally spray along the line if mouse moves fast? 
                // Classic paint just sprays at current pos.
            } else if (this.currentTool === 'blur') {
                this.blurLine(this.lastDrawX, this.lastDrawY, coords.x, coords.y);
            } else if (this.currentTool === 'distort') {
                this.distortLine(this.lastDrawX, this.lastDrawY, coords.x, coords.y);
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

        if (this.isSelecting) {
            this.isSelecting = false;
            // Normalize selection rect (handle negative w/h)
            if (this.selectionRect) {
                let { x, y, w, h } = this.selectionRect;
                if (w < 0) { x += w; w = -w; }
                if (h < 0) { y += h; h = -h; }
                this.selectionRect = { x, y, w, h };

                // Enable apply button if selection is big enough
                const applyBtn = document.getElementById('apply-crop-btn');
                if (applyBtn) applyBtn.disabled = w < 10 || h < 10;
            }
            this.drawSelection();
        }

        if (this.isDrawing) {
            this.isDrawing = false;
            if (this.currentTool === 'spray') {
                this.stopSpray();
            }
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

    redrawCanvas() {
        // Redraw base state
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (this.history.length > 0) {
            this.ctx.putImageData(this.history[this.history.length - 1], 0, 0);
        } else {
            this.ctx.drawImage(this.targetImageElement, 0, 0);
        }

        if (this.currentTool === 'crop' && this.selectionRect) {
            this.drawSelectionLayer();
        }
    }

    drawSelection() {
        this.redrawCanvas();
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

        // Use composite operation for eraser? 
        // Or just draw the underlying control image?
        // Current implementation draws control image.

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

    startSpray(x, y) {
        if (this.sprayInterval) clearInterval(this.sprayInterval);

        // Instant spray first tick
        this.spray(x, y);

        // Then interval
        const fps = 30;
        this.sprayInterval = setInterval(() => {
            this.spray(this.lastDrawX, this.lastDrawY);
        }, 1000 / fps);
    }

    stopSpray() {
        if (this.sprayInterval) {
            clearInterval(this.sprayInterval);
            this.sprayInterval = null;
        }
    }

    spray(x, y) {
        const radius = this.brushSize / 2;
        const area = Math.PI * radius * radius;
        const dots = Math.max(1, Math.floor(area * 0.05)); // Density factor

        this.ctx.fillStyle = this.brushColor;
        this.ctx.globalAlpha = this.brushOpacity;

        for (let i = 0; i < dots; i++) {
            // Random point in circle
            const angle = Math.random() * Math.PI * 2;
            // Distribute uniformly in area (sqrt of random)
            const r = Math.sqrt(Math.random()) * radius;

            const dx = Math.cos(angle) * r;
            const dy = Math.sin(angle) * r;

            this.ctx.fillRect(Math.floor(x + dx), Math.floor(y + dy), 1, 1);
        }

        this.ctx.globalAlpha = 1;
    }

    blurLine(x1, y1, x2, y2) {
        // Reduced frequency: step by half the brush size instead of 1/4
        const stepSize = Math.max(1, this.brushSize / 2);
        const steps = Math.ceil(Math.hypot(x2 - x1, y2 - y1) / stepSize);
        for (let i = 0; i <= steps; i++) {
            const t = steps > 0 ? i / steps : 0;
            const x = x1 + (x2 - x1) * t;
            const y = y1 + (y2 - y1) * t;
            this.applyLocalBlur(x, y);
        }
    }

    applyLocalBlur(x, y) {
        const radius = this.brushSize / 2;
        const brushSize = Math.ceil(radius * 2);

        // Define source rectangle on main canvas
        let sx = Math.floor(x - radius);
        let sy = Math.floor(y - radius);
        let sw = brushSize;
        let sh = brushSize;

        // Clipped source bounds
        let csx = Math.max(0, sx);
        let csy = Math.max(0, sy);
        let csw = Math.min(this.canvas.width, sx + sw) - csx;
        let csh = Math.min(this.canvas.height, sy + sh) - csy;

        if (csw <= 0 || csh <= 0) return;

        // Resize blurBuffer only if needed
        if (this.blurBuffer.width < csw || this.blurBuffer.height < csh) {
            this.blurBuffer.width = Math.max(csw, 128);
            this.blurBuffer.height = Math.max(csh, 128);
            this.blurBufferCtx.imageSmoothingEnabled = true;
        }

        // 1. Grab chunk from main canvas
        this.blurBufferCtx.clearRect(0, 0, csw, csh);
        this.blurBufferCtx.drawImage(this.canvas, csx, csy, csw, csh, 0, 0, csw, csh);

        // 2. Perform downscale/upscale (already using pre-cached 32x32 filter)
        // Down
        this.blurSmallBufferCtx.drawImage(this.blurBuffer, 0, 0, csw, csh, 0, 0, 32, 32);

        // Up back to blurBuffer (replaces chunk with blurry version)
        this.blurBufferCtx.save();
        this.blurBufferCtx.globalCompositeOperation = 'copy';
        this.blurBufferCtx.drawImage(this.blurSmallBuffer, 0, 0, 32, 32, 0, 0, csw, csh);
        this.blurBufferCtx.restore();

        // 3. Draw back to main context using clipping for circular brush shape
        // This is more robust than separate mask buffer because clip path handles edge offsets automatically
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.clip();

        this.ctx.globalAlpha = this.brushOpacity;
        this.ctx.drawImage(this.blurBuffer, 0, 0, csw, csh, csx, csy, csw, csh);
        this.ctx.restore();
    }

    distortLine(x1, y1, x2, y2) {
        const stepSize = Math.max(1, this.brushSize / 2);
        const steps = Math.ceil(Math.hypot(x2 - x1, y2 - y1) / stepSize);
        for (let i = 0; i <= steps; i++) {
            const t = steps > 0 ? i / steps : 0;
            const x = x1 + (x2 - x1) * t;
            const y = y1 + (y2 - y1) * t;
            this.applyLocalDistort(x, y);
        }
    }

    applyLocalDistort(x, y) {
        const radius = this.brushSize / 2;
        const brushSize = Math.ceil(radius * 2);

        // Define source rectangle
        let sx = Math.floor(x - radius);
        let sy = Math.floor(y - radius);

        let csx = Math.max(0, sx);
        let csy = Math.max(0, sy);
        let csw = Math.min(this.canvas.width, sx + brushSize) - csx;
        let csh = Math.min(this.canvas.height, sy + brushSize) - csy;

        if (csw <= 0 || csh <= 0) return;

        // Ensure buffer is ready
        if (this.blurBuffer.width < csw || this.blurBuffer.height < csh) {
            this.blurBuffer.width = Math.max(csw, 128);
            this.blurBuffer.height = Math.max(csh, 128);
        }

        // 1. Grab chunk
        this.blurBufferCtx.clearRect(0, 0, csw, csh);
        this.blurBufferCtx.drawImage(this.canvas, csx, csy, csw, csh, 0, 0, csw, csh);

        // 2. Pixelate (extreme downscale/upscale with smoothing off)
        const blockSize = 24; // Fixed size for blockiness
        this.blurSmallBufferCtx.imageSmoothingEnabled = false;
        this.blurSmallBufferCtx.drawImage(this.blurBuffer, 0, 0, csw, csh, 0, 0, blockSize, blockSize);

        this.blurBufferCtx.save();
        this.blurBufferCtx.imageSmoothingEnabled = false;
        this.blurBufferCtx.globalCompositeOperation = 'copy';
        this.blurBufferCtx.drawImage(this.blurSmallBuffer, 0, 0, blockSize, blockSize, 0, 0, csw, csh);
        this.blurBufferCtx.restore();

        // 3. Draw back with Ghosting effect
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        this.ctx.clip();

        // Layer 1: Base pixelated
        this.ctx.globalAlpha = this.brushOpacity;
        this.ctx.drawImage(this.blurBuffer, 0, 0, csw, csh, csx, csy, csw, csh);

        // Layer 2: Offset Ghosting (doubling)
        const offset = Math.max(2, this.brushSize / 10);
        this.ctx.globalAlpha = this.brushOpacity * 0.5;
        this.ctx.drawImage(this.blurBuffer, 0, 0, csw, csh, csx + offset, csy + offset, csw, csh);

        this.ctx.restore();
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

            this.updateUndoButton();
            this.updateSaveButton();
        }
        // If undoing while crop tool active, might want to clear crop selection?
        // or just undo the paint operations. 
        // Currently Crop UI is separate layer (drawn on top of canvas content) 
        // so undo affects underlying pixels.
    }

    reset() {
        this.history = [];
        this.selectionRect = null; // Clear crop selection too
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.targetImageElement, 0, 0);
        this.hasChanges = false;
        this.saveState();
        this.fitToScreen();
        this.updateSaveButton();

        // Disable apply crop if reset
        const applyBtn = document.getElementById('apply-crop-btn');
        if (applyBtn) applyBtn.disabled = true;
    }

    updateUndoButton() {
        if (this.onStateChange) {
            this.onStateChange();
            return;
        }
        const undoBtn = document.getElementById('undo-btn');
        if (undoBtn) undoBtn.disabled = this.history.length <= 1;
    }

    updateSaveButton() {
        if (this.onStateChange) {
            this.onStateChange();
            return;
        }
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

        let wasSelecting = null;
        try {
            // Before saving, verify we are not in crop mode or clear crop handles?
            // The toBlob uses the context, which might have selection handles drawn if we didn't clear them?
            // redrawCanvas() calls drawSelectionLayer() at the end.
            // We should temporarily clear selection handles for the save.

            wasSelecting = this.selectionRect;
            this.selectionRect = null;
            this.redrawCanvas();

            const extension = this.currentFilename.split('.').pop().toLowerCase();
            const mimeType = {
                png: 'image/png',
                jpg: 'image/jpeg',
                jpeg: 'image/jpeg',
                webp: 'image/webp'
            }[extension];
            if (!mimeType) throw new Error('Unsupported image extension');
            const blob = await new Promise(resolve => this.canvas.toBlob(resolve, mimeType, 0.92));
            if (!blob) throw new Error('Browser failed to encode the edited image');
            const formData = new FormData();
            formData.append('file', blob, this.currentFilename);

            const response = await fetch(`/api/save/${encodeURIComponent(this.currentFilename)}?folder=${encodeURIComponent(this.currentFolder)}&subfolder=${encodeURIComponent(this.subfolder)}`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                this.hasChanges = false;
                this.history = [this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height)];

                // Notify app to update cache and UI
                if (window.onImageSaved) {
                    window.onImageSaved();
                }

                return true;
            } else {
                const data = await response.json().catch(() => ({}));
                alert(`Failed to save image: ${data.error || response.statusText}`);
                return false;
            }

        } catch (e) {
            console.error('Save error:', e);
            alert('Error saving image');
            return false;
        } finally {
            if (wasSelecting) {
                this.selectionRect = wasSelecting;
                this.redrawCanvas();
            }
            saveBtn.disabled = this.history.length <= 1;
            saveBtn.innerHTML = originalText;
            this.onStateChange?.();
        }
    }

    async applyCrop() {
        if (!this.selectionRect || !this.currentFolder || !this.currentFilename) return;
        if (this.hasChanges && !await this.save()) return;

        const applyBtn = document.getElementById('apply-crop-btn');
        const sourceExceptionSelect = document.getElementById('crop-source-exception');
        const sourceException = sourceExceptionSelect ? sourceExceptionSelect.value : '';

        const originalText = applyBtn.textContent;
        applyBtn.disabled = true;
        applyBtn.textContent = 'Creating...';

        const cropModeSelect = document.getElementById('crop-mode');
        const cropMode = cropModeSelect ? cropModeSelect.value : 'keep';

        try {
            const payload = {
                folder: this.currentFolder,
                filename: this.currentFilename,
                crop: this.selectionRect,
                sourceException: sourceException,
                mode: cropMode
            };

            const response = await fetch('/api/augment/crop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.success) {
                alert('Augmented pair created successfully!');
                // Reset selection
                this.selectionRect = null;
                this.redrawCanvas();
                applyBtn.disabled = true;

                // Refresh grid if needed
                if (window.onImageSaved) {
                    window.onImageSaved(true); // reloadList = true
                }
            } else {
                alert(`Failed to create augment: ${data.error}`);
            }

        } catch (e) {
            console.error('Crop error:', e);
            alert('Error creating augmented pair');
        } finally {
            applyBtn.textContent = originalText;
            if (this.selectionRect) applyBtn.disabled = false;
        }
    }
}

ImageEditor.activeEditor = null;
ImageEditor.globalControlsBound = false;
