(function () {
    'use strict';

    const queue = [];
    let activeRequest = null;
    let previousFocus = null;
    let previousOverflow = '';

    function iconFor(tone) {
        if (tone === 'success') {
            return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"></path></svg>';
        }
        if (tone === 'error') {
            return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8v5"></path><path d="M12 17h.01"></path><circle cx="12" cy="12" r="9"></circle></svg>';
        }
        if (tone === 'warning') {
            return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 9v4"></path><path d="M12 17h.01"></path><path d="M10.3 4.3 2.7 17.5A2 2 0 0 0 4.4 20h15.2a2 2 0 0 0 1.7-2.5L13.7 4.3a2 2 0 0 0-3.4 0Z"></path></svg>';
        }
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 11v6"></path><path d="M12 7h.01"></path><circle cx="12" cy="12" r="9"></circle></svg>';
    }

    function ensureDialog() {
        let root = document.getElementById('app-dialog');
        if (root) return root;

        root = document.createElement('div');
        root.id = 'app-dialog';
        root.className = 'app-dialog hidden';
        root.setAttribute('role', 'dialog');
        root.setAttribute('aria-modal', 'true');
        root.setAttribute('aria-labelledby', 'app-dialog-title');
        root.setAttribute('aria-describedby', 'app-dialog-message');
        root.innerHTML = `
            <div class="app-dialog-card" tabindex="-1">
                <div class="app-dialog-accent"></div>
                <button class="app-dialog-close" type="button" aria-label="Close dialog">✕</button>
                <div class="app-dialog-body">
                    <div class="app-dialog-icon"></div>
                    <div class="app-dialog-copy">
                        <span class="app-dialog-kicker">Qwen Dataset Manager</span>
                        <h2 id="app-dialog-title"></h2>
                        <p id="app-dialog-message"></p>
                    </div>
                </div>
                <div class="app-dialog-actions">
                    <button class="action-btn app-dialog-cancel" type="button">Cancel</button>
                    <button class="action-btn export-btn app-dialog-confirm" type="button">OK</button>
                </div>
            </div>`;

        root.querySelector('.app-dialog-close').addEventListener('click', () => finish(false));
        root.querySelector('.app-dialog-cancel').addEventListener('click', () => finish(false));
        root.querySelector('.app-dialog-confirm').addEventListener('click', () => finish(true));
        root.addEventListener('click', (event) => {
            if (event.target === root) finish(false);
        });
        document.addEventListener('keydown', (event) => {
            if (!activeRequest || event.key !== 'Escape') return;
            event.preventDefault();
            finish(false);
        });
        document.body.appendChild(root);
        return root;
    }

    function render(request) {
        const root = ensureDialog();
        const options = request.options;
        const tone = ['success', 'error', 'warning'].includes(options.tone) ? options.tone : 'info';
        const hasCancel = Boolean(options.cancelLabel);

        root.dataset.tone = tone;
        root.querySelector('.app-dialog-icon').innerHTML = iconFor(tone);
        root.querySelector('#app-dialog-title').textContent = options.title || 'Notice';
        root.querySelector('#app-dialog-message').textContent = String(options.message || '');

        const closeButton = root.querySelector('.app-dialog-close');
        const cancelButton = root.querySelector('.app-dialog-cancel');
        const confirmButton = root.querySelector('.app-dialog-confirm');
        cancelButton.textContent = options.cancelLabel || 'Cancel';
        cancelButton.classList.toggle('hidden', !hasCancel);
        confirmButton.textContent = options.confirmLabel || 'OK';
        confirmButton.classList.toggle('is-danger', Boolean(options.danger));
        closeButton.setAttribute('aria-label', hasCancel ? 'Cancel' : 'Close dialog');

        previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
        previousOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        root.classList.remove('hidden');

        requestAnimationFrame(() => {
            (options.danger && hasCancel ? cancelButton : confirmButton).focus();
        });
    }

    function finish(result) {
        if (!activeRequest) return;

        const root = ensureDialog();
        const request = activeRequest;
        activeRequest = null;
        root.classList.add('hidden');
        document.body.style.overflow = previousOverflow;
        request.resolve(result);

        if (previousFocus && previousFocus.isConnected) previousFocus.focus();
        previousFocus = null;

        if (queue.length) {
            activeRequest = queue.shift();
            render(activeRequest);
        }
    }

    function show(options) {
        return new Promise((resolve) => {
            const request = { options: options || {}, resolve };
            if (activeRequest) {
                queue.push(request);
                return;
            }
            activeRequest = request;
            render(request);
        });
    }

    window.appDialog = Object.freeze({
        alert(message, options = {}) {
            return show({ ...options, message, cancelLabel: null });
        },
        confirm(message, options = {}) {
            return show({ ...options, message, cancelLabel: options.cancelLabel || 'Cancel' });
        }
    });
})();
