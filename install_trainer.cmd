@echo off
setlocal
cd /d "%~dp0"

echo Installing Qwen Dataset Manager CUDA trainer...
py -3.12 -m venv trainer\.venv
call trainer\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-cache-dir -r trainer\ai_toolkit\requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch installed, but CUDA is not available'; print('CUDA trainer ready:', torch.cuda.get_device_name(0))"

echo Trainer installation complete. Restart Qwen Dataset Manager.
pause
