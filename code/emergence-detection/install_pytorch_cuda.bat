@echo off
echo Installing PyTorch with CUDA 12.1 support...
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo Installation completed. Running CUDA check...
python check_cuda.py 