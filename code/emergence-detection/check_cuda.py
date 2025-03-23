import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU only.")
    
print("\nHere are some potential things to check if CUDA is not working:")
print("1. Make sure NVIDIA drivers are installed")
print("2. Make sure CUDA toolkit is installed")
print("3. Make sure the PyTorch version matches your CUDA version")
print("4. Make sure your GPU is CUDA-compatible") 