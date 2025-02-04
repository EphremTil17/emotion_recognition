import torch
import sys

def check_pytorch_cuda():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("\nCUDA is not available. This might be because:")
        print("1. PyTorch was not installed with CUDA support")
        print("2. NVIDIA drivers are not properly installed")
        print("3. Your PyTorch version doesn't match your CUDA version")
        print("\nTo fix this:")
        print("1. Uninstall PyTorch: pip uninstall torch torchvision torchaudio")
        print("2. Install PyTorch with CUDA: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Make sure NVIDIA drivers are up to date")

if __name__ == "__main__":
    check_pytorch_cuda()