import os
import wget

def download_models():
    # Create directories
    os.makedirs('experiments/pretrained_models', exist_ok=True)
    
    # GFPGAN model
    gfpgan_model = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    gfpgan_path = 'experiments/pretrained_models/GFPGANv1.4.pth'
    
    # RealESRGAN model
    realesrgan_model = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    realesrgan_path = 'experiments/pretrained_models/RealESRGAN_x2plus.pth'
    
    # Download models if they don't exist
    if not os.path.exists(gfpgan_path):
        print("Downloading GFPGAN model...")
        wget.download(gfpgan_model, gfpgan_path)
        print("\nGFPGAN model downloaded successfully!")
    
    if not os.path.exists(realesrgan_path):
        print("Downloading RealESRGAN model...")
        wget.download(realesrgan_model, realesrgan_path)
        print("\nRealESRGAN model downloaded successfully!")

if __name__ == "__main__":
    download_models()