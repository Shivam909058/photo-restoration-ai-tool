from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import os
import cv2
import numpy as np
import wget
import sys

# Add GFPGAN to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
gfpgan_path = os.path.join(current_dir, 'GFPGAN')
if not os.path.exists(gfpgan_path):
    os.system(f'git clone https://github.com/TencentARC/GFPGAN.git {gfpgan_path}')
sys.path.append(gfpgan_path)

# Now import GFPGAN related modules
from gfpgan import GFPGANer
from basicsr.utils import imwrite
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI(title="Photo Restoration API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("experiments/pretrained_models", exist_ok=True)

templates = Jinja2Templates(directory="templates")

def download_models():
    """Download required model weights if they don't exist"""
    models = {
        'GFPGAN': {
            'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
            'path': 'experiments/pretrained_models/GFPGANv1.4.pth'
        },
        'RealESRGAN': {
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'path': 'experiments/pretrained_models/RealESRGAN_x2plus.pth'
        }
    }
    
    for model_name, model_info in models.items():
        if not os.path.exists(model_info['path']):
            print(f"Downloading {model_name} model...")
            wget.download(model_info['url'], model_info['path'])
            print(f"\n{model_name} model downloaded successfully!")

def initialize_models():
    """Initialize GFPGAN and RealESRGAN models with optimized settings"""
    try:
        # Ensure models are downloaded
        download_models()
        
        # Set device and optimize CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize GFPGAN with optimized settings
        gfpgan = GFPGANer(
            model_path='experiments/pretrained_models/GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device
        )
        
        # Initialize RealESRGAN with optimized settings
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='experiments/pretrained_models/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False,
            device=device
        )
        
        gfpgan.bg_upsampler = bg_upsampler
        return gfpgan
    
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Initialize models at startup
try:
    gfpgan_model = initialize_models()
except Exception as e:
    print(f"Failed to initialize models: {str(e)}")
    gfpgan_model = None

@app.post("/restore-photo/")
async def restore_photo(file: UploadFile = File(...)):
    try:
        if gfpgan_model is None:
            raise Exception("Model not initialized properly")
            
        # Save uploaded file
        file_path = os.path.join("uploads", file.filename)
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        except Exception as e:
            raise Exception(f"Failed to save uploaded file: {str(e)}")
        
        # Read image
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Failed to read uploaded image")
        
        # Process image
        try:
            with torch.no_grad():  # Disable gradient calculation
                _, _, restored_img = gfpgan_model.enhance(
                    img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")
        
        # Save result
        result_path = os.path.join("results", f"restored_{file.filename}")
        try:
            imwrite(restored_img, result_path)
        except Exception as e:
            raise Exception(f"Failed to save restored image: {str(e)}")
        
        # Clean up
        try:
            os.remove(file_path)
        except:
            pass  # Ignore cleanup errors
        
        return FileResponse(
            result_path,
            media_type="image/jpeg",
            filename=f"restored_{file.filename}"
        )
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/")
async def root():
    return {
        "message": "Welcome to Photo Restoration API",
        "status": "active",
        "model_loaded": gfpgan_model is not None
    }

@app.get("/ui")
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)