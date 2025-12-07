import os
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

app = FastAPI()

# Mount folders for templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- CONFIGURATION (IMPORTANT) ---
# You MUST change this to match the size you used during training!
# Example: If you resized images to 64x64 during training, set this to (64, 64).
TARGET_SIZE = (64, 64) 

# --- LOAD MODELS ---
# We load models once at startup to make the app fast
models = {}
model_dir = "models"
model_files = {
    "forest": "forest.pkl",
    "knn": "knn.pkl",
    "logistic": "logistic.pkl",
    "svm": "svm.pkl"
}

print("Loading models...")
for name, filename in model_files.items():
    path = os.path.join(model_dir, filename)
    try:
        # Using joblib to load standard sklearn models
        models[name] = joblib.load(path)
        print(f"✅ Loaded {name}")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")

# --- HELPER FUNCTIONS ---
def process_image(image_bytes):
    """
    Converts raw image bytes into a flattened numpy array 
    that sklearn models can understand.
    """
    # Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (in case of PNG with transparency)
    img = img.convert('RGB')
    
    # Resize to the same size used in training
    img = img.resize(TARGET_SIZE)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Flatten the array (sklearn expects 1D array per sample)
    # Example: 64x64x3 becomes 12288 features
    flattened_img = img_array.flatten().reshape(1, -1)
    
    return flattened_img

# --- ROUTES ---

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    model_type: str = Form(...)
):
    # Validate model selection
    if model_type not in models:
        return {"error": "Model not found or failed to load."}
    
    selected_model = models[model_type]
    
    try:
        # Read and process image
        contents = await file.read()
        processed_data = process_image(contents)
        
        # Predict
        prediction = selected_model.predict(processed_data)[0]
        
        # Try to get confidence score (probability)
        # Note: SVM requires probability=True during training to support this
        confidence = "N/A"
        try:
            probs = selected_model.predict_proba(processed_data)
            max_prob = np.max(probs) * 100
            confidence = f"{max_prob:.2f}%"
        except:
            confidence = "Score Unavailable"

        return {
            "animal": prediction,
            "confidence": confidence,
            "model_used": model_type
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)