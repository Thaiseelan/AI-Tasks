import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from PIL import Image

# --- CONFIGURATION ---
# Update this to match where your images are!
DATA_DIR = "images" 
MODEL_DIR = "models"

TARGET_SIZE = (64, 64)  

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"🚀 looking for images in: {os.path.abspath(DATA_DIR)}")

# --- LOAD DATA ---
data = []
labels = []

if not os.path.exists(DATA_DIR):
    print("❌ Error: Images folder not found!")
    exit()

categories = os.listdir(DATA_DIR)
for category in categories:
    folder_path = os.path.join(DATA_DIR, category)
    if not os.path.isdir(folder_path):
        continue

    print(f"   Processing {category}...")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(TARGET_SIZE)
            img_array = np.array(img).flatten() # This creates 12288 features
            data.append(img_array)
            labels.append(category)
        except Exception as e:
            pass

if len(data) == 0:
    print("❌ No images found. Check your DATA_DIR path!")
    exit()

X = np.array(data)
y = np.array(labels)
print(f"✅ Training on {len(X)} images with {X.shape[1]} features each.")

# --- TRAIN ---
models = {
    "forest": RandomForestClassifier(n_estimators=100),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "logistic": LogisticRegression(max_iter=1000),
    "svm": SVC(probability=True) 
}

for name, model in models.items():
    print(f"⚙️  Training {name}...")
    model.fit(X, y)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    print(f"💾 Saved {name}.pkl")