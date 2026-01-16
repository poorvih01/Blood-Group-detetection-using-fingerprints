import os  #folder and paths
import cv2  #reading and resizing fingerprint imag
import joblib #saving model and scaler
import numpy as np #numerical operations
from skimage.morphology import skeletonize 
from skimage.filters import threshold_otsu 
from skimage.feature import local_binary_pattern #Extracts texture information
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATASET_DIR = "dataset"
IMG_SIZE = 224

def preprocess_fingerprint(img_path): # Preprocess fingerprint image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Read image in grayscale
    if img is None: 
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # Resize image to 224x224
    blur = cv2.GaussianBlur(img, (5,5), 0)  
    thresh = threshold_otsu(blur)  
    binary = blur < thresh 
    skeleton = skeletonize(binary) #Reduces ridges to thin lines for better feature extraction.
    return skeleton.astype(np.uint8) # Convert boolean to uint8

def is_fingerprint(skeleton):
    ridge_density = np.sum(skeleton) / (IMG_SIZE*IMG_SIZE) 
    return ridge_density > 0.02  # Minimum ridge density threshold

def extract_features(skeleton):
    ridge_density = np.sum(skeleton) / (IMG_SIZE*IMG_SIZE) 
    y, x = np.where(skeleton == 1) # Get coordinates of ridge pixels
    curvature = np.std(np.gradient(x)) if len(x) > 10 else 0 

    lbp = local_binary_pattern(skeleton, P=8, R=1, method="uniform") 
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True) 

    return np.concatenate([[ridge_density, curvature], lbp_hist]) 

X, y = [], []
label_map = {name: idx for idx, name in enumerate(sorted(os.listdir(DATASET_DIR)))}

for blood_group in os.listdir(DATASET_DIR):
    folder = os.path.join(DATASET_DIR, blood_group)
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        if not file.lower().endswith(".bmp"):
            continue

        path = os.path.join(folder, file)
        skeleton = preprocess_fingerprint(path)
        if skeleton is None:
            continue

        if not is_fingerprint(skeleton):
            continue

        features = extract_features(skeleton)
        X.append(features)
        y.append(label_map[blood_group])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300, 
    max_depth=15, 
    class_weight="balanced",  
    random_state=42 
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print("Random forest ") #print(f"✅ Accuracy: {acc:.4f}")

joblib.dump(model, "bloodgroup_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_map, "label_map.pkl")

print("✅ Model files saved successfully")
