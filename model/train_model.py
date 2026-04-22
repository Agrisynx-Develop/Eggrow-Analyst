import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.optimizers import Adam


# ======================
# PATH SETUP
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "..", "Data", "train_data.csv")
IMG_DIR = os.path.join(BASE_DIR, "..", "Data", "Train")

# ======================
# LOAD CSV
# ======================
df = pd.read_csv(CSV_PATH)

# ======================
# PARAMETER
# ======================
IMG_SIZE = (224, 224)
images = []
labels = []

print("🔄 Loading images...")

# ======================
# LOOP DATA
# ======================
for idx, row in df.iterrows():

    # 1. Ambil path gambar
    img_path = os.path.join(IMG_DIR, row["images"])

    # 2. Cek file ada
    if not os.path.exists(img_path):
        print(f"❌ NOT FOUND: {img_path}")
        continue

    # 3. Baca gambar
    img = cv2.imread(img_path)

    # 4. Cek gagal baca
    if img is None:
        print(f"❌ FAILED READ: {img_path}")
        continue

    # 5. Resize (auto fit ke 128x128)
    img = cv2.resize(img, (128, 128))

    # 6. Normalisasi
    img = img.astype("float32") / 255.0

    # 7. Simpan ke list
    images.append(img)
    labels.append(row["label"])

# ======================
# VALIDASI DATA
# ======================
if len(images) == 0:
    raise ValueError("❌ Tidak ada gambar yang berhasil dimuat!")

print(f"✅ Total gambar berhasil: {len(images)}")
# ======================
# VALIDASI DATA
# ======================
if len(images) == 0:
    raise ValueError("❌ Tidak ada gambar yang berhasil dibaca. Cek path dataset!")

X = np.array(images)
y = np.array(labels)

# ======================
# ENCODING LABEL
# ======================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# ======================
# SPLIT DATA
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# ======================
# MODEL
# ======================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    BatchNormalization(),
    Dense(128, activation="relu"),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# TRAINING
# ======================
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16
)

for layer in base_model.layers[-30:]:
    layer.trainable = True
# ======================
# SAVE MODEL
# ======================
MODEL_PATH = os.path.join(BASE_DIR,  "eggrow_vision_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "labels.npy")

model.save(MODEL_PATH)
np.save(LABEL_PATH, le.classes_)

print("✅ Model saved:", MODEL_PATH)
print("✅ Labels saved:", LABEL_PATH)
