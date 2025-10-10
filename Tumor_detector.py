import kagglehub

dataset_base_path = kagglehub.dataset_download("ahmedhamada0/brain-tumor-detection")

print("Path to dataset files:", dataset_base_path)
import os
import shutil
import random
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
# === CONFIGURATION ===
BASE_DIR = r"C:\Users\raulb\Desktop\Sandbox\Tumor_detector"

path_to_yes_images = os.path.join(dataset_base_path, "yes")
path_to_no_images = os.path.join(dataset_base_path, "no")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# === DIRECTORY STRUCTURE ===
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PRE_DIR = os.path.join(DATA_DIR, "preprocessed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [DATA_DIR, RAW_DIR, PRE_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# === COPY RAW DATA (if not already) ===
if not os.path.exists(os.path.join(RAW_DIR, "yes")):
    print("[INFO] Copying raw data (first run only)...")
    shutil.copytree(path_to_yes_images, os.path.join(RAW_DIR, "yes"))
    shutil.copytree(path_to_no_images, os.path.join(RAW_DIR, "no"))

# === SPLIT DATASET (train/val/test) ===
def split_dataset():
    print("[INFO] Splitting dataset...")
    for cls in ["yes", "no"]:
        src = os.path.join(RAW_DIR, cls)
        files = os.listdir(src)
        random.shuffle(files)
        n = len(files)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split, imgs in splits.items():
            split_dir = os.path.join(PRE_DIR, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in imgs:
                src_path = os.path.join(src, img)
                dst_path = os.path.join(split_dir, img)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)

if not os.path.exists(os.path.join(PRE_DIR, "train", "yes")):
    split_dataset()
else:
    print("[INFO] Preprocessed data already split. Skipping...")

# === DATA GENERATORS (with augmentation for training) ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(PRE_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = val_test_datagen.flow_from_directory(
    os.path.join(PRE_DIR, "val"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_gen = val_test_datagen.flow_from_directory(
    os.path.join(PRE_DIR, "test"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# === BUILD AUTOENCODER ===
def build_autoencoder():
    input_img = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2), padding='same', name="latent_space")(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    model = models.Model(input_img, decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

autoencoder = build_autoencoder()
print(autoencoder.summary())

# === TRAIN AUTOENCODER ===
def autoencoder_generator(gen):
    while True:
        x_batch, _ = next(gen)
        yield x_batch, x_batch

print("[INFO] Training autoencoder...")

auto_train_gen = autoencoder_generator(train_gen)
auto_val_gen = autoencoder_generator(val_gen)

history = autoencoder.fit(
    auto_train_gen,
    validation_data=auto_val_gen,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    epochs=EPOCHS,
    verbose=1
)

autoencoder.save(os.path.join(MODELS_DIR, "autoencoder.h5"))

# === SAVE RECONSTRUCTION SAMPLES ===
print("[INFO] Saving reconstruction samples...")
sample_batch, _ = next(val_gen)
decoded_imgs = autoencoder.predict(sample_batch[:5])
plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(sample_batch[i].squeeze(), cmap='gray')
    plt.axis("off")
    plt.subplot(2,5,i+6)
    plt.imshow(decoded_imgs[i].squeeze(), cmap='gray')
    plt.axis("off")
plt.suptitle("Top: Original | Bottom: Reconstructed")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "reconstructions.png"))
plt.close()

# === ENCODER EXTRACTION ===
encoder = models.Model(autoencoder.input, autoencoder.get_layer("latent_space").output)
encoder.save(os.path.join(MODELS_DIR, "encoder.h5"))

# === CLASSIFIER ON LATENT SPACE ===
def build_classifier(latent_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=latent_shape[1:]),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def latent_generator(gen):
    for batch, labels in gen:
        latent = encoder.predict(batch, verbose=0)
        yield latent, labels

latent_shape = encoder.output_shape
classifier = build_classifier(latent_shape)
print(classifier.summary())

print("[INFO] Training classifier...")
train_latent = latent_generator(train_gen)
val_latent = latent_generator(val_gen)

hist = classifier.fit(
    train_latent,
    steps_per_epoch=len(train_gen),
    validation_data=val_latent,
    validation_steps=len(val_gen),
    epochs=EPOCHS,
    verbose=1
)

classifier.save(os.path.join(MODELS_DIR, "classifier.h5"))

# === PLOT CLASSIFIER METRICS ===
plt.figure(figsize=(8, 5))
plt.plot(hist.history["accuracy"], label="Train Acc")
plt.plot(hist.history["val_accuracy"], label="Val Acc")
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Val Loss")
plt.title("Classifier Training Metrics")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "metrics_history.png"))
plt.close()

# === FINAL EVALUATION ===
print("[INFO] Evaluating on test set...")
test_latent = []
test_labels = []

for batch, labels in test_gen:
    latent = encoder.predict(batch, verbose=0)
    test_latent.append(latent)
    test_labels.append(labels)
    if len(test_latent) >= len(test_gen):
        break

X_test = np.concatenate(test_latent)
y_test = np.concatenate(test_labels)

preds = classifier.predict(X_test)
preds_binary = (preds > 0.5).astype(int)

cm = confusion_matrix(y_test, preds_binary)
report = classification_report(y_test, preds_binary, target_names=["No Tumor", "Tumor"])

with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

print("\n=== Evaluation Summary ===")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# === GRAD-CAM INTERPRETATION ===

def grad_cam_encoder(encoder, classifier, img_array, layer_name):
    img_tensor = tf.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=encoder.input,
        outputs=[encoder.get_layer(layer_name).output, encoder.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, latent_features = grad_model(img_tensor)
        predictions = classifier(latent_features)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

# === Generar Grad-CAMs ===
print("[INFO] Generating Grad-CAM visualizations...")

sample_gen = next(val_gen)
sample_imgs, sample_labels = sample_gen

indices = [i for i in range(6)]
layer_name = [layer.name for layer in encoder.layers if 'conv' in layer.name][-1]  # última capa conv

os.makedirs(os.path.join(RESULTS_DIR, "gradcam"), exist_ok=True)




tumor_dir = os.path.join(RESULTS_DIR, "gradcam", "tumor")
no_tumor_dir = os.path.join(RESULTS_DIR, "gradcam", "no_tumor")
os.makedirs(tumor_dir, exist_ok=True)
os.makedirs(no_tumor_dir, exist_ok=True)

sample_gen = next(val_gen)
sample_imgs, sample_labels = sample_gen

tumor_indices = [i for i, label in enumerate(sample_labels) if label == 1][:10]
no_tumor_indices = [i for i, label in enumerate(sample_labels) if label == 0][:10]

layer_name = [layer.name for layer in encoder.layers if 'conv' in layer.name][-1]

def save_gradcam(img_array, save_path):
    
    base, ext = os.path.splitext(save_path)
    original_path = f"{base}_original{ext}"

    
    original_img = np.uint8(255 * img_array.squeeze())
    heatmap = grad_cam_encoder(encoder, classifier, img_array, layer_name)
    heatmap = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    original = np.uint8(255 * img_array.squeeze())
    overlay = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), 0.6, heatmap_colored, 0.4, 0)

    combined = np.hstack((cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR), overlay))

    cv2.imwrite(f"{base}_combined{ext}", combined)


for idx in tumor_indices:
    out_path = os.path.join(tumor_dir, f"gradcam_{idx}.png")
    save_gradcam(sample_imgs[idx], out_path)

for idx in no_tumor_indices:
    out_path = os.path.join(no_tumor_dir, f"gradcam_{idx}.png")
    save_gradcam(sample_imgs[idx], out_path)

print(f"[INFO] Grad-CAM visualizations saved to:\n - Tumor: {tumor_dir}\n - No Tumor: {no_tumor_dir}")


print("[INFO] All done.")
