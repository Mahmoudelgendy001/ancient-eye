import os, json, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 15

TRAIN_DIR   = "train"
VAL_DIR     = "val"
TEST_DIR    = "test"

MODELS_DIR  = "models"
MODEL_PATH      = os.path.join(MODELS_DIR, "ancienteye.h5")
CLASS_MAP_PATH  = os.path.join(MODELS_DIR, "class_indices.json")
HISTORY_PATH    = os.path.join(MODELS_DIR, "history.json")

os.makedirs(MODELS_DIR, exist_ok=True)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=20.0,
    fill_mode='nearest'
)

plain_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

val_gen = plain_aug.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = plain_aug.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_gen.num_classes
print(f"[INFO] عدد الكلاسات: {num_classes}")
print("[INFO] class indices:", train_gen.class_indices)

with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump({int(v): k for k, v in train_gen.class_indices.items()}, f, ensure_ascii=False, indent=2)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

class_weight_dict = dict(enumerate(class_weights))
print("[INFO] Class weights:", class_weight_dict)

# ====== Build Model ======
def build_transfer_model(num_classes=21):
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    base.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base

model, base_model = build_transfer_model(num_classes=num_classes)
model.summary()

def get_callbacks():
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

# ====== Phase 1 ======
print("\n[PHASE 1] Training head...")

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=get_callbacks(),
    class_weight=class_weight_dict
)

# ====== Phase 2 ======
print("\n[PHASE 2] Fine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=get_callbacks(),
    class_weight=class_weight_dict
)

# ====== Merge History ======
full_history = {}
for key in history1.history:
    full_history[key] = history1.history[key] + history2.history[key]

# ====== Save History (FIXED JSON) ======
def convert_history(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, list):
        return [convert_history(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_history(v) for k, v in obj.items()}
    return obj

with open(HISTORY_PATH, "w") as f:
    json.dump(convert_history(full_history), f, indent=2)

print(f"[INFO] History saved to {HISTORY_PATH}")

# ====== Test Evaluation ======
print("\n[INFO] Evaluating on TEST set...")

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_acc = model.evaluate(test_gen)
print(f"[RESULT] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# ======  CONFUSION MATRIX ====== 
print("\n[INFO] Generating Confusion Matrix...")

y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

class_names = list(test_gen.class_indices.keys())

cm = confusion_matrix(y_true, y_pred_classes)

print("\n[INFO] Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# ====== Save Test Results ======
test_results = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss)
}

with open(os.path.join(MODELS_DIR, "test_results.json"), "w") as f:
    json.dump(test_results, f, indent=2)

print("[INFO] Test results saved.")