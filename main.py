import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# 1. SETUP & INSTANT DOWNLOAD
# We use 'with_info=True' to get metadata automatically
print("Loading Malaria Dataset (Standard Medical Imaging Benchmark)...")
dataset, info = tfds.load('malaria', split='train', as_supervised=True, with_info=True)

# 2. DATA PIPELINE (The Professional Way)
# Real projects don't use 'for loops' to load images. They use pipelines.
def preprocess(image, label):
    image = tf.image.resize(image, (100, 100)) # Resize to standard 100x100
    image = tf.cast(image, tf.float32) / 255.0 # Normalize to [0,1]
    return image, label

# Split into Train (80%) and Test (20%)
train_size = int(0.8 * info.splits['train'].num_examples)
train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

# Apply processing efficiently
BATCH_SIZE = 64
train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Data ready: {info.splits['train'].num_examples} Microscopic Images.")

# 3. BUILD THE CNN (VGG-Style Architecture)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(100, 100, 3)),
    
    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Classification Head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), # Prevent overfitting
    tf.keras.layers.Dense(1, activation='sigmoid') # Binary Output (0=Uninfected, 1=Parasitized)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. TRAIN (Fast on GPU)
print("\nStarting Training...")
history = model.fit(train_ds, epochs=5, validation_data=test_ds)

# 5. RESULTS & VISUALIZATION (For your GitHub README)
print("\nEvaluating...")
loss, accuracy = model.evaluate(test_ds)
print(f"Final Medical Diagnosis Accuracy: {accuracy*100:.2f}%")

# Plot standard accuracy curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.legend()
plt.savefig('training_curves.png') # Saves image for your GitHub
plt.show()

# Save the model file for GitHub upload
model.save('malaria_detection_cnn.h5')
print("Model saved as 'malaria_detection_cnn.h5'")
