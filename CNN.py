
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Set image dimensions and batch size
img_width, img_height = 128, 128  # Resizing images to uniform dimensions
batch_size = 32
epochs = 20  # Number of training epochs
num_classes = 5  # Adjust this to match the number of categories (like colors or types)

# Define directory paths for training and validation data
train_dir = '/Users/shimasarah/Desktop/CNN-attempt/Data/Trainingg'  # Directory containing training images organized by class
val_dir = 'Data/validation'  # Directory with validation images, also organized by class

# Image Data Augmentation for training to improve model generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizing pixel values to be between 0 and 1
    rotation_range=20,  # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Shear transformation (like a slant)
    zoom_range=0.2,  # Randomly zoom into images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # How to fill in new pixels after a transformation
)

# Validation DataGenerator - No augmentation, only normalization
val_datagen = ImageDataGenerator(rescale=1./255)

# Generating training and validation data in batches
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),  # Resizing images to target dimensions
    batch_size=batch_size,
    class_mode='categorical'  # Assuming multiple classes, categorical encoding for multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model architecture - CNN for image classification
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),  # First Conv layer with 32 filters
    MaxPooling2D(pool_size=(2, 2)),  # Pooling layer to reduce spatial dimensions
    Conv2D(64, (3, 3), activation='relu'),  # Second Conv layer with 64 filters
    MaxPooling2D(pool_size=(2, 2)),  # Second Pooling layer
    Conv2D(128, (3, 3), activation='relu'),  # Third Conv layer with 128 filters for more complex features
    MaxPooling2D(pool_size=(2, 2)),  # Third Pooling layer
     Conv2D(256, (3, 3), activation='relu'),  # Fourth Conv layer with 256 filters for more complex features
    MaxPooling2D(pool_size=(2, 2)),  # Fourth Pooling layer
    Flatten(),  # Flattening to convert 2D matrix into a 1D vector
    Dense(256, activation='relu'),  # Fully connected layer with 256 neurons
    Dropout(0.5),  # Dropout to prevent overfitting by ignoring 50% of neurons during training
    Dense(num_classes, activation='softmax')  # Output layer with softmax for multi-class classification
])

# CNN.py
import tensorflow as tf

class UnoCardRecognizer:
    def __init__(self, model_path):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image):
        # Add your preprocessing for the input image here
        processed_image = self.preprocess(image)
        # Predict the class
        prediction = self.model.predict(processed_image)
        return prediction

    def preprocess(self, image):
        # Preprocess image to fit model requirements (resize, normalize, etc.)
        return image  # Modify this line as per your needs


# Compile the model - setting optimizer, loss function, and evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to monitor the validation loss and stop training when performance stabilizes
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Model training - fit the model on training data and validate using validation data
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stopping]  # Use early stopping as callback to avoid overfitting
)

# Evaluate model performance on the validation set
val_generator.reset()  # Reset validation generator to avoid shuffling issues
predictions = np.argmax(model.predict(val_generator), axis=-1)  # Predictions as class indices
true_labels = val_generator.classes  # True labels from the generator
class_labels = list(val_generator.class_indices.keys())  # Class label names

# Display model performance metrics
print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=class_labels))

print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))

# Save the trained model for later use in predictions
model.save('card_recognition_model.h5')

