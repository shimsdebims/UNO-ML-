import tensorflow as tf  # Import TensorFlow for loading and using the model
import cv2  # OpenCV library for image processing
import numpy as np  # NumPy for numerical operations on arrays
from tensorflow.keras.models import load_model  # Function to load the trained Keras model
import logging  # Logging module for logging messages
import os  # OS module for file and directory operations
import matplotlib.pyplot as plt  # For plotting results
import seaborn as sns  # For creating statistical graphics
from sklearn.metrics import confusion_matrix, classification_report  # For evaluating model performance
import argparse  # For parsing command-line arguments
from datetime import datetime  # For handling date and time

# Configure logging to log information, errors, etc. to a file with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'uno_recognition_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='w'
)
logger = logging.getLogger(__name__)  # Create a logger for this module

class UnoCardRecognizer:
    def __init__(self, model_path='card_recognition_model.h5'):
        """Initialize the UNO card recognizer with the trained model."""
        try:
            self.model = load_model(model_path)  # Load the trained model from the specified path
            logger.info(f"Model loaded successfully from {model_path}")  # Log success message
            self.img_width = 128  # Width for resizing images
            self.img_height = 128  # Height for resizing images
            self.class_labels = ['Red', 'Blue', 'Green', 'Yellow', 'Special']  # Class labels for the output
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")  # Log error if model loading fails
            raise  # Raise the exception

    def preprocess_image(self, image):
        """Preprocess image for model prediction."""
        try:
            # Detect and crop card
            card_image = self.detect_card(image)
            # Resize image to match model's expected input size
            resized = cv2.resize(card_image, (self.img_width, self.img_height))
            # Normalize pixel values to the range [0, 1]
            normalized = resized.astype('float32') / 255.0
            # Add batch dimension (1 sample) for prediction
            processed = np.expand_dims(normalized, axis=0)
            return processed, card_image  # Return the processed image and the cropped card image
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")  # Log error if preprocessing fails
            raise

    def detect_card(self, image):
        """Detect and crop UNO card from image."""
        try:
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Detect edges in the image
            edges = cv2.Canny(blurred, 50, 150)
            # Find contours in the edge-detected image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:  # If no contours are found
                logger.warning("No card detected in image")  # Log warning message
                return image  # Return original image

            # Find the largest contour, which is assumed to be the card
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)  # Get bounding box of the contour
            
            # Add padding around the detected card
            padding = 10
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            return image[y_start:y_end, x_start:x_end]  # Return the cropped card image
        except Exception as e:
            logger.error(f"Error in card detection: {str(e)}")  # Log error if card detection fails
            raise

    def predict(self, image):
        """Predict card class from image."""
        try:
            processed_image, card_image = self.preprocess_image(image)  # Preprocess the input image
            predictions = self.model.predict(processed_image)  # Get model predictions
            predicted_class = np.argmax(predictions[0])  # Get the index of the class with the highest probability
            confidence = float(predictions[0][predicted_class])  # Get the confidence of the prediction
            
            result = {
                'class': self.class_labels[predicted_class],  # Map index to class label
                'confidence': confidence,  # Store confidence score
                'processed_image': card_image  # Store the processed card image for display
            }
            
            logger.info(f"Prediction: {result['class']} with confidence {confidence:.2f}")  # Log prediction details
            return result  # Return the prediction result
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")  # Log error if prediction fails
            raise

    def process_file(self, image_path):
        """Process image file for prediction."""
        try:
            image = cv2.imread(image_path)  # Read image from the specified file path
            if image is None:  # Check if the image was read successfully
                raise ValueError(f"Could not read image: {image_path}")  # Raise an error if not
            return self.predict(image)  # Return the prediction for the loaded image
        except Exception as e:
            logger.error(f"Error processing file {image_path}: {str(e)}")  # Log error if file processing fails
            raise

    def process_camera(self):
        """Process camera input for real-time card recognition."""
        try:
            cap = cv2.VideoCapture(0)  # Open the default camera
            if not cap.isOpened():  # Check if the camera opened successfully
                raise RuntimeError("Could not open camera")  # Raise error if not
            
            logger.info("Camera opened successfully")  # Log success message
            
            while True:  # Loop to continuously get frames from the camera
                ret, frame = cap.read()  # Read a frame from the camera
                if not ret:  # Check if the frame was grabbed successfully
                    logger.warning("Failed to grab frame")  # Log warning if frame retrieval fails
                    break  # Exit loop if not successful
                
                try:
                    result = self.predict(frame)  # Predict the card class from the frame
                    
                    # Draw the prediction text on the frame
                    cv2.putText(frame, f"{result['class']} ({result['confidence']:.2f})",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show processed card if available
                    if result['processed_image'] is not None:
                        cv2.imshow('Processed Card', result['processed_image'])
                    
                    # Show the main video frame with predictions
                    cv2.imshow('UNO Card Recognition', frame)
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")  # Log error if frame processing fails
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()  # Release the camera
            cv2.destroyAllWindows()  # Close all OpenCV windows
            
        except Exception as e:
            logger.error(f"Camera processing error: {str(e)}")  # Log error if camera processing fails
            raise

    def evaluate_model(self, test_dir):
        """Evaluate model performance on a test dataset."""
        try:
            predictions = []  # List to store predicted classes
            true_labels = []  # List to store true labels
            
            # Process each image in the test directory
            for class_name in os.listdir(test_dir):  # Loop through each class folder
                class_dir = os.path.join(test_dir, class_name)  # Get the full path to the class directory
                if not os.path.isdir(class_dir):  # Check if it is a directory
                    continue  # Skip if not
                
                for image_name in os.listdir(class_dir):  # Loop through each image in the class folder
                    image_path = os.path.join(class_dir, image_name)  # Get the full path to the image
                    try:
                        result = self.process_file(image_path)  # Process the image file for prediction
                        predictions.append(result['class'])  # Store predicted class
                        true_labels.append(class_name)  # Store true class label
                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {str(e)}")  # Log error if image processing fails
            
            # Calculate confusion matrix and classification report
            cm = confusion_matrix(true_labels, predictions, labels=self.class_labels)
            report = classification_report(true_labels, predictions, labels=self.class_labels)
            logger.info(f"Confusion Matrix:\n{cm}")  # Log confusion matrix
            logger.info(f"Classification Report:\n{report}")  # Log classification report
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_labels, yticklabels=self.class_labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()  # Display the plot
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")  # Log error if evaluation fails
            raise

if __name__ == "__main__":
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(description='UNO Card Recognition using CNN.')
    parser.add_argument('--model', type=str, default='card_recognition_model.h5', help='Path to the trained model file.')
    parser.add_argument('--image', type=str, help='Path to the image file for prediction.')
    parser.add_argument('--camera', action='store_true', help='Use camera for real-time prediction.')
    parser.add_argument('--test', type=str, help='Path to the test dataset directory for evaluation.')
    args = parser.parse_args()  # Parse command-line arguments

    # Initialize the UNO card recognizer
    recognizer = UnoCardRecognizer(model_path=args.model)  # Load the model

    # Run different modes based on the command-line arguments
    if args.image:  # If an image file is provided
        result = recognizer.process_file(args.image)  # Process the image file
        print(f"Predicted Class: {result['class']}, Confidence: {result['confidence']:.2f}")  # Print prediction result
    elif args.camera:  # If camera mode is specified
        recognizer.process_camera()  # Start camera processing
    elif args.test:  # If a test dataset directory is provided
        recognizer.evaluate_model(args.test)  # Evaluate the model on the test dataset
    else:
        print("Please provide an image file, use camera mode, or specify a test dataset.")  # Show usage message

 