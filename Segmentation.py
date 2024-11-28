import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, morphology
from keras.models import load_model
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from skimage.segmentation import clear_border

# Function to display images
def ShowImageWithPrediction(title, img, prediction_text=None):
    plt.figure(figsize=(10, 10))
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
    else:
        rgb_img = img
        plt.gray()
    if prediction_text:
        text_position = (10, int(img.shape[0] * 0.1))
        cv2.putText(rgb_img, prediction_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title(title)
    plt.show()

# Load a pre-trained model for classification
model_path = r'D:\New folder (2)\brain_tumor_detector.h5'
model = load_model(model_path)

# Transfer learning model for improved classification
classification_model = ResNet50(weights='imagenet', include_top=False, input_shape=(240, 240, 3))

# Load and process the image
img_path = r'D:\New folder (2)\brain_tumor_dataset\no\9 no.jpg'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found. Please check the path: {img_path}")

# Convert image to grayscale and enhance contrast
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = exposure.equalize_hist(gray) * 255
gray = gray.astype(np.uint8)

# Apply Gaussian Blur
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
ShowImageWithPrediction('Enhanced Grayscale Brain MRI', gray_blur)

# Apply Otsu's thresholding and morphological operations
ret, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = clear_border(thresh)
morph = morphology.remove_small_objects(thresh.astype(bool), min_size=100).astype(np.uint8) * 255
ShowImageWithPrediction('Refined Segmentation', morph)

# Apply U-Net or advanced segmentation model if available (Placeholder here)
# For better accuracy, a pre-trained U-Net model trained on similar medical images should be used.

# Mask the brain region
brain_mask = np.zeros_like(gray)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    cv2.drawContours(brain_mask, contours, -1, 255, thickness=cv2.FILLED)
brain_out = cv2.bitwise_and(img, img, mask=brain_mask)
ShowImageWithPrediction('Segmented Brain Region', brain_out)

# Preprocess the image for classification
input_image = cv2.resize(brain_out, (240, 240))
input_image = img_to_array(input_image)
input_image = preprocess_input(input_image)
input_image = np.expand_dims(input_image, axis=0)

# Classify the image
prediction = model.predict(input_image)
predicted_class = 'Tumor Detected' if prediction[0][0] > 0.5 else 'No Tumor'
confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

# Display the prediction and confidence
print(f'Prediction: {predicted_class} (Confidence: {confidence * 100:.2f}%)')
prediction_text = f'{predicted_class} ({confidence * 100:.2f}%)'
ShowImageWithPrediction('Final Brain Region with Prediction', brain_out, prediction_text)
