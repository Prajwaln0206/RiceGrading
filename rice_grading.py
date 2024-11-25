import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("rice_classifier.h5")
print("Model loaded successfully.")

categories = ["High-Quality Rice", "Low-Quality Rice", "Stone"]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not loaded. Check the file path.")
        exit()
    image_resized = cv2.resize(image, (64, 64))
    image_normalized = image_resized / 255.0
    return image, np.expand_dims(image_normalized, axis=0)

def segment_and_classify(image_path):
    original_image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)

    cv2.imwrite("binary_debug.jpg", binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours detected: {len(contours)}")

    output_image = original_image.copy()
    count_classes = {"High-Quality Rice": 0, "Low-Quality Rice": 0, "Stone": 0}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:  # Adjust minimum contour area
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi = original_image[y:y + h, x:x + w]

        roi_resized = cv2.resize(roi, (64, 64))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.expand_dims(roi_normalized, axis=0)

        prediction = model.predict(roi_reshaped)
        class_index = np.argmax(prediction)
        label = categories[class_index]
        count_classes[label] += 1

        print(f"Detected contour at ({x}, {y}), Prediction: {label}, Probabilities: {prediction}")

        color = (0, 255, 0) if label == "High-Quality Rice" else (0, 0, 255)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print(f"Final Count: {count_classes}")
    cv2.imwrite("processed_output.jpg", output_image)

def main():
    image_path = "test_image.jpg"
    segment_and_classify(image_path)

if __name__ == "__main__":
    main()
