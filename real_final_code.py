import cv2
import numpy as np
from cvlib.object_detection import detect_common_objects, draw_bbox
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text, draw_bounding_box, apply_offsets
from utils.preprocessor import preprocess_input

# Load the emotion model and labels
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# starting lists for calculating modes
emotion_window = []


def process_image(bgr_image):
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Initialize output image
    out = bgr_image.copy()

    # Detect common objects
    bbox, label, conf = detect_common_objects(bgr_image)

    # Draw bounding box over detected objects
    out = draw_bbox(out, bbox, label, conf, write_conf=True)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # Color mapping based on emotion
        color = None
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        # Draw bounding box only when a person is detected in the image
        draw_bounding_box(face_coordinates, out, color)
        draw_text(face_coordinates, out, emotion_mode, color, 0, -45, 1, 1)

    # Display the result
    cv2.imshow('Output', out)
    cv2.waitKey(0)


# Use the provided image
image_path = './part 2/women_book.jpg'

if image_path:
    # Load the image
    bgr_image = cv2.imread(image_path)
    process_image(bgr_image)
else:
    print("Error: Image path not provided.")