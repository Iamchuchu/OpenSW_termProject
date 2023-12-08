# Analyzes and visually displays facial emotions in real time with Open CV

The code is a program that reads images from a webcam or video file, detects and visually displays emotions around your face by detecting faces and objects. The code is written in Python's language, using different libraries and modules.

김단아, 김예서, and 노동훈 are in charge of code production, and 추성윤 manages GitHub

---

## Code Step by Step
### 1. Importing Libraries:

- 'cv2': OpenCV library for computer vision.
- 'numpy' (as np): Library for numerical operations.
- 'cvlib': A simple library for common computer vision tasks built on top of OpenCV.
- 'keras': Deep learning library.
- 'statistics': Library for statistical operations.
- Various utility functions from a custom module (probably created by the user), including functions for handling datasets, inference, and preprocessing.

### 2. Setting Configuration:

- 'USE_WEBCAM': Boolean flag to determine whether to use a webcam ('True') or a video file ('False').
- 'emotion_model_path': Path to the pre-trained emotion detection model.
- 'emotion_labels': Labels for different emotions from the dataset.
- 'frame_window': Number of frames to consider for calculating the mode of detected emotions.
- 'emotion_offsets': Offsets for the region of interest around detected faces.

### 3. Loading Models:

- Loading the Haar Cascade classifier for face detection ('face_cascade').
- Loading the pre-trained emotion detection model using Keras ('emotion_classifier').

### 4. Setting up Video Capture:

- Creating a named window for displaying the video feed.
- Opening a video capture object ('video_capture') based on whether a webcam or a video file is to be used.

### 5. Main Loop:

- A loop that captures frames from the video source until it's closed.
- Reads a frame from the video source ('cap.read()').
- Converts the frame to grayscale and RGB formats.
- Performs common object detection using 'cv.detect_common_objects'.
- Draws bounding boxes around detected objects.
- Detects faces using the Haar Cascade classifier.
- For each detected face:
  - Applies offsets to define the region of interest for emotion detection.
  - Preprocesses the face image for the emotion detection model.
  - Makes predictions using the emotion detection model.
  - Updates a window of detected emotions.
  - Calculates the mode of the detected emotions in the window.
  - Determines a color based on the predicted emotion.
  - Draws bounding boxes and text on the output frame.
- Combines the original frame and the frame with bounding boxes and emotion labels.
- Displays the combined frame.
- The loop continues until the user presses 'q'.

### 6. Cleanup:

Releases the video capture object and closes all windows when the loop is exited.


#### Note
The code assumes the existence of certain utility functions and model files (like Haar Cascade file, emotion model file, etc.) that are not provided in the code snippet. Ensure that these dependencies are satisfied for the code to run successfully.

## Requirements
1. Python
2. opencv-python
3. keras
4. tensorflow
5. numpy
6. scipy
7. pillow
8. pandas
9. matplotlib
10. h5py

## Result
