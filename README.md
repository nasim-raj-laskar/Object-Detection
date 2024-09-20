# Object Detection using SSD MobileNet V3
## Project Overview
This project implements object detection using the SSD MobileNet V3 model, trained on the COCO dataset, capable of detecting objects in both static images and real-time video streams from a webcam. The project is optimized for detecting everyday objects such as people, cars, animals, and more, using the lightweight SSD MobileNet V3 architecture.

## It features:

- Object detection from static images or video files.
- Real-time object detection using a webcam, displaying live annotated results with bounding boxes and class labels.


## Features
- Static & Real-time Detection: Detect objects in images, video files, or directly from a webcam.
- Efficient Model: SSD MobileNet V3 is lightweight yet accurate, suitable for real-time performance.
- 80 Classes Supported: Detect everyday objects from the COCO dataset, including people, animals, vehicles, and more.
- OpenCV Integration: Built using OpenCV's DNN module for easy integration and fast inference.


## Requirements and Installation

### Prerequisites

Make sure your system meets the following prerequisites before running the project:

- Operating System: Any OS with Python support (Windows, macOS, Linux)
- Python Version: 3.6 or above
- Jupyter Notebook: Required to run the .ipynb file.

### Python Libraries
To run the project, you need to install the following libraries:

- OpenCV: Open-source computer vision library for object detection.
- Matplotlib: Used for visualization of detected objects.
Install the dependencies by running the following commands:
```
pip install opencv-python-headless matplotlib
```
If you don't have Jupyter Notebook installed, you can install it using:
```
pip install notebook
```

### Directory Structure
You need to place the following files in the same directory:

```
/Object-Detection
    ├── obj-dect.ipynb                                 # The main Jupyter notebook
    ├── obj-dect-real-time.ipynb                       # Main Jupyter notebook for real time video testing
    ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt   # Model configuration file
    ├──frozen_inference_graph.pb                       # Pre-trained model file
    ├──bmw-m4.jpg                                      # sample image
    ├──1721294-hd_1920_1080_25fps.mp4                  #sample video
    └──Labels.txt                                      # COCO class labels
    
```
Make sure to download or move the required files into the project folder to ensure smooth execution.

## Step-by-Step Guide
### 1. Cloning the Repository
First, clone the repository to your local machine or download the zip file.
```
git clone https://github.com/your-username/Object-Detection.git
cd Object-Detection
```
### 2. Running the Notebook
To run the object detection model, follow these steps:

1.Open the Jupyter Notebook in the terminal or command prompt:
```
jupyter notebook
```
2.Navigate to the directory where `the obj-dect.ipynb`  & `obj-dect-real-time.ipynb` file is located.

3.Open the notebook and run the cells in sequence.
### 3. Model Description

This project uses SSD MobileNet V3 from TensorFlow’s object detection model zoo, pre-trained on the COCO dataset.

- SSD (Single Shot Multibox Detector): Efficient object detection algorithm.
- MobileNetV3: Lightweight model architecture ideal for mobile and embedded vision applications.
- COCO dataset: Common dataset with 80 classes, including everyday objects like person, bicycle, car, etc.

### 4. Using the Model

- Loading the Model: The model and its configuration file are loaded using OpenCV's `cv2.dnn_DetectionModel`.
- Labeling Objects: The `Labels.txt` file contains the class names corresponding to the detected objects.
- Visualizing Results: Detected objects are annotated with bounding boxes, class names, and confidence scores.
```
import cv2
import matplotlib.pyplot as plt


# Load model and config
model = cv2.dnn_DetectionModel('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Load class labels
classLabels = []
with open('Labels.txt', 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
```

## Sample Output
When you run the project, you will see images or video frames with bounding boxes drawn around the detected objects. The label of the detected object and its confidence score will also be displayed.

Sample output from object detection:

![Object Detection Sample](img/output-img.png)




## Customization
### 1. Use your own images or videos
You can replace the image or video file paths in the notebook to detect objects in your custom media:
```
img = cv2.imread('your-image.jpg')
video = cv2.VideoCapture('your-video.mp4')
```
### 2. Adjust Detection Parameters
- You can tweak parameters like confidence threshold or model input size to balance accuracy and performance for your specific application.

```
model.setInputSize(300, 300)  # Adjust for speed vs accuracy
model.setInputSwapRB(True)    # Adjust color channel swapping
```
- Adjusting the Confidece Threshold
Modify the confThreshold parameter in the model.detect() function to adjust the detection sensitivity. Higher values will reduce false positives, but may also miss some objects:
```
ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
```
### 3.Changing Webcam Feed
If you have multiple webcams, you can change the camera input by updating the `cv2.VideoCapture()` function:
```
cap = cv2.VideoCapture(1)  # For another webcam
```
## Troubleshooting

- 1.Model not found error: Ensure that the `frozen_inference_graph.pb` , `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` and `Labels.txt`files are in the same directory as the notebook.
- 2.Incorrect paths: Double-check file paths for images, videos, and model files.
- 3.Missing dependencies: Ensure all required Python libraries are installed (use pip list to verify).
- 4.Webcam not opening: Ensure your webcam is functioning properly and that the correct device index is used (`0` for the default webcam).

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests. You can contribute by:

- Adding more sample images or videos for testing.
- Improving the detection accuracy by fine-tuning the model.
- Extending the project to detect objects in real-time via a webcam or live video feed.
