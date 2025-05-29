# Age and Gender Detection Using OpenCV

This project demonstrates age and gender detection on images using OpenCV's deep learning (DNN) module and pre-trained models.

---

## Overview

- Detects faces in an image using a pre-trained face detection model.
- Predicts the age group and gender of each detected face using dedicated DNN models.
- Draws bounding boxes and labels on the image showing the predicted age and gender.

---

## Model Files

You need the following model files for the program to work. Place all of them in the same directory as the script:

| Model File                     | Description                     | Download Link                                            |
|-------------------------------|---------------------------------|----------------------------------------------------------|
| `opencv_face_detector.pbtxt`       | Face detection model config file | [Download Link](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt) |
| `opencv_face_detector_uint8.pb`     | Face detection model weights file | [Download Link](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel) |
| `age_deploy.prototxt`           | Age detection model config file | [Download Link](https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_deploy.prototxt) |
| `age_net.caffemodel`             | Age detection model weights file | [Download Link](https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_net.caffemodel) |
| `gender_deploy.prototxt`         | Gender detection model config file | [Download Link](https://github.com/spmallick/learnopencv/blob/master/AgeGender/gender_deploy.prototxt) |
| `gender_net.caffemodel`           | Gender detection model weights file | [Download Link](https://github.com/spmallick/learnopencv/blob/master/AgeGender/gender_net.caffemodel) |

> **Note:** Some links might point to `.caffemodel` or `.pb` files; ensure you download the correct files and rename if necessary.

---

## Setup and Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/age-gender-detection.git
   cd age-gender-detection

2. **Install required Python packages**
--- ```bash
pip install opencv-python matplotlib

3.  **Place the model files**
   Download all the required model files (see above) and place them in the project folder.


4.  **Prepare your input image**
  Update the `image_path` variable in `Age.py` with the full path to your input image.


5.  **Run the script**
--- ```bash
   python Age.py

6.  **View the results**
A window will pop up displaying the image with detected faces, predicted ages, and genders.
How It Works

## How It Works
Uses OpenCV DNN to load the face detection model and detect faces.
For each detected face, extracts the region of interest (ROI).
Passes the ROI through pre-trained age and gender models.
Annotates the image with bounding boxes and labels.


## Troubleshooting
Make sure all model files are in the correct directory and paths in the code are accurate.
Verify OpenCV version (4.2+ recommended).
If you encounter file not found errors, double-check model filenames and locations.
For any OpenCV DNN errors, verify model compatibility.
