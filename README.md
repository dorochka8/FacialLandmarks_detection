# Facial Landmarks Detection

This repository contains code for evaluating the performance of facial landmarks detection model. The code calculates various metrics, such as mean errors, per-landmark positioning errors, overall detection accuracy, and overall positioning error, to assess the accuracy and positioning of detected landmarks and provides visualizations to help understand the model's performance.


One can use the provided functions to calculate metrics, visualize predictions, and plot metrics for facial landmarks detection model. It also provides visualizations to help understand the model's performance by plotting images with ground truth and predicted landmarks, highlighting errors, and showing the best and worst predictions. Additionally, the code generates a histogram of mean errors and a bar chart of per-landmark positioning errors for comprehensive analysis. Example usage is provided to demonstrate how to integrate and utilize these functions effectively.


## Model parameters
Install required libraries:
```
pip install opencv-python dlib
```

Make sure to upload the `shape_predictor_68_face_landmarks.dat` model.
```
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
```
## Landmarks on random data
Examples of provided URLs:

![cbd0fd67-74a4-460d-9104-38d1f1de4acb](https://github.com/dorochka8/FacialLandmarks_detection/assets/97133490/bab080d5-d3aa-4802-831a-3dbebcc419aa)


Results of evaluating `shape_predictor_68_face_landmarks.dat` predictor:

![ddc886d1-18cc-44df-8fc7-e70127a1e0bb-1](https://github.com/dorochka8/FacialLandmarks_detection/assets/97133490/abf1e1cc-dc77-4f31-b5dd-0cb18c902b64)
