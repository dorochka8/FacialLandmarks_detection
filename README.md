# Facial Landmarks Detection

This repository contains code for evaluating the performance of facial landmarks detection with [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) model. The code calculates various metrics, such as mean errors, per-landmark positioning errors, overall detection accuracy, and overall positioning error, to assess the accuracy and positioning of detected landmarks and provides visualizations to help understand the model's performance.

One can use the provided functions to calculate metrics, visualize predictions, and plot metrics for facial landmarks detection model. It also provides visualizations to help understand the model's performance by plotting images with ground truth and predicted landmarks, highlighting errors, and showing the best and worst predictions. Additionally, the code generates a histogram of mean errors and a bar chart of per-landmark positioning errors for comprehensive analysis. Example usage is provided to demonstrate how to integrate and utilize these functions effectively. The entire notebook is provided in the `whole_notebook.ipynb` file.

## Model parameters
Install required libraries:
```
pip install opencv-python dlib supervisely
```

Make sure to upload the `shape_predictor_68_face_landmarks.dat` model.
```
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
```
## Landmarks on random data
Examples of provided URLs: \
![Unknown](https://github.com/dorochka8/FacialLandmarks_detection/assets/97133490/9c0aa89f-84ae-4426-8d3e-cbc3c91f3974)

Results of evaluating `shape_predictor_68_face_landmarks.dat` predictor:\
![Unknown-2](https://github.com/dorochka8/FacialLandmarks_detection/assets/97133490/fc9402bc-d220-400a-8d72-4771625ec5ce)

## Processing dataset
The dataset is way to big. \
Directly download `training.csv` from [Kaggle/Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/data?select=training.zip).

Best detections, predicted with `shape_predictor_68_face_landmarks.dat` model:\
![3983e62f-d47f-4ee3-850d-2ddfef5bfabc](https://github.com/dorochka8/FacialLandmarks_detection/assets/97133490/c61081bb-af46-4140-bd42-9b9eb012fd91)

Worst detections, predicted with `shape_predictor_68_face_landmarks.dat` model:\
![e2b4ad19-0187-4405-8f49-fe4f0107a958](https://github.com/dorochka8/FacialLandmarks_detection/assets/97133490/29f5f62e-c2e1-4b3f-938b-ae42fb880a5d)

Not detected landmarks, fails: \
![e3e49fc8-063d-4639-b6da-a3f7a9e944a4](https://github.com/dorochka8/FacialLandmarks_detection/assets/97133490/fc75c90b-71eb-4e12-a23f-465f8c219e3c)

Metrics:\
![7cccedc6-a881-4135-9b66-2c127a7bc393](https://github.com/dorochka8/FacialLandmarks_detection/assets/97133490/d74776e0-0f9e-4df7-81aa-bb68e065468c)


