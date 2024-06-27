import dlib
import numpy as np
import pandas as pd 

from main_functions import detecting_landmarks, add_eye_centers, calculate_metrics, plot_imgs_w_landmarks, plot_imgs_wo_landmarks, plot_metrics

data = pd.read_csv('training.csv')
print(data.head(3))
print(data.info())
print(data.isna().sum())

shape_of_an_image = int(np.fromstring(data['Image'][0], sep=' ').shape[0] ** 0.5)
data['Image'] = data['Image'].apply(lambda x: np.fromstring(x, sep=' ').reshape(shape_of_an_image, shape_of_an_image).astype(np.uint8))
dataset_imgs = np.stack(data['Image'].values)
ground_truth_landmarks = data.drop(columns='Image', axis=1).values.reshape(-1, 15, 2)

relevant_idx = [0,  # placeholder for left_eye_center
                0,  # placeholder for right_eye_center
                42, # left_eye_inner_corner
                45, # left_eye_outer_corner
                39, # right_eye_inner_corner
                36, # right_eye_outer_corner

                22, # left_eyebrow_inner_end
                26, # left_eyebrow_outer_end
                21, # right_eyebrow_inner_end
                17, # right_eyebrow_outer_end

                30, # nose_tip

                54, # mouth_left_corner
                48, # mouth_right_corner
                51, # mouth_center_top_lip
                57, # mouth_center_bottom_lip
]
landmark_names = ['left_eye_center', 'right_eye_center',
                  'left_eye_inner_corner', 'left_eye_outer_corner',
                  'right_eye_inner_corner', 'right_eye_outer_corner',
                  'left_eyebrow_inner_end', 'left_eyebrow_outer_end',
                  'right_eyebrow_inner_end', 'right_eyebrow_outer_end',
                  'nose_tip', 'mouth_left_corner', 'mouth_right_corner',
                  'mouth_center_top_lip', 'mouth_center_bottom_lip'
]


detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detected_landmarks, valid_images, valid_ground_truth_landmarks, invalid_images = detecting_landmarks(dataset_imgs, detector, predictor)

pred_landmarks = detected_landmarks[:, relevant_idx]
pred_landmarks = add_eye_centers(pred_landmarks)

metrics = calculate_metrics(valid_ground_truth_landmarks, pred_landmarks, landmark_names)

num_images_to_display = 5
best_indices = np.argsort(metrics['mean_errors'])[:num_images_to_display]
worst_indices = np.argsort(metrics['mean_errors'])[-num_images_to_display:]

plot_imgs_w_landmarks(valid_images, metrics['ground_truth_valid_list'], metrics['preds_valid_list'], best_indices,  'Best detections', metrics['mean_errors'])
plot_imgs_w_landmarks(valid_images, metrics['ground_truth_valid_list'], metrics['preds_valid_list'], worst_indices, 'Worst detections', metrics['mean_errors'])
plot_imgs_wo_landmarks(invalid_images, 'Images with no detected landmarks', num_imgs_to_plot=num_images_to_display)
plot_metrics(metrics, landmark_names)
