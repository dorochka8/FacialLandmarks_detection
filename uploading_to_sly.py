import time

from main_functions import creating_a_sly_project, upload_imgs_sly, upload_landmarks_sly
from processing_dataset import valid_images, pred_landmarks, landmark_names

api_token = 'YOUR_API_TOKEN'
project_name = 'PROJECT_NAME'
dataset_name = 'DATASET_NAME'

api, workspace, project, dataset, dir = creating_a_sly_project(api_token, project_name, dataset_name)
start_idx, end_idx, step = 500, 1000, 50
img_names = upload_imgs_sly(api, dataset.id, valid_images, start_idx=start_idx, end_idx=end_idx, step=step, temp_dir='/mnt/data/datasets/DATASET_NAME')

print("Start manual annotation in Supervisely...")
start_time = time.time()
upload_landmarks_sly(api, dataset.id, pred_landmarks, img_names, landmark_names, start_idx, end_idx, step, time_per_image=True)
end_time = time.time()
total_annotation_time = end_time - start_time
print(f'Total annotation time: {total_annotation_time:.5f} seconds')