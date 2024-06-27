import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import os 
import random
import requests
import supervisely as sly
import time 

# landmarks on random data
def download_imgs(URLs):
  return [Image.open(requests.get(URL, stream=True).raw) for URL in URLs]

def show_examples(imgs, title, num_pic_to_show=4):
  fig, ax = plt.subplots(1, num_pic_to_show, figsize=(20, 5))
  fig.suptitle(title, fontsize=20)
  total_images = len(imgs) - 1

  for i in range(num_pic_to_show):
    ax[i].imshow(imgs[random.randint(0, total_images)])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
  plt.show()

  return 


def pil_to_cv2(img):
  return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detecting(img, detector, predictor):
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = detector(img_gray)
  landmarks = [predictor(img_gray, f) for f in face]
  return face, landmarks


def show_results(detector, predictor, imgs, num_samples_to_show=5):
  faces_landmarks = [detecting(pil_to_cv2(img), detector, predictor) for img in imgs]

  num_imgs = len(imgs)
  idxs_to_show = random.sample(range(num_imgs), k=num_samples_to_show)

  _, ax = plt.subplots(2, num_samples_to_show, figsize=(num_samples_to_show*10, 20))

  for i, idx in enumerate(idxs_to_show):
    img = imgs[idx]
    imgtocv2 = pil_to_cv2(img)
    face, landmarks = faces_landmarks[idx]

    for f, landmark in zip(face, landmarks):
      x, y, w, h = f.left(), f.top(), f.width(), f.height()
                          # start_pt,         end_pt,       color, thickness
      cv2.rectangle(imgtocv2, (x, y), (x + w, y + h), (0, 255, 0), 5)

      for j in range(68):
                                                        # center, radius, color, thickness
        cv2.circle(imgtocv2, (landmark.part(j).x, landmark.part(j).y), 10, (0, 0, 255), -1)

    if num_samples_to_show == 1:
      ax[0].imshow(img)
      ax[0].set_xticks([])
      ax[0].set_yticks([])

      ax[1].imshow(cv2.cvtColor(imgtocv2, cv2.COLOR_BGR2RGB))
      ax[1].set_xticks([])
      ax[1].set_yticks([])

    else:
      ax[0, i].imshow(img)
      ax[0, i].set_xticks([])
      ax[0, i].set_yticks([])

      ax[1, i].imshow(cv2.cvtColor(imgtocv2, cv2.COLOR_BGR2RGB))
      ax[1, i].set_xticks([])
      ax[1, i].set_yticks([])

  plt.show()

  return


# working with dataset
def detecting_from_dataset(img_bgr, detector, predictor):
  dets = detector(img_bgr, 1)
  if len(dets) > 0:
    shape = predictor(img_bgr, dets[0])
    return dets[0], shape
  return None, None


def detecting_landmarks(imgs, detector, predictor, ground_truth_landmarks, to_numpy=True):
  detected_landmarks = []
  valid_images = []
  valid_ground_truth_landmarks = []
  invalid_images = []

  for idx, img in enumerate(imgs):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, landmarks = detecting_from_dataset(img_bgr, detector, predictor)

    if landmarks:
      landmarks = [(p.x, p.y) for p in landmarks[0].parts()]
      detected_landmarks.append(landmarks)
      valid_images.append(img)
      valid_ground_truth_landmarks.append(ground_truth_landmarks[idx])
    else:
      invalid_images.append(img)

  if to_numpy:
    detected_landmarks = np.array(detected_landmarks)
    valid_images = np.array(valid_images)
    valid_ground_truth_landmarks = np.array(valid_ground_truth_landmarks)

  print(f'Detected landmarks shape: {detected_landmarks.shape}')
  print(f'Valid images shape: {valid_images.shape}')
  print(f'Valid ground truth shape: {valid_ground_truth_landmarks.shape}') 
  print(f'Invalid images: {len(invalid_images)}')

  return detected_landmarks, valid_images, valid_ground_truth_landmarks, invalid_images


def add_eye_centers(pred_landmarks, left_eye_inner_col=2, left_eye_outer_col=3, right_eye_inner_col=4, right_eye_outer_col=5):
  left_eye_center = (pred_landmarks[:, left_eye_inner_col] + pred_landmarks[:, left_eye_outer_col]) / 2
  right_eye_center = (pred_landmarks[:, right_eye_inner_col] + pred_landmarks[:, right_eye_outer_col]) / 2

  pred_landmarks[:, 0] = left_eye_center  # left_eye_center
  pred_landmarks[:, 1] = right_eye_center # right_eye_center

  return pred_landmarks


def calculate_metrics(ground_truth, preds, landmark_names):
  errors = []
  ground_truth_valid_list = []
  preds_valid_list = []

  per_landmark_errors = [[] for _ in range(len(landmark_names))]

  for i in range(ground_truth.shape[0]):
    valid_pts = ~np.isnan(ground_truth[i, :, 0]) & ~np.isnan(preds[i, :, 0])
    ground_truth_valid = ground_truth[i][valid_pts]
    preds_valid = preds[i][valid_pts]
    ground_truth_valid_list.append(ground_truth_valid)
    preds_valid_list.append(preds_valid)

    if ground_truth_valid.shape[0] > 0:
      error = np.linalg.norm(ground_truth_valid - preds_valid, axis=1)
      errors.append(np.mean(error))

      for j, (gt, pred) in enumerate(zip(ground_truth_valid, preds_valid)):
        landmark_error = np.linalg.norm(gt - pred)
        per_landmark_errors[j].append(landmark_error)

  mean_errors = np.array(errors)
  per_landmark_positioning_error = [np.mean(err) if err else np.nan for err in per_landmark_errors]
  overall_detection_accuracy = np.mean([~np.isnan(preds[:, j, 0]).sum() / len(preds[:, j, 0]) for j in range(len(landmark_names))])
  overall_positioning_error = np.nanmean(per_landmark_positioning_error)

  metrics = {'mean_errors': mean_errors,
             'per_landmark_positioning_error': dict(zip(landmark_names, per_landmark_positioning_error)),
             'overall_detection_accuracy': overall_detection_accuracy,
             'overall_positioning_error': overall_positioning_error,
             'ground_truth_valid_list': ground_truth_valid_list,
             'preds_valid_list': preds_valid_list
             }

  return metrics


def plot_imgs_w_landmarks(imgs, ground_truth, preds, idxs, title, mean_errors):
  fig, ax = plt.subplots(1, len(idxs), figsize=(20, len(idxs)))
  fig.suptitle(title)

  for i, idx in enumerate(idxs):
    img = imgs[idx]
    gt = ground_truth[idx]
    pred = preds[idx]

    ax[i].imshow(img, cmap='gray')
    valid_pts  = ~np.isnan(gt[:, 0])
    gt_valid   = gt[valid_pts]
    pred_valid = pred[valid_pts]

    if gt_valid.shape[0] > 0:
      ax[i].scatter(gt_valid[:, 0], gt_valid[:, 1], s=50, label='Ground truth')
      ax[i].scatter(pred_valid[:, 0], pred_valid[:, 1], s=50, label='Predinction')

      for (gt_x, gt_y), (pred_x, pred_y) in zip(gt_valid, pred_valid):
        ax[i].plot([gt_x, pred_x], [gt_y, pred_y], c='green', linewidth=1.5)

    ax[i].set_title(f'Error: {mean_errors[idx]:.3f}')

    if i == 0:
      ax[i].legend()

  plt.show()


def plot_imgs_wo_landmarks(imgs, title, num_imgs_to_plot=5):
  fig, ax = plt.subplots(1, 5, figsize=(20, num_imgs_to_plot))
  fig.suptitle(title)

  for i, img in enumerate(imgs[:num_imgs_to_plot]):
    ax[i].imshow(img, cmap='gray')
    ax[i].set_title(f'No landmarks detected')

  plt.show()


def plot_metrics(metrics):
  mean_errors = metrics['mean_errors']
  per_landmark_positioning_error = metrics['per_landmark_positioning_error']

  _, ax = plt.subplots(1, 2, figsize=(20, 10))

  ax[0].hist(mean_errors, bins=10, alpha=0.7)
  ax[0].set_title('Histogram of mean errors')
  ax[0].set_xlabel('Mean error')
  ax[0].set_ylabel('Frequency')

  landmarks = list(per_landmark_positioning_error.keys())
  errors = list(per_landmark_positioning_error.values())
  ax[1].bar(landmarks, errors, alpha=0.7)
  ax[1].set_title('Per-landmark positioning error')
  ax[1].set_xlabel('Landmarks')
  ax[1].set_ylabel('Positioning error')
  ax[1].set_xticks(range(len(landmarks)))
  ax[1].set_xticklabels(landmarks, rotation=75)

  plt.tight_layout()
  plt.show()


# uploading to sly
def creating_a_sly_project(api, project_name, dataset_name, make_temp_dir=True):
  api = sly.Api('https://app.supervisely.com/', api)

  workspace = api.workspace.get_list(api.team.get_list()[0].id)[0]
  print(workspace)

  project = api.project.create(workspace.id, project_name, change_name_if_conflict=True)
  dataset = api.dataset.create(project.id,   dataset_name, change_name_if_conflict=True)
  print(f'Project #{project.id} with dataset #{dataset.id} created')

  if make_temp_dir:
    temp_dir = '/mnt/data/temp_images'
    os.makedirs(temp_dir, exist_ok=True)

  return api, workspace, project, dataset, temp_dir if make_temp_dir else None


def upload_imgs_sly(api, dataset_id, data, start_idx=0, end_idx=10, step=1, temp_dir='/mnt/data/temp_images'):
  img_dir  = os.path.join(temp_dir, 'images')
  os.makedirs(img_dir, exist_ok=True)

  img_paths = []
  for i, img_arr in enumerate(data[start_idx:end_idx:step]):
    img_path  = os.path.join(img_dir,  f'image_{i+1}.png')
    plt.imsave(img_path, img_arr, cmap='gray')
    img_paths.append(img_path)

  img_names = [os.path.basename(path) for path in img_paths]
  api.image.upload_paths(dataset_id, img_names, img_paths)
  print(f'Uploaded {len(img_paths)} images successfully')

  return img_names


def upload_landmarks_sly(api, dataset_id, preds, img_names, landmark_names, start_idx, end_idx, step, time_per_image=True):
  obj_classes = [sly.ObjClass(name, sly.Point) for name in landmark_names]
  dataset_info = api.dataset.get_info_by_id(dataset_id)

  project_id = dataset_info.project_id
  meta_json = api.project.get_meta(project_id)
  meta = sly.ProjectMeta.from_json(meta_json)

  new_classes = [obj_class for obj_class in obj_classes if obj_class.name not in [cls.name for cls in meta.obj_classes]]
  if new_classes:
    for obj_class in new_classes:
      meta = meta.add_obj_class(obj_class)
    api.project.update_meta(project_id, meta.to_json())
    print(f'Added new {len(new_classes)} object classes to project metadata')

  for img_name, pred_coords in zip(img_names, preds[start_idx:end_idx:step]):
    if time_per_image:
      start = time.time()
    img_info = api.image.get_info_by_name(dataset_id, img_name)
    img_size = (img_info.height, img_info.width)
    ann = sly.Annotation(img_size)

    print(f'--- Processing img {img_name} with size {img_size} ---')
    for j, _ in enumerate(landmark_names):
      x, y = float(pred_coords[j][0]), float(pred_coords[j][1])
      pt = sly.Point(row=y, col=x)
      label = sly.Label(pt, obj_classes[j])
      ann = ann.add_label(label)
      print(f'Added point ({x}, {y}) as {obj_classes[j].name} to annotation')

    # unmute in case to read the annotation for each image
    # ann_json = ann.to_json()
    # print(f'annotation for image {img_name}: {ann_json}')

    api.annotation.upload_ann(img_info.id, ann)
    if time_per_image:
      end = time.time()

    print(f'--- Uploaded preds for image {img_name} successfully ---')
    if time_per_image:
      print(f'-s-s- TIME SPENT on this image sample: {end - start:.5f} seconds -s-s-')
      
  print('\nUploaded all annotations')

  return