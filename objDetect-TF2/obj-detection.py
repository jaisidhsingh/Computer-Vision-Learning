import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import numpy as np
from PIL import Image


#gpu memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')	
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)


# store the asset paths
IMAGE_PATHS = [r'D:\Local Code\heavy models\obj_det_assets\image1.jpg', r'D:\Local Code\heavy models\obj_det_assets\image2.jpg']

PATH_TO_LABELS = r"D:\Local Code\heavy models\obj_det_assets\mscoco_label_map.pbtxt"
PATH_TO_MODEL_DIR = r'D:\Local Code\heavy models\faster_rcnn_nas_coco_24_10_2017'

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + r"\saved_model"
print('Loading model...', end='')
start_time = time.time()

# load saved model and build detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

print(list(detect_fn.signatures.keys())) #['serving_default']

infer = detect_fn.signatures['serving_default']
end_time = time.time()
eta = end_time - start_time
print(f"took {eta} seconds")

cat_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# lets start on the good stuff now

def load_img_into_nparray(path):
	return np.array(Image.open(path))

for image_path in IMAGE_PATHS:
	print("inferencing for {}....".format(image_path), end='')

	image_np = load_img_into_nparray(image_path)

	input_tensor = tf.convert_to_tensor(image_np)
	input_tensor = input_tensor[tf.newaxis,...]

	detections = infer(input_tensor)

	num_detections = int(detections.pop('num_detections'))
	detections = {key: value[0, :num_detections].numpy()
				  for key, value in detections.items()}

	detections['num_detections'] = num_detections
	detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
	image_np_with_detections = image_np.copy()

	
	viz_utils.visualize_boxes_and_labels_on_image_array(
		image_np_with_detections,
		detections['detection_boxes'], 
		detections['detection_classes'], 
		detections['detection_scores'], 
		cat_index, 
		use_normalized_coordinates=True, 
		max_boxes_to_draw=200, 
		min_score_thresh=0.30, 
		agnostic_mode=False)
	img = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)
	cv2.imshow('object-detection', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("done")    


# ASSET DOWNLOAD FUNCTIONS : 

# def download_images():
# 	base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
# 	filenames = ['image1.jpg', 'image2.jpg']
# 	image_paths = []
# 	for filename in filenames:
# 		image_path = tf.keras.utils.get_file(fname=filename, origin=base_url+filename, untar=False)
# 		image_path = pathlib.Path(image_path)
# 		image_paths.append(str(image_path))
# 	return image_paths


# def download_labels(filename):
#     base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
#     label_dir = tf.keras.utils.get_file(fname=filename,
#                                         origin=base_url + filename,
#                                         untar=False)
#     label_dir = pathlib.Path(label_dir)
#     return str(label_dir)


# def download_model(model_name, model_date):
#     base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
#     model_file = model_name + '.tar.gz'
#     model_dir = tf.keras.utils.get_file(fname=model_name,
#                                         origin=base_url + model_date + '/' + model_file,
#                                         untar=True)
#     return str(model_dir)