#cd into the workspace directory to run this script

import tensorflow as tf
import numpy as np
from PIL import Image
import PIL 
import PIL.Image
import matplotlib.pyplot as plt
import sys
import os
import cv2
import io

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

os.chdir('..')
path2scripts = './models/research'
os.chdir('workspace')
sys.path.insert(0, path2scripts)

class buttonDetector():
    
    def __init__(self):
        self.path2config ='./exported_models/d0/v3/pipeline.config'
        self.path2model = './exported_models/d0/v3/checkpoint/'
        # self.detection_model = None
        self.path2label_map = './data/button_label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(self.path2label_map,use_display_name=True)
        
        configs = config_util.get_configs_from_pipeline_file(self.path2config) # importing config
        model_config = configs['model'] # recreating model config
        self.detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(self.path2model, 'ckpt-0')).expect_partial()

        print('Button detector initialized')
        
    def detect_fn(self, image):
        """
        Detect objects in image.
        
        Args:
        image: (tf.tensor): 4D input image
        
        Returs:
        detections (dict): predictions that model made
        """

        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        
        return detections
    
    def load_image_into_numpy_array(self, path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
            path: the file path to the image

        Returns:
            numpy array with shape (img_height, img_width, 3)
        """
        
        return np.array(Image.open(path))
    
    def nms(self, rects, thd=0.5):
        """
        Filter rectangles
        rects is array of oblects ([x1,y1,x2,y2], confidence, class)
        thd - intersection threshold (intersection divides min square of rectange)
        """
        out = []

        remove = [False] * len(rects)

        for i in range(0, len(rects) - 1):
            if remove[i]:
                continue
            inter = [0.0] * len(rects)
            for j in range(i, len(rects)):
                if remove[j]:
                    continue
                inter[j] = self.intersection(rects[i][0], rects[j][0]) / min(self.square(rects[i][0]), self.square(rects[j][0]))

            max_prob = 0.0
            max_idx = 0
            for k in range(i, len(rects)):
                if inter[k] >= thd:
                    if rects[k][1] > max_prob:
                        max_prob = rects[k][1]
                        max_idx = k

            for k in range(i, len(rects)):
                if (inter[k] >= thd) & (k != max_idx):
                    remove[k] = True

        for k in range(0, len(rects)):
            if not remove[k]:
                out.append(rects[k])

        boxes = [box[0] for box in out]
        scores = [score[1] for score in out]
        classes = [cls[2] for cls in out]
        return boxes, scores, classes


    def intersection(self, rect1, rect2):
        """
        Calculates square of intersection of two rectangles
        rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
        return: square of intersection
        """
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
        overlapArea = x_overlap * y_overlap
        return overlapArea


    def square(self, rect):
        """
        Calculates square of rectangle
        """
        return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])
    
    def button_candidates(self, boxes, scores, img_path):
    
        with open(img_path, 'rb') as f:
                image = np.asarray(PIL.Image.open(io.BytesIO(f.read())))
        img_height = image.shape[0]
        img_width = image.shape[1]

        button_scores = [] #stores the score of each button (confidence)
        button_patches = [] #stores the cropped image that encloses the button
        button_positions = [] #stores the coordinates of the bounding box on buttons

        for box, score in zip(boxes, scores):
            if score < 0.5: continue

            y_min = int(box[0] * img_height)
            x_min = int(box[1] * img_width)
            y_max = int(box[2] * img_height)
            x_max = int(box[3] * img_width)

            button_patch = image[y_min: y_max, x_min: x_max]
            button_patch = cv2.resize(button_patch, (180, 180))

            button_scores.append(score)
            button_patches.append(button_patch)
            button_positions.append([x_min, y_min, x_max, y_max])
            
        return button_patches, button_positions, button_scores


    def detect_buttons(self, image_path, box_th = 0.5,nms_th = 0.5):
        
        # print("Start")
        
        """
        Function that performs inference and return filtered predictions and prediction detections
        
        Args:
        path2images: a pathe to image
        box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
        nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
        to_file: (boolean). When passed as True => results are saved into a file. Writing format is
        path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
        data: (str) name of the dataset you passed in (e.g. test/validation)
        path2dir: (str). Should be passed if path2images has only basenames. If full pathes provided => set False.
        
        Returs:
        detections (dict): filtered predictions that model made
        """
        
            
        image_np = self.load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        # checking how many detections we got
        num_detections = int(detections.pop('num_detections'))
        
        # filtering out detection in order to get only the one that are indeed detections
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        # defining what we need from the resulting detection dict that we got from model output
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
        
        # filtering out detection dict in order to get only boxes, classes and scores
        detections = {key: value for key, value in detections.items() if key in key_of_interest}
        
        # print('detections processed')
        if box_th: # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > box_th]
                detections[key] = filtered_current_array
        
        # print('box_th done')
        if nms_th: # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as
            output_info = list(zip(detections['detection_boxes'],
                                    detections['detection_scores'],
                                    detections['detection_classes']
                                    )
                                )
            boxes, scores, classes = self.nms(output_info)
            
            detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
            detections['detection_scores'] = scores
            detections['detection_classes'] = classes
        
        # print('nms_th done')
       
        return detections