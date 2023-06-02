import os
from button_labelling import characterRecognizer
from button_detection import buttonDetector
import tqdm
import numpy as np
import cv2

path2images = []
test_panels = './data/test_panels/'
for filename in os.listdir(test_panels):
    file_path = os.path.join(test_panels, filename)
    path2images.append(file_path)   

detector = buttonDetector()
recognizer = characterRecognizer(verbose=False)
overall_det_times = []
overall_lbl_times = []

for image_path in tqdm(path2images):
    time_det, dets = detector.detect_buttons(image_path)
    boxes, scores, classes = dets['detection_boxes'], dets['detection_scores'], dets['detection_classes']
    boxes, scores, classes = [np.squeeze(x) for x in [boxes, scores, classes]]

    button_patches, button_positions, _ = detector.button_candidates(boxes, scores, image_path)
    t0 = cv2.getTickCount()
    for button_imgs in button_patches:
        button_text, button_score, _ = recognizer.predict(button_imgs)
    t1 = cv2.getTickCount()
    time_lbl = (t1-t0)/cv2.getTickFrequency()
    
    overall_det_times.append(time_det)
    overall_lbl_times.append(time_lbl)

    print(f"Det time = {time_det}")
    print(f"Lbl time = {time_lbl}")
    
overall_times = []
for i in range(len(overall_det_times)):
    overall_times.append(overall_det_times[i] + overall_lbl_times[i])

