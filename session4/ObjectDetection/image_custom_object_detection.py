from imageai.Detection import ObjectDetection
import os
from time import time

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "../models/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

our_time = time()

custom = detector.CustomObjects(person=True, dog=True)

detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3new-custom.jpg"), minimum_percentage_probability=30)
print("IT TOOK : ", time() - our_time)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"]  )
    print("--------------------------------")
