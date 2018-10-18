from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "../models/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

custom = detector.CustomObjects(person=True, motorcycle=True, bus=True)

video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom, input_file_path=os.path.join(execution_path, "traffic-mini.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic-mini_detected_custom")
                                , frames_per_second=20, log_progress=True)
print(video_path)