from predictor import Predictor
import cv2

predictor =  Predictor(model_path='/home/baophuc/ARI/Face/face_tracking/model/lite-model_object_detection_mobile_object_labeler_v1_1.tflite',
                    label_path='/home/baophuc/ARI/Face/face_tracking/model/mobile_object_labeler_v1_labelmap.csv')
image = cv2.imread('frame.png')
print(predictor.classify_image(image))
