import tensorflow as tf 
import tensorflow.keras

import numpy as np



tf.keras.backend.clear_session()

def change_model(path):
    model = tf.lite.Interpreter(path)
    config = model.get_tensor_details()
    print(config)


if __name__ == "__main__":
    change_model('model/lite-model_object_detection_mobile_object_labeler_v1_1.tflite')
