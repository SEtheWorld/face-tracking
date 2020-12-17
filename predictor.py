# For testing
import tensorflow as tf
import multiprocessing
import asyncio
import base64

# For Pi
# from tflite_runtime.interpreter import Interpreter

import numpy as np
from PIL import Image
import time


def load_labels(path):
    with open(path, "r") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


class Predictor(multiprocessing.Process):
    def __init__(self, model_path, label_path, result_queue, infer_lock):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.label = load_labels(label_path)
        # For testing
        self.interpreter = tf.lite.Interpreter(model_path)
        self.infer_lock = infer_lock
        # For Pi
        # self.interpreter = Interpreter(model_path)

        self.interpreter.allocate_tensors()
        _, self.height, self.width, _ = self.interpreter.get_input_details()[0]["shape"]

    def set_input_tensor(self, image):
        tensor_index = self.interpreter.get_input_details()[0]["index"]
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def classify_image(self, frame, top_k=1):
        """ Classify image by 

        Args:
            image ([CV2 BGR Image]): extracted image from live camera

        Returns:
            [tuple]: (label, time consumed)
        """

        # convert CV2 BGR to PIL RGB image
        try:
            image = (
                Image.fromarray(frame)
                .convert("RGB")
                .resize((self.width, self.height), Image.ANTIALIAS)
            )
        except Exception as e:
            print(e)
            return (0,0)
        start_time = time.time()
        self.set_input_tensor(image)
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()[0]
        output = np.squeeze(self.interpreter.get_tensor(output_details["index"]))

        # If the model is quantized (uint8 data), then dequantize the results
        if output_details["dtype"] == np.uint8:
            scale, zero_point = output_details["quantization"]
            output = scale * (output - zero_point)

        consumed_time = (time.time() - start_time) * 1000
        ordered = np.argpartition(-output, top_k)
        results = [(i, output[i]) for i in ordered[:top_k]]

        label = results[0]

        return label, consumed_time

    def run(self):
        while True:
            # with self.infer_lock:
            result = self.result_queue.get()
            label, infer_time = self.classify_image(result)
            encoded_image = base64.b64encode(result)
            metadata = {"image": encoded_image, "info": label}
            print(label)
            self.send(metadata)

    def send(self, metadata):
        pass
