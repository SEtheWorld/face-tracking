from pipeline import Pipeline
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--REDETECT", dest="FRAME_FOR_REDETECT", type=int, default=20)
    parser.add_argument("--CAP", dest="INTERVAL", type=int, default=5)
    parser.add_argument("--FPS", dest="CAMERA_FPS", type=int, default=30)
    parser.add_argument("--IOU", dest="IOU_THRESHOLD", type=float, default=0.7)
    parser.add_argument(
        "--label",
        type=str, 
        default="model/mobile_object_labeler_v1_labelmap.csv"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/lite-model_object_detection_mobile_object_labeler_v1_1.tflite"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    config = parse_args()
    pipeline = Pipeline(config)
    pipeline.run()
