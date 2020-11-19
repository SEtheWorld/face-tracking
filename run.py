from pipeline import Pipeline
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--REDETECT',dest='FRAME_FOR_REDETECT',type=int,default=2)
    parser.add_argument('--CAP',dest='NUM_FRAME_CAPTURE',type=int,default=4)
    parser.add_argument('--FPS',dest='CAMERA_FPS',type=int,default=10)
    parser.add_argument('--IOU',dest='IOU_THRESHOLD',type=float,default=0.5)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    config = parse_args()
    pipeline = Pipeline(config)
    pipeline.run()
