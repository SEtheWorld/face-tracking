import cv2
import numpy as np


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def send_faces(frame, bboxes):
    for ix, bbox in enumerate(bboxes):
        (x, y, w, h) = bbox
        if w * h != 0:
            cropped_face = frame[int(x) : int(x + w), int(y) : int(y + h)]
            cv2.imwrite("output/cropped{}.png".format(ix), cropped_face)


def convert_box_to_int(bbox):
    (x, y, w, h) = bbox
    return (int(x), int(y), int(w), int(h))


def draw_boxes(frame, boxes, color=(0, 255, 0)):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    return frame


def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image


def get_iou(bbox1, bbox2):

    # Bounding box follows (x,y,w,h) order
    (x1, y1, w1, h1) = map(lambda x: int(x), bbox1)
    (x2, y2, w2, h2) = map(lambda x: int(x), bbox2)

    max_size = (max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2))

    window1 = np.zeros(max_size)
    window2 = np.zeros(max_size)

    window1[x1 : x1 + w1, y1 : y1 + h1] = 1
    window2[x2 : x2 + w2, y2 : y2 + h2] = 1

    intersection = np.logical_and(window1, window2)
    union = np.logical_or(window1, window2)

    iou = np.sum(intersection) / np.sum(union)

    return iou
