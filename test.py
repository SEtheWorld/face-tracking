from utils.utils import get_iou

box1 = (0,0,100,100)
box2= (50,50,100,100)
print(get_iou(box1,box2))