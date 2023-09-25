import cv2 as cv 
import albumentations as A
from time import time 

def flipImage(image):
    transform = A.Compose([A.HorizontalFlip(always_apply=True, p=1.0), A.Resize(500, 600)])
    img = transform(image = image)['image']
    return img

vc = 0
cap = cv.VideoCapture(vc, cv.CAP_DSHOW) 
opened = cap.isOpened()
print(f'ID : {vc} - {opened}')

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    _, frame = cap.read()
    image = flipImage(frame)
    font = cv.FONT_HERSHEY_SIMPLEX
    new_frame_time = time()
    fps = str(int(1/(new_frame_time-prev_frame_time)))
    prev_frame_time = new_frame_time
    cv.putText(image, fps, (7, 30), font, 1, (100, 255, 0), 3, cv.LINE_AA)
    cv.imshow(f'cam test : res ({image.shape})', image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break