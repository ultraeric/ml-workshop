import cv2
import os
from datetime import datetime

label = 0
directory_name = './data/{}'.format(label)
window_name = "Capturing Label {}".format(label)

try:
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
except OSError:
    print ('Error: Creating directory. ' + directory_name)

cv2.namedWindow(window_name)
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    exists, frame = vc.read()
else:
    exists = False

i = 0
while exists:
    start_x = (frame.shape[1] - frame.shape[0]) // 2
    end_x = frame.shape[1] - start_x
    frame = frame[:,start_x:end_x,:]
    cv2.imshow(window_name, frame)
    frame = cv2.resize(frame, (224, 224))
    key = cv2.waitKey(10)
    if key == 13 and i % 2 == 0:
        filename = directory_name + '/' + str(datetime.now()) + '.png'
        cv2.imwrite(filename, frame)
        print('{} saved'.format(filename))
    if key == 27:
        break
    exists, frame = vc.read()
    i += 1

cv2.destroyWindow(window_name)
