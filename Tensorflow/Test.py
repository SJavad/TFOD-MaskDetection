import cv2 as cv


vid = cv.VideoCapture(0)

while True:
    fps = cv.CAP_PROP_FPS