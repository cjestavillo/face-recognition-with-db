import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

id = input('enter user id: ')
sampleNum = 0

while (True):
    # read the video camera
    ret, img = cam.read()
    # convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        # create face image file
        cv2.imwrite("dataSet/User." + str(id) + "." + str(sampleNum) + ".jpg", gray[y: y + h, x: x + w])
        # create rectangle to face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # wait for response
        cv2.waitKey(100)

    # show images
    cv2.imshow("Face", img)

    # wait for response
    cv2.waitKey(1)

    # if (cv2.waitKey(1) == ord('q')):
    #     break

    if (sampleNum > 20):
        break;

cam.release()
cv2.destroyAllWindows()