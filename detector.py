import cv2
import numpy as np
from PIL import Image
import sqlite3

rec = cv2.face.LBPHFaceRecognizer_create()

# load trained recognizer
rec.read('recognizer/trainingData.yml')

cascadePath = "Classifiers/haarcascade_frontalface_default.xml"
faceDetect = cv2.CascadeClassifier(cascadePath)
path = "dataSet"

# get profile
def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID=" + str(id)
    cursor = conn.execute(cmd)
    
    profile = None

    for row in cursor:
        profile = row

    # close the connection
    conn.close()

    return profile

# initialize video capture
cam = cv2.VideoCapture(0)

# import font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while (True):
    # read the video camera
    ret, img = cam.read()
    # convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # create rectangle to face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, conf = rec.predict(gray[y: y + h, x: x + w])
        
        # print(id)
        # get profile
        profile = getProfile(id)

        if (profile != None):
            # place text in the detector
            # id
            cv2.putText(img, str(profile[1]), (x, y + h + 30), font, 2, (0, 0, 255))
            # name
            cv2.putText(img, str(profile[2]), (x, y + h + 60), font, 2, (0, 0, 255))
            # age
            cv2.putText(img, str(profile[3]), (x, y + h + 90), font, 2, (0, 0, 255))
            # criminal records
            cv2.putText(img, str(profile[4]), (x, y + h + 120), font, 2, (0, 0, 255))
        else:
          cv2.putText(img, 'Unknown Profile', (x, y + h + 30), font, 2, (0, 0, 255))

    # show images
    cv2.imshow("Face", img)

    if (cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()