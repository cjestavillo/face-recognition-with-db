import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

def getImagesWithID(path):
    # list directories inside dataSet directory
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)
    faces = []
    IDs = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        # convert into numpy array
        faceNp = np.array(faceImg, 'uint8')

        # get ID, count image pathname backward
        ID = int(os.path.split(imagePath)[-1].split('.')[1])

        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training: ", faceNp)
        cv2.waitKey(10)

    return np.array(IDs), faces

# get IDs and faces
Ids, faces = getImagesWithID(path)
# train the recognizer
recognizer.train(faces, Ids)
# create faces's training data file
# recognizer.save() worked on Mac, but not on Pi
recognizer.save('recognizer/trainingData.yml')
# close windows
cv2.destroyAllWindows()
