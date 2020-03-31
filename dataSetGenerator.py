import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')

def insertOrUpdate(Id, Name):
    # connect to database
    conn = sqlite3.connect("FaceBase.db")
    
    cmd = "SELECT * FROM People WHERE ID=" + str(Id)
    
    # execute the SQL statement
    cursor = conn.execute(cmd)

    isRecordExist = 0

    for row in cursor:
        isRecordExist = 1

    if (isRecordExist == 1):
        cmd = "UPDATE People SET Name='" + str(Name) + "' WHERE  ID=" + str(Id)
    else:
        cmd = "INSERT INTO People(ID, Name) VALUES (" + str(Id) + ", '" + str(Name) + "')"

    # insert or update the database
    conn.execute(cmd)

    # commit changes
    conn.commit()
    
    # close the connection
    conn.close()

# MAIN FUNCTION
# enter id
id = input('Enter your ID: ')
name = input('Enter your name: ')

# insert or update database
insertOrUpdate(id, name)

sampleNum = 0

while True:
    # read the video camera
    ret, img = cam.read()
    # convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        # create face image file
        cv2.imwrite("dataSet/User." + str(id) + "." + str(sampleNum) + ".jpg", gray[y: y + h, x: x + w])
        # create rectangle to face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show images
    cv2.imshow("Face", img)

    # wait for response
    cv2.waitKey(100)

    if (sampleNum > 20):
        cam.release()
        cv2.destroyAllWindows()
        break