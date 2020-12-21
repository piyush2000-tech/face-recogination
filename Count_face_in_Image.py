import cv2

from matplotlib import pyplot as plt

image = cv2.imread('pg.jpg')

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

print(image.shape)

print(gray.shape)

print(gray)

cascPath = "haarcascade_frontalface_default.xml"
harr_face= cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

#harr_face = cv2.CascadeClassifier('opencv-3.0.0/data/harcascades/haarcascade_frontalface.xml') 

faces = harr_face.detectMultiScale(image,scaleFactor=1.06,minNeighbors=5);

print("face Found",len(faces))

for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(image)

plt.axis('off')

plt.show()

