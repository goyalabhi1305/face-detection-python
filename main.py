import cv2
#loading the ai trained data
trained_face_data = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

#test photo
img = cv2.imread('20170513_174131.jpg')

#test webcam
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    #converting to geryscale
    grayscaled_img= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detecting faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0), 2)

    print(face_coordinates)
    #some function
    cv2.imshow('Abhishek | Face Detector', frame)
    cv2.waitKey(1)

