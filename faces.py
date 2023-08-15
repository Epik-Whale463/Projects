import cv2

face_cascade = cv2.CascadeClassifier(
    'C:\\Users\\ramud\\Desktop\\PYTHON\\BIG_PROJECT\\haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# img = cv2.imread('C:\\Users\ramud\\Desktop\\PYTHON\\BIG_PROJECT\faces\\charan\1.jpg')

cap = cv2.VideoCapture(0)
while True:
    # capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
      # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # recognize?
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)

        img_item = "my_img.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


# release capture when done
cap.release()
cv2.destroyAllWindows()
