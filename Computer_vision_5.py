import cv2

# CC for face + smile
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
cap = cv2.VideoCapture(0)
while True:
    # read a frame
    ret, frame = cap.read()
    # Cnvt grsc
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Dtct fce
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (245, 13, 209), 2)

        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # smile
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            # rectangle around smile
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (232, 23, 75), 2)
    # display frame with the detection
    cv2.imshow('Smile Detection', frame)
    # loops until q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()