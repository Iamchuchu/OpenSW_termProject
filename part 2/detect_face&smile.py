# Detect Face & Smile
# pip install opencv-python

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('./image.png') # 이미지
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for(x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    face_gray = gray[y:y+h, x:x+w]
    face_color = img[y:y+h, x:x+w]
    smile = smile_cascade.detectMultiScale(face_gray, 1.2, 4, minSize=(50,20))
    
    for(sx, sy, sw, sh) in smile:
        print(sx, sy, sw, sh) 
        mid1 = (sx+sw)/2
        mid2 = (sy+sh)/2
        
        cv2.putText(img, 'Smile', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
