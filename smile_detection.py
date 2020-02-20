import cv2

## loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

## defining a function that will do the detection

## gray is image in black white
# frame is original image

def detect(gray , frame):
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    ## faces have tuple of X,Y,W,H of image coordinates of rectangle
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y),  (x+w, y+h) , (255,0,0) , 2)
        ## (x, y) is top left corner
        ##(x+w, y+h) is bottom rights corner
        
        roi_gray = gray[y:y+h , x:x+w]
        ## region of interest in gray
        roi_color = frame[y:y+h , x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.15,3)
        
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey),  (ex+ew, ey+eh) , (0,255,0) , 2)
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7,22)
        
        for(sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy),  (sx+sw, sy+sh) , (0,0,255) , 5)
            cv2.putText(roi_color,'Smile',(sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2, cv2.LINE_AA)
        
        
    return frame 


## doing face recog with webcam


video_capture = cv2.VideoCapture(0)
## 0 - webcam of computer
## 1 - external webcam

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
video_capture.release()
cv2.destroyAllWindows()    