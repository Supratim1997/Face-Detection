import cv2
import sys
import os
from datetime import datetime

#cascPath = sys.argv[1]
cascPath = 'haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
# can not use the get(CV_CAP_PROP_FPS) method for webcam 
# :
if int(major_ver)  < 3 :
    fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


frame_count = 0 #int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

vid_cod = cv2.VideoWriter_fourcc(*'XVID')
flag = "stop"
img_counter = 0
while True:
    # Capture frame-by-frame
    duration = frame_count/fps
    ret, frame = video_capture.read()
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    # for (x,y,w,h) in faces:
    #     frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray)
    #     for (ex,ey,ew,eh) in eyes:
    #         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if (len(faces) >= 1 and flag == "stop"):
        timestamp = datetime.timestamp(datetime.now())
        fileName = "cam_video_{}.avi".format(str(timestamp))
        output = cv2.VideoWriter(fileName, vid_cod, 20.0, (640,480))
        flag = "start"
        print("Recording Starts at {}\n".format(datetime.now()))


    if (len(faces) >= 1 and flag == "start"):
        # img_name = "opencv_frame_{}.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        # img_counter = 1
        frame_count = frame_count + 1
        output.write(frame)
        flag = "start"
        
    if(flag == "start" and len(faces) == 0 and duration >= 6):
        print("Recording Stops at {}".format(datetime.now()))
        print('--------------------------')

        print(duration)
        print(frame_count)

        print('--------------------------')
        print("Saved as : {}\n".format(fileName))
        output.release()
        flag = "stop"
        duration = 0
        frame_count = 0

    if(flag == "start" and len(faces) == 0 and duration < 6):
        output.release()
        if os.path.exists(fileName):
            os.remove(fileName)
        flag = "stop"
        duration = 0
        frame_count = 0
        


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
output.release()
cv2.destroyAllWindows()