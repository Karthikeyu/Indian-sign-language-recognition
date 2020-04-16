#imports
import numpy as np
import cv2
import os

IMG_SIZE = 128
CAPTURE_FLAG = False

#ROI
start = (450,75)
end = (800,425)

directory = input('Enter dataset directory name: ')

exit = '**'

try:
    os.mkdir(directory)
except:
    print('Directory already exists!')
    
subDirectory = input('Enter sub directory name or press ** to exit: ')

if subDirectory == exit:
    print('exit')
else:
    path = directory + '/'+ subDirectory+ '/'
    try:
        os.mkdir(path)
    except:
        print('Sub directory already exists!')
    
    
camera = cv2.VideoCapture(1)
print('Now camera window will be open, then \n1) Place your hand gesture in ROI and press c key to start capturing images . \n2) Press esc key to exit.')

count = 0

while(True):
    (t,frame) = camera.read()
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame,start, end,(0,255,0),2 )
    pressedKey = cv2.waitKey(1)
    if pressedKey == 27:
        break
    elif pressedKey == ord('c'):
        if(CAPTURE_FLAG):
            CAPTURE_FLAG = False
        else:
            CAPTURE_FLAG = True
        #frame = cv2.flip(frame,1)
    
        # Region of Interest
        #for count in range(0,500):
    if(CAPTURE_FLAG):
        if(count<1200):
            roi = frame[ start[1]+5:end[1], start[0]+5:end[0]]
            #roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            #roi_gray = cv2.GaussianBlur(roi_gray, (7,7),0)
            cv2.imshow("Gesture", roi)
            frame = cv2.putText(frame, 'Capturing..', (50,70), cv2.FONT_HERSHEY_SIMPLEX,  
                   1.5, (0,255,0), 2, cv2.LINE_AA)
            roi = cv2.resize(roi, (IMG_SIZE,IMG_SIZE))
            cv2.imwrite("%s/%d.jpg"%(path,count), roi)
            count +=1
            print(count)
        else:
            break
    frame = cv2.putText(frame, str(count), (50,450), cv2.FONT_HERSHEY_SIMPLEX,  
                   2, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Video",frame)
        
camera.release()
cv2.destroyAllWindows()
    
print('Completed!')
