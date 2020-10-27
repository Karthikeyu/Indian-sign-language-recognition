#imports
import numpy as np
import cv2
import os
import imagePreprocessingUtils as ipu

CAPTURE_FLAG = False


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
    
    
camera = cv2.VideoCapture(0)
print('Now camera window will be open, then \n1) Place your hand gesture in ROI and press c key to start capturing images . \n2) Press esc key to exit.')

count = 0

while(True):
    (t,frame) = camera.read()
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame,ipu.START, ipu.END,(0,255,0),2 )
    # only for windows (remove lines 41 and 43 if you are using mac)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # please resize the window according to your screen.
    cv2.resizeWindow('image', 1200,800)
    ##
    pressedKey = cv2.waitKey(1)
    if pressedKey == 27:
        break
    elif pressedKey == ord('c'):
        if(CAPTURE_FLAG):
            CAPTURE_FLAG = False
        else:
            CAPTURE_FLAG = True
    
        # Region of Interest
    if(CAPTURE_FLAG):
        if(count<1200):
            roi = frame[ ipu.START[1]+5:ipu.END[1], ipu.START[0]+5:ipu.END[0]]
            cv2.imshow("Gesture", roi)
            frame = cv2.putText(frame, 'Capturing..', (50,70), cv2.FONT_HERSHEY_SIMPLEX,  
                   1.5, (0,255,0), 2, cv2.LINE_AA)
            roi = cv2.resize(roi, (ipu.IMG_SIZE,ipu.IMG_SIZE))
            cv2.imwrite("%s/%d.jpg"%(path,count), roi)
            count +=1
            print(count)
        else:
            break
    frame = cv2.putText(frame, str(count), (50,450), cv2.FONT_HERSHEY_SIMPLEX,  
                   2, (0,255,0), 2, cv2.LINE_AA)
    #cv2.imshow("Video",frame)
    cv2.imshow('image',frame)
        
camera.release()
cv2.destroyAllWindows()
    
print('Completed!')
