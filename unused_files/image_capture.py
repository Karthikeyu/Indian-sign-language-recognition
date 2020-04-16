#imports
import numpy as np
import cv2
import os

IMG_SIZE = 96
CAPTURE_FLAG = False

#ROI
start = (100,100)
end = (500,500)

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
print('Now camera window will be open, then \n1) Place your hand gesture in ROI and press s key to start capturing images . \n2) Press esc key to exit.')

count = 0

while(True):
    
    (t,frame) = camera.read()
    cv2.rectangle(frame,start, end,(0,255,0),2 )
    cv2.imshow("Video",frame)
    pressedKey = cv2.waitKey(1)
    if pressedKey == 27:
        break
    elif pressedKey == ord('s'):
        if(CAPTURE_FLAG):
            CAPTURE_FLAG = False
        else:
            CAPTURE_FLAG = True
        #frame = cv2.flip(frame,1)
    
        # Region of Interest
        #for count in range(0,500):
    if(CAPTURE_FLAG):
        if(count<500):
            roi = frame[ start[1]:end[1], start[0]:end[0]]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.GaussianBlur(roi_gray, (7,7),0)
            roi_gray = cv2.resize(roi_gray, (IMG_SIZE,IMG_SIZE))
            cv2.imshow("GrayScale", roi_gray)
            cv2.imwrite("%s/%d.jpg"%(path,count), roi_gray)
            count +=1
            print(count)
        else:
            break
        
camera.release()
cv2.destroyAllWindows()
    
print('Completed!')