import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    croppedImage = None
    crop = None
    for i in range(10):
        for j in range(5):
            if np.any(croppedImage == None):
                croppedImage = img[y:y+h, x:x+w]
            else:
                croppedImage = np.hstack((croppedImage, img[y:y+h, x:x+w]))
            #print(croppedImage.shape)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            x+=w+d
        if np.any(crop == None):
            crop = croppedImage
        else:
            crop = np.vstack((crop, croppedImage)) 
        croppedImage = None
        x = 420
        y+=h+d
    return crop

def show_threshold(hsv, hist):
    # back projection of histogram using hsv of frame
            backProjection = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            # smoothhing using elliptical structuring kernel 
            kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)) # kernal
            # Convoluting using kernam 
            cv2.filter2D(backProjection,-1,kernal,backProjection)
            # blurring of gray scale using gaussian
            blur = cv2.GaussianBlur(backProjection, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            #cv2.imshow('blurr', blur)
            #thresholding using otsu Binarization which takes automatic threshold value using histogram
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #cv2.imshow("Thresh", thresh)
            # merging image with b g r
            thresh = cv2.merge((thresh,thresh,thresh))
            #cv2.imshow("res", res)
            # output window of final threshold frame
            cv2.imshow("Thresh", thresh)
            return

def get_hand_hist():
    cam = cv2.VideoCapture(1)
    if cam.read()[0]==False:
        cam = cv2.VideoCapture(0)
    #x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    croppedImage = None
    while True:
        (t, img) = cam.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            #cv2.imshow('croppedImage', croppedImage)
            hsvCrop = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            # Calculating histogram of cropped frame (HSV)
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif keypress == ord('s'):
            flagPressedS = True
            break
        elif keypress == 27:
            break
        if flagPressedC:
            show_threshold(hsv,hist)
        if not flagPressedS:
            #print('Not pressed S')
            croppedImage = build_squares(img)
        cv2.imshow("Set hand histogram", img)
    cam.release()
    cv2.destroyAllWindows()
    with open("hist", "wb") as f:
        pickle.dump(hist, f)


get_hand_hist()
