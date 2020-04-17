import numpy as np
import cv2
import os
import random
import pickle

PATH = 'data'
TRAIN_FACTOR = 80
TOTAL_IMAGES = 1200
N_CLASSES = 35
CLUSTER_FACTOR = 8


START = (450,75)
END = (800,425)
IMG_SIZE = 128



def get_canny_edge(image):
   
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert from RGB to HSV
    HSVImaage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 

    # Finding pixels with itensity of skin
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(HSVImaage, lowerBoundary, upperBoundary)
    
    # blurring of gray scale using medianBlur
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(grayImage, grayImage, mask = skinMask)
    #cv2.imshow("masked2",skin)
    
    #. canny edge detection
    canny = cv2.Canny(skin,60,60)
    #plt.imshow(img2, cmap = 'gray')
    return canny,skin

def get_SIFT_descriptors(canny):
    # Intialising SIFT
    surf = cv2.xfeatures2d.SURF_create()
    #surf.extended=True
    canny = cv2.resize(canny,(256,256))
    # computing SIFT descriptors
    kp, des = surf.detectAndCompute(canny,None)
    #print(len(des))
    #sift_features_image = cv2.drawKeypoints(canny,kp,None,(0,0,255),4)
    return des

# Find the index of the closest central point to the each sift descriptor.   
def find_index(image, center):
    count = 0
    index = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i]) 
        else:
            calculated_distance = distance.euclidean(image, center[i]) 
            if(calculated_distance < count):
                index = i
                count = calculated_distance
    return index

def get_labels():
    class_labels = []
    for (dirpath,dirnames,filenames) in os.walk(PATH):
        dirnames.sort()
        for label in dirnames:
            #print(label)
            if not (label == '.DS_Store'):
                class_labels.append(label)
    
    return class_labels