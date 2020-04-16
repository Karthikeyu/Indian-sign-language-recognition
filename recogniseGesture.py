#imports
import numpy as np
import cv2
import os
import pickle
import imagePreprocessingUtils as ipu


CAPTURE_FLAG = False

class_labels = ipu.get_labels()
                

def recognise(cluster_model, classify_model):
    global CAPTURE_FLAG
    camera = cv2.VideoCapture(1)
    print('Now camera window will be open, then \n1) Place your hand gesture in ROI (rectangle) \n2) Press esc key to exit.')
    count = 0
    while(True):
        (t,frame) = camera.read()
        frame = cv2.flip(frame,1)
        cv2.rectangle(frame,ipu.START, ipu.END,(0,255,0),2 )
        pressedKey = cv2.waitKey(1)
        if pressedKey == 27:
            break
        elif pressedKey == ord('p'):
            if(CAPTURE_FLAG):
                CAPTURE_FLAG = False
            else:
                CAPTURE_FLAG = True
        if(CAPTURE_FLAG):
            # Region of Interest
            roi = frame[ ipu.START[1]+5:ipu.END[1], ipu.START[0]+5:ipu.END[0]]
            if roi is not None:
                roi = cv2.resize(roi, (ipu.IMG_SIZE,ipu.IMG_SIZE))
                img = ipu.get_canny_edge(roi)[0]
                print(img)
                sift_disc = ipu.get_SIFT_descriptors(img)
            print(type(sift_disc))
            if sift_disc is not None:
                visual_words = cluster_model.predict(sift_disc)
                print('visual words collected.')
                bovw_histogram = np.array(np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR))
                pred = classify_model.predict([bovw_histogram])
                label = class_labels[pred[0]]
                frame = cv2.putText(frame, label, (50,70), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0,255,0), 2, cv2.LINE_AA)
       
        cv2.imshow("Video",frame)
    camera.release()
    cv2.destroyAllWindows()
  
    
clustering_model = pickle.load(open('mini_kmeans_model.sav', 'rb'))    
classification_model = pickle.load(open('svm_model.sav', 'rb'))
recognise(clustering_model,classification_model )