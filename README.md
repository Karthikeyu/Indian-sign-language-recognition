# Indian-sign-language-recognition

Hello, This repository contains python implementation for recognising Indian sign language gestures.

Run files in order:
Step 1: (Optional) To create your own Dataset run

  python image_capture.py

Step 2:  If you want to use our dataset download from below and extract it in the root directory of the repository.  Then run

  python imagePreprocessing.py

to preprocess all the images (from raw images to histograms of bovw model) and to classify using SVM.

Step 3: To visualise the confusion matrix run the file

  python visualise.py

Dataset can be downloaded from : https://drive.google.com/open?id=1keWr7-X8aR4YMotY2m8SlEHlyruDDdVi
