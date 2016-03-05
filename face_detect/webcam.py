
import sys
import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection as m


WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector

def show(area): 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Video', frame)


def crop(area): 
    crop = img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] #img[y: y + h, x: x + w]
    return crop

"""
given a jpg image, vectorize the grayscale pixels to 
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""

def vectorize(filename):
    size = WIDTH, HEIGHT # (width, height)
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array


cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

"""
load training data
"""
# create a list for filenames of smiles pictures
smilefiles = []
with open('smilesnew.csv', 'rb') as csvfile:
    for rec in csv.reader(csvfile, delimiter=','):
        smilefiles += rec

# create a list for filenames of neutral pictures
neutralfiles = []
with open('neutralnew.csv', 'rb') as csvfile:
    for rec in csv.reader(csvfile, delimiter=','):
        neutralfiles += rec

# N x dim matrix to store the vectorized data (aka feature space)       
phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
# 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
labels = []

# load smile data
PATH = "../data/smile/"
for idx, filename in enumerate(smilefiles):
    print PATH + filename
    phi[idx] = vectorize(PATH + filename)
    labels.append(1)

# load neutral data    
PATH = "../data/neutral/"
offset = idx + 1
for idx, filename in enumerate(neutralfiles):
    phi[idx + offset] = vectorize(PATH + filename)
    labels.append(0)

"""
training the data with logistic regression
"""
lr = logistic.Logistic(dim)
lr.train(phi, labels)



video_capture = cv2.VideoCapture(0)
counter = 0;
while True:
    counter = counter+1
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    name = "frame"+str(counter)+".jpg"
    cv.SaveImage(name, cv.fromarray(frame))
    img = cv.LoadImage(name) # input image
    mouth = m.findmouth(img)
    show(mouth)
    if mouth != 2: # did not return error
        mouthimg = crop(mouth)
        cv.SaveImage(name, mouthimg)
        # predict the captured emotion
        result = lr.predict(vectorize(name))
        if result == 1:
            print "you are smiling! :-) "
        else:
            print "you are not smiling :-| "
    else:
        print "failed to detect mouth. Try hold your head straight and make sure there is only one face."
    
    show(faces)
    
    # Draw a rectangle around the faces

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
