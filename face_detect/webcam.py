import sys
import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection as m


WIDTH, HEIGHT = 30, 12 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector
cascPath = "haarcascade/haarcascade_frontalface_default.xml"

class processVideo(object):

    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.average_happy = 0
        self.average_sad = 0
        self.frame = ''
        self.img = ''
        self.lr = logistic.Logistic(dim)
        self.faceCascade = cv2.CascadeClassifier(cascPath)


    def train(self):

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
        print phi
        # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
        labels = []

        # load smile data
        PATH = "data/smile/"
        for idx, filename in enumerate(smilefiles):
            print PATH + filename
            phi[idx] = self.vectorize(PATH + filename)
            labels.append(1)

        # load neutral data    
        PATH = "data/neutral/"
        offset = idx + 1
        for idx, filename in enumerate(neutralfiles):
            phi[idx + offset] = self.vectorize(PATH + filename)
            labels.append(0)

        """
        training the data with logistic regression
        """
        self.lr.train(phi, labels)


    def show(self,faces): 
        for (x, y, w, h) in faces:
            cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Video', self.frame)


    def crop(self,area): 
        crop = self.img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] #img[y: y + h, x: x + w]
        return crop

    """
    given a jpg image, vectorize the grayscale pixels to 
    a (width * height, 1) np array
    it is used to preprocess the data and transform it to feature space
    """

    def vectorize(self,filename):
        size = WIDTH, HEIGHT # (width, height)
        im = Image.open(filename) 
        resized_im = im.resize(size, Image.ANTIALIAS) # resize image
        im_grey = resized_im.convert('L') # convert the image to *greyscale*
        im_array = np.array(im_grey) # convert to np array
        oned_array = im_array.reshape(1, size[0] * size[1])
        print oned_array
        return oned_array

    def face_detect(self):
        counter = 0;
        while True:
            counter = counter+1
            # Capture frame-by-frame
            ret, self.frame = self.video_capture.read()

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            name = "frame"+str(counter)+".jpg"
            cv.SaveImage(name, cv.fromarray(self.frame))
            self.img = cv.LoadImage(name) # input image
            mouth = m.findmouth(self.img)
            if mouth != 2: # did not return error
                mouthimg = self.crop(mouth)
                cv.SaveImage(name, mouthimg)
                # predict the captured emotion
                result = self.lr.predict(self.vectorize(name))
                if result == 1:
                    self.average_happy +=1
                    print "you are smiling! :-) "
                else:
                    self.average_sad += 1
                    print "you are not smiling :-| "
            else:
                print "failed to detect mouth. Try hold your head straight and make sure there is only one face."

                self.show(faces)

                    # Draw a rectangle around the faces

                # Display the resulting frame
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # When everything is done, release the capture
            if self.average_happy or self.average_sad > 0:
                print "Percent Happy/Neutral Smile detected {0}".format(self.average_happy)
                print "Percent Sad Smile detected {0}".format(self.average_sad)
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    videoobj = processVideo()
    videoobj.train()
    videoobj.face_detect()