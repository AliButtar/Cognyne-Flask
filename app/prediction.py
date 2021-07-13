# from matplotlib import pyplot as plt
from PIL import Image
import os
# print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
# print("PATH:", os.environ.get('PATH'))

# import tomopy
# import mkl
# mkl.domain_set_num_threads(1, domain='fft') # Intel(R) MKL FFT functions to run sequentially

from mtcnn.mtcnn import MTCNN

import pandas
import numpy as np
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras import backend as K
from pkg_resources import NullProvider

class Model:

    def __init__(self):
        self.faceDetectorModel = MTCNN()
        self.faceRecognizierModel = load_model('app/face-recognition')
        

    def extractFace(self, pixels, required_size=(224, 224)):
        # detect faces in the image
        results = self.faceDetectorModel.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)

        # print(x1, x2, y1, y2)
    
        return face_array

    def getEmbedding(self, img):
        
        # extract faces
        face = [self.extractFace(f) for f in img]
        
        
        # convert into an array of samples
        sample = np.asarray(face, 'float32')
        # sample = np.asarray(img[0], 'float32')

        # prepare the face for the model, e.g. center pixels
        samples = self.preprocess_input(sample, version=2)

        # perform prediction
        yhat = self.faceRecognizierModel.predict(samples)

        return yhat

    def readImage(self, image):
        
        img = Image.open(image)

        finalImage = np.array(img)

        return finalImage

    def run_model(self, img):

        # Read the Image
        image = self.readImage(img)

        # Perform Inference
        embedding = self.getEmbedding([image])

        return embedding

    def preprocess_input(self, x, data_format=None, version=1):
        x_temp = np.copy(x)
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}

        if version == 1:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 93.5940
                x_temp[:, 1, :, :] -= 104.7624
                x_temp[:, 2, :, :] -= 129.1863
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= 93.5940
                x_temp[..., 1] -= 104.7624
                x_temp[..., 2] -= 129.1863

        elif version == 2:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 91.4953
                x_temp[:, 1, :, :] -= 103.8827
                x_temp[:, 2, :, :] -= 131.0912
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= 91.4953
                x_temp[..., 1] -= 103.8827
                x_temp[..., 2] -= 131.0912
        else:
            raise NotImplementedError

        return x_temp
        