# tensor memory management
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

import numpy as np
from ResNet50.resnet50 import ResNet50
from keras.layers import Input
from keras.preprocessing import image as keras_image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

import cv2
import PIL
from PIL import Image

import pandas as pd
from tqdm.notebook import tqdm
import re

import matplotlib.pyplot as plt  

class RN50:
    def __init__(self, size=224):
        self.image_size = size
        self.image_input = Input(shape=(size, size, 3))
        self.feature_model = ResNet50(input_tensor=self.image_input, include_top=False, weights='imagenet')
    
    # <--이미지 전처리-->
    def resize(self, image, size):
        '''resize -> size x size'''
        if isinstance(image, np.ndarray): # image form : cv2
            image = cv2.resize(image, (size, size))
        elif isinstance(image, PIL.Image.Image): # image form : PIL
            image = image.resize((size, size))
        
        return image

    def convert_type(self, image, TYPE=0, GRAY=False):
        '''convert type
        TYPE == 0 : cv2 -> PIL
        TYPE == 1 : PIL -> cv2
        '''
        if GRAY:
            if TYPE == 0: # form : cv2 -> PIL
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = Image.fromarray(image)
            elif TYPE == 1: # form : PIL -> cv2
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            if TYPE == 0: # form : cv2 -> PIL
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            elif TYPE == 1: # form : PIL -> cv2
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def set_threshold(self, image, GRAY=True):
        '''preprocessing threshold'''
        if isinstance(image, np.ndarray) : # image form : cv2
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        elif isinstance(image, PIL.Image.Image): # image form : PIL
            image = self.convert_type(image, TYPE=1, GRAY=GRAY)
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        return image
    
    def convert_gray(self, image):
        '''convert color to gray'''
        if isinstance(image, np.ndarray): # image form : cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(image, PIL.Image.Image): # image form : PIL
            image = self.convert_type(image, TYPE=1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def convert_input_form(self, image, GRAY=True):
        '''convert image to input form'''
        if isinstance(image, np.ndarray): # image form : cv2
            image = self.convert_type(image, GRAY=GRAY)
            image_array = np.expand_dims(image, axis=0)
            image_array = preprocess_input(image_array)
        elif isinstance(image, PIL.Image.Image): # image form : PIL
            image_array = keras_image.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)
        
        return image_array
    
    def get_feature_vector(self, image):
        '''convert image to feature vector'''

        feature_vector = self.feature_model.predict(image)
        a, b, c, n = feature_vector.shape
        feature_vector= feature_vector.reshape(b,n)
        
        return feature_vector
    
    # <--이미지 코사인 유사도 계산-->
    def cal_cos_sim(self, image_vector_1, image_vector_2):
        '''calculate cosine similarity between image1 and image2'''
        return cosine_similarity(image_vector_1, image_vector_2)
    
    # <--전체 동작 코드-->
    def run(self, image_1, image_2):
        images = [image_1, image_2]
        vectors = []
        
        for image in images:
            image = self.resize(image, size=self.image_size)
            image = self.convert_gray(image)
            image = self.set_threshold(image, GRAY=True)
            image = self.convert_input_form(image, GRAY=True)
            image = self.get_feature_vector(image)
            vectors.append(image)
        
        cos_sim = cosine_similarity(vectors[0], vectors[1])
        
        return cos_sim

if __name__ == '__main__':
    
    PATH_1 = '../image_data/Actor/'
    PATH_2 = '../image_data/Webtoon/'
    
    image_1 = image.load_img(PATH_1 + '고아성.jpg', target_size=(224, 224))
    image_2 = image.load_img(PATH_2 + '폭풍의전학생_이연희.jpg', target_size=(224, 224))
    
    model = RN50()
    print(model.run(image_1, image_2))