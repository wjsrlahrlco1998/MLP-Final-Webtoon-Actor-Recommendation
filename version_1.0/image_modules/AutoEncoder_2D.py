# TensorFlow Memory Management
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

# Package Load
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from PIL import Image

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

# Define Auto Encoder Model
class AutoEncoder(Model):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        # 인코더 정의
        self.encoder = tf.keras.Sequential([
            Input(shape=(64, 64, 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(latent_dim, activation='relu')
        ])
        
        self.decoder = tf.keras.Sequential([
            Reshape((16, 16, 3)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

# Define Model AutoEncoder
class ModelAE:
    
    def __init__(self, model_path):
        '''Model load and init'''
        self.model = AutoEncoder(768)
        self.model.load_weights(model_path)
        self.model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mse'])
    
    def convert_type(self, image):
        '''Convert image type : cv2, PIL'''
        if isinstance(image, np.ndarray):
            pass
        elif isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        return image
    
    def convert_type_custom(self, image, TYPE=0, GRAY=False):
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
    
    def resize_image(self, image):
        '''Resize image'''
        resize_image = cv2.resize(image, (64, 64))
        
        return resize_image
    
    def convert_bgr_to_rgb(self, image):
        '''Convert image bgr to rgb'''
        conv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return conv_image
    
    def normalize_image(self, image):
        '''Normalization image'''
        norm_image = image.astype(np.float32) / 255.
        
        return norm_image
    
    def reshape_image(self, image):
        '''Reshape image'''
        reshap_image = image.reshape(1, 64, 64, 3)
        
        return reshap_image
    
    def get_latent_vector(self, image):
        '''Get latent vector from image'''
        vec_image = self.model.encoder(image)
        
        return vec_image
    
    def run(self, image_1, image_2):
        '''All process run -> result : cosine similarity image_1 and image_2'''
        
        # image_1 preprocessing
        image_1 = self.convert_type(image_1)
        image_1 = self.resize_image(image_1)
        image_1 = self.convert_bgr_to_rgb(image_1)
        image_1 = self.normalize_image(image_1)
        image_1 = self.reshape_image(image_1)
        image_1 = self.get_latent_vector(image_1)
        
        # image_2 preprocessing
        image_2 = self.convert_type(image_2)
        image_2 = self.resize_image(image_2)
        image_2 = self.convert_bgr_to_rgb(image_2)
        image_2 = self.normalize_image(image_2)
        image_2 = self.reshape_image(image_2)
        image_2 = self.get_latent_vector(image_2)
        
        # get cosine similarity
        cos_sim = cosine_similarity(image_1, image_2)
        
        return cos_sim[0][0]