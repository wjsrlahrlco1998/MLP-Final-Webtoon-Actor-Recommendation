import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import re
import dlib
import torchvision.transforms as transforms # 이미지 사이즈 변환을 위한 클래스

class OrbKNN:

    def __init__(self):
        '''init class'''
        pass
  
    def img_to_rgb(self, image):
        '''convert image to rgb'''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def img_to_bgr(self, image):
        '''convert image to bgr'''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    def img_adjust(self, image, brightness = 0, contrast = 30):
        '''adjust image'''
        image = np.int16(image)
        image = image * (contrast / 127 + 1) - contrast + brightness
        image = np.clip(image, 0, 255)
        image = np.uint8(image)

        return image

    def img_resize(self, image, size = 256):
        '''resize image'''
        image = cv2.resize(image, (size, size))

        return image
  
    def img_face_align(self, image, size = 256, padding = 0.65):
        '''face align image'''
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('../util/shape_predictor_5_face_landmarks.dat')
        dets = detector(image)
        if dets:
            pass
        else:
            # print("No detect face")
            return image
        s = sp(image, dets[0])
        image = dlib.get_face_chip(image, s, size=size, padding=padding)

        return image

    def img_compare(self, image_1, image_2, color_image_1, color_image_2, ratio=0.75, show=False):
        '''compare image'''
        # Initiate SIFT detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(image_1, None)
        kp2, des2 = orb.detectAndCompute(image_2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good.append([m])

        # Draw first 10 matches.
        knn_image = cv2.drawMatchesKnn(color_image_1, kp1, color_image_2, kp2, good, None, flags=2)
        if show:
            plt.imshow(knn_image)
            plt.show()

        return knn_image

    def img_show(self, image):
        '''show image'''
        plt.figure(figsize=(16, 10))
        plt.imshow(image)

    def run(self, image_1, image_2, size = 256, padding = 0.65, ratio = 0.75, brightness = 0, contrast = 30, show=False):
        '''run compare image'''
        image_1 = self.img_to_rgb(image_1)
        image_1 = self.img_adjust(image_1, brightness=brightness, contrast=contrast)
        image_1 = self.img_resize(image_1, size=size)
        image_1 = self.img_face_align(image_1, size=size, padding=padding)
        color_image_1 = image_1
        image_1 = self.img_to_bgr(image_1)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

        image_2 = self.img_to_rgb(image_2)
        image_2 = self.img_adjust(image_2, brightness=brightness, contrast=contrast)
        image_2 = self.img_resize(image_2, size=size)
        image_2 = self.img_face_align(image_2, size=size, padding=padding)
        color_image_2 = image_2
        image_2 = self.img_to_bgr(image_2)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        knn_image = self.img_compare(image_1, image_2, color_image_1, color_image_2, ratio=0.75, show=show)

        return knn_image
    
if __name__ == '__main__':
    img = OrbKNN()
    
    actor_image_path = '../image_data/Actor2Webtoon'
    webtoon_image_path = '../image_data/Webtoon'
    image_1 = cv2.imread(f'{actor_image_path}/webtoon_고아성.jpg')
    image_2 = cv2.imread(f'{webtoon_image_path}/폭풍의전학생_이연희.jpg')
    
    knn_image = cImg.run(image_1, image_2, show=True)
    plt.imshow(knn_image)
    plt.show()