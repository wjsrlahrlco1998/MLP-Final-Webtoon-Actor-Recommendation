import pickle
import argparse
import time
import os
from tqdm import tqdm_notebook
import cv2
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from tqdm import tqdm

from model import Encoder as Encoder_2
from model import Generator as Generator_2
from stylegan2.model import Generator as StyleGAN

class Face2Webtoon:
    
    def __init__(self, device='cuda:0', image_size=256, 
                 ae_model_path='./checkpoint/0325600.pt', 
                 stylegan_model_path='./checkpoint/NaverWebtoon_StructureLoss.pt'):
        self.device = device
        self.image_size = image_size
        torch.set_grad_enabled(False)
        
        self.encoder = Encoder_2(32).to(self.device)
        self.generator = Generator_2(32).to(self.device)
        
        ckpt = torch.load(ae_model_path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["e_ema"])
        self.generator.load_state_dict(ckpt["g_ema"])
        
        self.encoder.eval()
        self.generator.eval()
        
        stylegan_ckpt = torch.load(stylegan_model_path, map_location=self.device)
        self.latent_dim = stylegan_ckpt['args'].latent
        
        self.stylegan = StyleGAN(image_size, self.latent_dim, 8).to(device)
        self.stylegan.load_state_dict(stylegan_ckpt["g_ema"], strict=False)
        self.stylegan.eval()

    def load_image(self, image, size):
        image = self.image2tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        w, h = image.shape[-2:]
        if w != h:
            crop_size = min(w, h)
            left = (w - crop_size)//2
            right = left + crop_size
            top = (h - crop_size)//2
            bottom = top + crop_size
            image = image[:,:,left:right, top:bottom]

        if image.shape[-1] != size:
            image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear", align_corners=True)

        return image
    
    def image2tensor(self, image):
        image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
        return (image-0.5)/0.5

    def tensor2image(self, tensor):
        tensor = tensor.clamp(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
        return tensor*0.5 + 0.5

    def imshow(self, img, size=5, cmap='jet'):
        plt.figure(figsize=(size,size))
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.show()

    def horizontal_concat(self, imgs):
        return torch.cat([img.unsqueeze(0) for img in imgs], 3) 
 
        
    def run(self, image, truncation=0.5, num_styles=5):
         
        trunc = self.stylegan.mean_latent(4096).detach().clone()
        latent = self.stylegan.get_latent(torch.randn(num_styles, self.latent_dim, device=self.device))
        imgs_gen, _ = self.stylegan([latent],
                                truncation=truncation,
                                truncation_latent=trunc,
                                input_is_latent=True,
                                randomize_noise=True)
        
        # 이미지 전처리
        image = self.load_image(image, self.image_size)
        inputs = torch.cat([image.to(self.device), imgs_gen])
        results = self.horizontal_concat(inputs.cpu())
        structures, target_textures = self.encoder(inputs)
        structure = structures[0].unsqueeze(0).repeat(len(target_textures),1,1,1)
        source_texture = target_textures[0].unsqueeze(0).repeat(len(target_textures),1)
        
        for swap_loc in [1000, 1000]: # 숫자 높이면 원본이랑 비슷해짐.
            textures = [source_texture for _ in range(swap_loc)] + [target_textures for _ in range(len(self.generator.layers) - swap_loc)]        
            fake_imgs = self.generator(structure, textures, noises=0)

            results = torch.cat([results, self.horizontal_concat(fake_imgs).cpu()], dim=2)
            
        anime_images = self.tensor2image(results) # type : cv2(RGB 상태)
        result_image = anime_images[512:768, 1280:1536]
        
        return result_image