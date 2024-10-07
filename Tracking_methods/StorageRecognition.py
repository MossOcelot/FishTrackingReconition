import os
import cv2
from Tracking_methods.Recognition import *

class FishRegister:
    def __init__(self, img, name, features):
        self.img = img
        self.name = name
        self.features = features
        

def get_fish_register():
    files = os.listdir("fishs")

    storage = []
    for i, file in enumerate(files):
        img_original = cv2.imread(f'fishs/{file}')

        img = cv2.resize(img_original, (100, 100))
        img_normalization = preprocess_image(img)
        key, des = extract_features_sift(img_normalization)
        
        # ขนาดของกรอบที่ต้องการ
        crop_size = 20

        # คำนวณตำแหน่งกลาง
        height, width = img.shape[:2]
        x_start = (width - crop_size) // 2
        y_start = (height - crop_size) // 2


        fish = FishRegister(img_original, i, des)
        storage.append(fish)
        
    return storage
