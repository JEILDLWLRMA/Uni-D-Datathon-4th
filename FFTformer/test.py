###### 모델 테스트(추론) ######
import os
from os import listdir
from os.path import join, splitext
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose
from fftformer import fftformer
from restormer import Restormer
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize, InterpolationMode
from PIL import Image
import json
import shutil

np.random.seed(42)

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


model = fftformer()

# 데이터셋 경로
noisy_data_path = './Datasets/Test'
output_path = './submission'

if not os.path.exists(output_path):
    os.makedirs(output_path)

class CustomDatasetTest(Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in sorted(os.listdir(noisy_image_paths))]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])
        
        # Convert numpy array to PIL image
        if isinstance(noisy_image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path


test_transform = Compose([
    ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 데이터셋 경로
test_data_path = './Datasets/Test'
output_path = './additional'

# 데이터셋 로드 및 전처리
test_dataset = CustomDatasetTest(test_data_path, transform=test_transform)

# 데이터 로더 설정
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if not os.path.exists(output_path):
    os.makedirs(output_path)

def classify(label):
    state_dict = './task_pth/'
    if label in [0,1,2,3]:
        state_dict += 'best_FF_RF_fftformer.pth'
    elif label in [4,5,6,7]:
        state_dict += 'best_UD_RL_restormer.pth'
    elif label in [8,9]:
        state_dict += 'best_LE_fftformer.pth'
    elif label in [10,11]:
        state_dict += 'best_DU_fftformer.pth'
    elif label in [12,13]:
        state_dict += 'best_SC_fftformer.pth'
    elif label in [14,15]:
        state_dict += 'best_FI_fftformer.pth'
    elif label in [16,18]:
        state_dict += 'best_High_light_restormer.pth'
    elif label in [17,19]:
        state_dict += 'best_Low_light_restormer.pth'
    return state_dict

file_path = 'predicted_results.json'

with open(file_path, 'r') as file:
    data = json.load(file)

data_sorted = sorted(data, key=lambda x: int(x['filepath'].split('_')[-1].split('.')[0]))
# 이미지 denoising 및 저장

i=0

with torch.no_grad():
    for noisy_image, noisy_image_path in test_loader:
        label = data_sorted[i]['label']
        if label in [4,5,6,7,16,17,18,19]:
            model = Restormer()
        else:
            model = fftformer()

        model.eval()
        state_dict = classify(label)
        state_dict = torch.load(state_dict)
        if "params" in state_dict:
            state_dict = state_dict["params"] 
        model.load_state_dict(state_dict,strict = True)
        model.to(device)
        noisy_image = noisy_image.to(device)
        denoised_image = model(noisy_image)
        
        # denoised_image를 CPU로 이동하여 이미지 저장
        denoised_image = denoised_image.cpu().squeeze(0)
        denoised_image = (denoised_image * 0.5 + 0.5).clamp(0, 1)
        denoised_image = transforms.ToPILImage()(denoised_image)

        # Save denoised image
        output_filename = noisy_image_path[0]
        denoised_filename = output_path + '/' + output_filename.split('/')[-1][:-4] + '.jpg'
        denoised_image.save(denoised_filename) 
        print(f'Saved denoised image: {denoised_filename}')
        i += 1

def zip_folder(folder_path, output_zip):
    shutil.make_archive(output_zip, 'zip', folder_path)
    print(f"Created {output_zip}.zip successfully.")

zip_folder(output_path, './submission')