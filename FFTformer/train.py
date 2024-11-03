import os
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import CenterCrop, Resize
from PIL import Image
from torchvision.transforms.functional import crop, hflip, vflip, rotate, adjust_brightness
from torchvision.transforms import v2, CenterCrop, Resize, ToTensor, Normalize, InterpolationMode
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize, InterpolationMode, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio
import numpy as np
import torchvision.transforms as transforms
from fftformer import fftformer
from os.path import join, splitext
from restormer import Restormer

# 시작 시간 기록
start_time = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
task = 'UD_RL' #Dataset안에서 fine-tuning할 폴더이름

import torch
from torchvision.transforms import ToPILImage

# 역정규화 클래스 정의
class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # std로 곱하고 mean 더하기
        return tensor

# 역정규화 객체 생성
denormalize = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
class Dataset_Train(Dataset):
    def __init__(self, clean_image_paths, noisy_image_paths, transform=None):
        self.clean_image_paths = [os.path.join(clean_image_paths, x) for x in os.listdir(clean_image_paths)]
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform
        self.center_crop = CenterCrop(1080)
        self.resize = Resize((224, 224), interpolation=InterpolationMode.BICUBIC)

        # Create a list of (noisy, clean) pairs
        self.noisy_clean_pairs = self._create_noisy_clean_pairs()

    def _create_noisy_clean_pairs(self):
        clean_to_noisy = {}
        for clean_path in self.clean_image_paths:
            clean_id = '_'.join(os.path.basename(clean_path).split('_')[:-1])
            clean_to_noisy[clean_id] = clean_path
        
        noisy_clean_pairs = []
        for noisy_path in self.noisy_image_paths:
            noisy_id = '_'.join(os.path.basename(noisy_path).split('_')[:-1])
            if noisy_id in clean_to_noisy:
                clean_path = clean_to_noisy[noisy_id]
                noisy_clean_pairs.append((noisy_path, clean_path))
        
        return noisy_clean_pairs

    def __len__(self):
        return len(self.noisy_clean_pairs)

    def __getitem__(self, index):
        noisy_image_path, clean_image_path = self.noisy_clean_pairs[index]

        noisy_image = Image.open(noisy_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")
        
        # Central Crop and Resize
        noisy_image = self.center_crop(noisy_image)
        clean_image = self.center_crop(clean_image)
        noisy_image = self.resize(noisy_image)
        clean_image = self.resize(clean_image)
        
        # 동일한 변환을 적용
        if self.transform:
            seed = torch.seed()  # 동일한 시드를 설정하여 같은 변환 적용
            torch.manual_seed(seed)
            noisy_image = self.transform(noisy_image)
            torch.manual_seed(seed)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

# train_transform 정의
train_transform = Compose([
    ToTensor(),
    # RandomHorizontalFlip(p=0.5),        # 50% 확률로 수평 뒤집기
    # RandomVerticalFlip(p=0.5),          # 50% 확률로 수직 뒤집기
    # RandomRotation((0, 90)),            # 0도에서 90도 사이의 랜덤 회전
    # ColorJitter(brightness=(0.7, 1.3)), # 밝기를 0.7배에서 1.3배 사이로 조절
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
    
    
class Dataset_Test(Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
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
    transforms.CenterCrop(1080),
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
val_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 하이퍼파라미터 설정
num_epochs = 10
batch_size = 1
learning_rate = 1e-3

# 데이터셋 경로
train_noisy_image_paths = f'./Datasets/{task}/Training/noise'
train_clean_image_paths = f'./Datasets/{task}/Training/GT'

val_noisy_image_paths = f'./Datasets/{task}/Validation/noise'
val_clean_image_paths = f'./Datasets/{task}/Validation/GT'
directory = f'./results/{task}'
os.makedirs(directory, exist_ok=True)

train_dataset = Dataset_Train(train_clean_image_paths, train_noisy_image_paths, train_transform)
val_dataset = Dataset_Train(val_clean_image_paths, val_noisy_image_paths, val_transform)

# 데이터 로더 설정
num_cores = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=int(num_cores/2), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=int(num_cores/2), shuffle=True)

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Restormer 모델 인스턴스 생성 및 GPU로 이동
model = Restormer().to(device)
state_dict = torch.load('./motion_deblurring.pth')
if 'params' in state_dict:
    state_dict = state_dict["params"]
model.load_state_dict(state_dict,strict = True)

#kbnet
# checkpoint = torch.load('defocus.pth')
# model.load_state_dict(checkpoint['params'])

#fftformer
# torch.load('motion_deblurring.pth')

# 손실 함수와 최적화 알고리즘 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.L1Loss()
scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=31950, eta_min=1e-6)

# 모델의 파라미터 수 계산
total_parameters = count_parameters(model)
print("Total Parameters:", total_parameters)

metric = PeakSignalNoiseRatio().to(device)

# 모델 학습
best_psnr = 0
over=False #psnr이 30이 넘으면 어떤 패치에 과적합이 된 것은 아닌지, 현재 모델이 힘들어하는 패치 검사
for epoch in range(num_epochs):
    model.train()
    epoch_start_time = time.time()
    mse_training_running_loss = 0.0
    train_psnr = []

    for noisy_images, clean_images in tqdm(train_loader):
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        
        optimizer.zero_grad()
        
        # with autocast():
        #     outputs = model(noisy_images)
        #     mse_loss = criterion(outputs, clean_images)
            
        outputs = model(noisy_images)
        mse_loss = criterion(outputs, clean_images)
        accuracy = metric(outputs.detach(), clean_images.detach())
        train_psnr.append(accuracy.detach().cpu().numpy())
        
        print(accuracy)
        print(mse_loss)

        scaler.scale(mse_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if accuracy > 30:
            over=True
            noisy_filename =  f'./results/{task}/over_noisy.jpg'
            noisy_image = noisy_images.detach().cpu().squeeze(0)
            noisy_image = denormalize(noisy_image)
            noisy_image = transforms.ToPILImage()(noisy_image)
            noisy_image.save(noisy_filename) 

            denoised_filename =  f'./results/{task}/over.jpg'
            denoised_image = outputs.detach().cpu().squeeze(0)
            denoised_image = denormalize(denoised_image)
            denoised_image = transforms.ToPILImage()(denoised_image)
            denoised_image.save(denoised_filename) 
        if over and accuracy < 10:
            noisy_filename =  f'./results/{task}/under_noisy.jpg'
            noisy_image = noisy_images.detach().cpu().squeeze(0)
            noisy_image = denormalize(noisy_image)
            noisy_image = transforms.ToPILImage()(noisy_image)
            noisy_image.save(noisy_filename) 

            denoised_filename =  f'./results/{task}/under.jpg'
            denoised_image = outputs.detach().cpu().squeeze(0)
            denoised_image = denormalize(denoised_image)
            denoised_image = transforms.ToPILImage()(denoised_image)
            denoised_image.save(denoised_filename) 

        mse_training_running_loss += mse_loss.item() * noisy_images.size(0)
        del noisy_images, clean_images, outputs, mse_loss
    torch.cuda.empty_cache()

    model.eval()
    mse_test_running_loss = 0.0
    test_psnr = []
    with torch.no_grad():
        for noisy_images, clean_images in tqdm(val_loader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            
            # with autocast():
            #     outputs = model(noisy_images)
            #     mse_loss = criterion(outputs, clean_images)

            outputs = model(noisy_images)
            mse_loss = criterion(outputs, clean_images)
            accuracy = metric(outputs.detach(), clean_images.detach())
            test_psnr.append(accuracy.detach().cpu().numpy())
            mse_test_running_loss += mse_loss.item() * noisy_images.size(0)
            del noisy_images, clean_images, outputs, mse_loss
            torch.cuda.empty_cache()

    # current_lr = scheduler.get_last_lr()[0]
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)

    train_psnr = np.array(train_psnr)
    train_mean_psnr = np.mean(train_psnr)
    test_psnr = np.array(test_psnr)
    test_mean_psnr = np.mean(test_psnr)

    training_mse_epoch_loss = mse_training_running_loss / len(train_dataset)
    test_mse_epoch_loss = mse_test_running_loss / len(val_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {training_mse_epoch_loss:.4f}, Train_PSNR: {train_mean_psnr}')
    print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_mse_epoch_loss:.4f}, Test_PSNR: {test_mean_psnr}')
    print(f"1epoch 훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")
    if best_psnr < test_mean_psnr:
        best_psnr = test_mean_psnr
        torch.save(model.state_dict(), f'best_{task}_restormer.pth')
        print(f"{epoch+1}epoch 모델 저장 완료")

    

# 종료 시간 기록
end_time = time.time()

# 소요 시간 계산
training_time = end_time - start_time
minutes = int(training_time // 60)
seconds = int(training_time % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)

# 결과 출력
print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")