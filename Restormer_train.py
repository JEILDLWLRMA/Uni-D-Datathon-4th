import torch.nn.functional as F
import random
import numpy as np
import cv2
import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomVerticalFlip, ToPILImage
from os.path import join
from os import listdir
from torchsummary import summary
import torchvision
import time
import zipfile
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from Restormer_model import Restormer

# 시작 시간 기록
start_time = time.time()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 커스텀 데이터셋 클래스 정의
class CustomDataset(data.Dataset):
    def __init__(self, noisy_image_paths, clean_image_paths, patch_size = 128, transform=None):
        self.clean_image_paths = [join(clean_image_paths, x) for x in listdir(clean_image_paths)]
        self.noisy_image_paths = [join(noisy_image_paths, x) for x in listdir(noisy_image_paths)]
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        # 이미지 불러오기
        noisy_image = load_img(self.noisy_image_paths[index])
        clean_image = load_img(self.clean_image_paths[index])
        
        H, W, _ = clean_image.shape

        # 이미지 랜덤 크롭
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        noisy_image = noisy_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        clean_image = clean_image[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        # transform 적용
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

# 하이퍼파라미터 설정
num_epochs = 100
batch_size = 16
learning_rate = 0.0005

# 데이터셋 경로
noisy_image_paths = '/local_datasets/team08/train/scan'
clean_image_paths = '/local_datasets/team08/train/clean'


# 데이터셋 로드 및 전처리
train_transform = Compose([
    ToPILImage(),
    ToTensor(),
])

# 커스텀 데이터셋 인스턴스 생성
train_dataset = CustomDataset(noisy_image_paths, clean_image_paths, transform=train_transform)
import os

num_cores = os.cpu_count()
#######

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=int(num_cores/2), shuffle=True)

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Restormer 모델 인스턴스 생성 및 GPU로 이동
model = Restormer().to(device)
# model.load_state_dict(torch.load('best_Restormer_200.pth'))
#model.apply(weights_init)
# 손실 함수와 최적화 알고리즘 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion =  nn.L1Loss()
scaler = GradScaler()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 모델의 파라미터 수 계산
total_parameters = count_parameters(model)
print("Total Parameters:", total_parameters)

# 모델 학습
model.train()

best_loss = 0.0306

#gpu 사용량 확인
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"GPU 사용 가능. GPU 수: {device_count}")
    for i in range(device_count):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i} - 이름: {gpu_properties.name}, 메모리: {gpu_properties.total_memory}MB")
        allocated_memory = torch.cuda.memory_allocated() / 1024**2  # MB 단위로 변환
        print(f"현재 할당된 GPU 메모리: {allocated_memory:.2f} MB")
else:
    print("GPU 사용 불가능")
        

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)  
for epoch in range(num_epochs):
    model.train()
    epoch_start_time=time.time()#시간확인용
    mixed_running_loss=0.0
    mse_running_loss = 0.0
    
    for noisy_images, clean_images in train_loader:
        
        
        noisy_images=noisy_images.to(device)
        clean_images=clean_images.to(device)
        #pctloss = PerceptualLoss().to(device).eval()
        
        optimizer.zero_grad()
        
        
        with autocast():
            # 모델 순전파
            
            outputs = model(noisy_images)
            mse_loss=criterion(outputs,clean_images)#기존과 확인하려고 넣은 loss 학습과는 무관
            #mixed_loss = 0.9*mse_loss+0.1*pctloss.forward(noisy_images-outputs,clean_images)

        # 역전파 및 가중치 업데이트
        scaler.scale(mse_loss).backward()
        #Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        #mixed_running_loss += mixed_loss.item() * noisy_images.size(0)
        mse_running_loss+=mse_loss.item()* noisy_images.size(0)#기존과 확인하려고 넣은 loss 학습과는 무관

    current_lr = scheduler.get_last_lr()[0]
    epoch_end_time=time.time()#시간 확인용
    epoch_time=epoch_end_time-epoch_start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)

    mse_epoch_loss = mse_running_loss / len(train_dataset)
    mixed_epoch_loss=mixed_running_loss / len(train_dataset)#기존과 확인하려고 넣은 loss 학습과는 무관
    print(f'Epoch {epoch+1}/{num_epochs}, MSE Loss: {mse_epoch_loss:.4f}, Mixed Loss: {mixed_epoch_loss:.4f},Lr:{current_lr:.8f}')
    print(f"1epoch 훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")

# 현재 epoch의 loss가 최소 loss보다 작으면 모델 갱신
    if mse_epoch_loss < best_loss:
        best_loss = mse_epoch_loss
        torch.save(model.state_dict(), 'best_Restormer_400_final.pth')
        print(f"{epoch+1}epoch 모델 저장 완료")

# 종료 시간 기록
end_time = time.time()

# 종료 시간 기록
end_time = time.time()

# 소요 시간 계산
training_time = end_time - start_time

# 시, 분, 초로 변환
minutes = int(training_time // 60)
seconds = int(training_time % 60)
hours = int(minutes // 60)
minutes = int(minutes % 60)

# 결과 출력
print(f"훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")