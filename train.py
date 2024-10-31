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
from torchvision.transforms import Compose, ToPILImage, ToTensor
from torch.utils.data import Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Restormer
from torchvision.transforms import CenterCrop, Resize
from PIL import Image

# 시작 시간 기록
start_time = time.time()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class CustomDataset(Dataset):
    def __init__(self, clean_image_paths, noisy_image_paths, transform=None):
        self.clean_image_paths = [os.path.join(clean_image_paths, x) for x in os.listdir(clean_image_paths)]
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform
        self.center_crop = CenterCrop(224)
        self.resize = Resize((224, 224))

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
            else:
                pass
        
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
        
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 하이퍼파라미터 설정
num_epochs = 1
batch_size = 8
learning_rate = 0.0005

# 데이터셋 경로
noisy_image_paths = '/data/hyeokseung1208/unid/data/Training/noisy'
clean_image_paths = '/data/hyeokseung1208/unid/data/Training/clean'

# 데이터셋 로드 및 전처리
train_transform = Compose([
    ToTensor(),
])

# 커스텀 데이터셋 인스턴스 생성
train_dataset = CustomDataset(clean_image_paths, noisy_image_paths, transform=train_transform)

# 데이터 로더 설정
num_cores = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=int(num_cores/2), shuffle=True)

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Restormer 모델 인스턴스 생성 및 GPU로 이동
model = Restormer().to(device)
# model.load_state_dict(torch.load('best_Restormer_200.pth'))
# model.apply(weights_init)

# 손실 함수와 최적화 알고리즘 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.L1Loss()
scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 모델의 파라미터 수 계산
total_parameters = count_parameters(model)
print("Total Parameters:", total_parameters)

# 모델 학습
model.train()
best_loss = 1000

# GPU 사용량 확인
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

for epoch in range(num_epochs):
    model.train()
    epoch_start_time = time.time()
    mse_running_loss = 0.0
    
    for noisy_images, clean_images in train_loader:
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(noisy_images)
            mse_loss = criterion(outputs, clean_images)
        
        scaler.scale(mse_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        mse_running_loss += mse_loss.item() * noisy_images.size(0)

    current_lr = scheduler.get_last_lr()[0]
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)
    hours = int(minutes // 60)
    minutes = int(minutes % 60)

    mse_epoch_loss = mse_running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, MSE Loss: {mse_epoch_loss:.4f}, Lr: {current_lr:.8f}')
    print(f"1epoch 훈련 소요 시간: {hours}시간 {minutes}분 {seconds}초")

    if mse_epoch_loss < best_loss:
        best_loss = mse_epoch_loss
        torch.save(model.state_dict(), 'best_Restormer_400_final.pth')
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