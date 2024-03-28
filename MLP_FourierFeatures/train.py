import torch
from torch import nn
from einops import rearrange

from opt import get_opts

# datasets
from dataset import ImageDataset
from torch.utils.data import DataLoader

# models
from models import PE, MLP

# metrics
from metrics import mse, psnr

# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim


class CoordMLPSystem(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # hparams 값 확인을 위한 print 문 추가
        print(f"use_pe: {hparams.use_pe}, arch: {hparams.arch}")

        if hparams.use_pe:
            P = torch.cat([torch.eye(2)*2**i for i in range(10)], 1)  # (2, 2*10)
            self.pe = PE(P)

        if hparams.arch in ['relu', 'bacon']:
            act = hparams.arch
            if hparams.use_pe:
                n_in = self.pe.out_dim
            else:
                n_in = 2
            self.mlp = MLP(n_in=n_in, act=act, act_trainable=hparams.act_trainable)

        elif hparams.arch == 'ff':
            P = hparams.sc * torch.normal(torch.zeros(2, 256), torch.ones(2, 256))  # (2, 256)
            self.pe = PE(P)
            self.mlp = MLP(n_in=self.pe.out_dim)

    def forward(self, x):
        if self.hparams.use_pe or self.hparams.arch == 'ff':
            x = self.pe(x)
        return self.mlp(x)

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        uv = batch['uv'].to(device)
        rgb = batch['rgb'].to(device)
        
        optimizer.zero_grad()
        rgb_pred = model(uv)
        loss = mse(rgb_pred, rgb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            uv = batch['uv'].to(device)
            rgb = batch['rgb'].to(device)
            
            rgb_pred = model(uv)
            loss = mse(rgb_pred, rgb)

            total_loss += loss.item()
    return total_loss / len(val_loader)

def main(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 및 데이터 로더 준비
    train_dataset = ImageDataset(hparams.image_path, hparams.img_wh, 'train')
    val_dataset = ImageDataset(hparams.image_path, hparams.img_wh, 'val')
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=hparams.batch_size, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=4, batch_size=hparams.batch_size, pin_memory=True)

    # 모델, 최적화기 및 학습률 스케줄러 설정
    model = CoordMLPSystem(hparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, hparams.num_epochs, hparams.lr/1e2)

    # 학습 및 검증 루프
    for epoch in range(hparams.num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)