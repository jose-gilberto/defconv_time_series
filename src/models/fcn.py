from typing import Any, Tuple
import torch
from torch import nn
from lightning.pytorch import LightningModule
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.nn.deformable_conv import PackedDeformableConvolution1d


class DeformableFCN(LightningModule):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(*[
            nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            
            PackedDeformableConvolution1d(
                in_channels=128, out_channels=128, kernel_size=5, padding='same', stride=1
            ),
        ])
        
        self.linear = nn.Linear(in_features=128, out_features=num_classes)
        
        self.softmax = nn.Softmax(dim=1)

        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCELoss()

    def configure_optimizers(self) -> any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001
        )
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'train_loss'
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.mean(x, dim=-1)
        x = self.linear(x)
        return self.softmax(x)
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        
        loss = self.criteria(preds, y)
        self.log('train_loss', loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        
        loss = self.criteria(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        
        y_pred = torch.argmax(preds, dim=1).detach().numpy()
        y_true = y.detach().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        
        
        self.log('accuracy', acc, prog_bar=True)
        self.log('f1 score', f1, prog_bar=True)
        self.log('recall', recall, prog_bar=True)
        self.log('precision', precision, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        
        y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
        y_true = y.cpu().detach().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        
        self.log('acc', acc)
        self.log('f1', f1)
        
        return


class FCN(LightningModule):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(*[
            nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
        ])
        
        self.linear = nn.Linear(in_features=128, out_features=num_classes)
        
        self.softmax = nn.Softmax(dim=1)

        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCELoss()

    def configure_optimizers(self) -> any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001
        )
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'train_loss'
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.mean(x, dim=-1)
        x = self.linear(x)
        return self.softmax(x)
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        
        loss = self.criteria(preds, y)
        self.log('train_loss', loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        
        loss = self.criteria(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        
        y_pred = torch.argmax(preds, dim=1).detach().numpy()
        y_true = y.detach().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        
        
        self.log('accuracy', acc, prog_bar=True)
        self.log('f1 score', f1, prog_bar=True)
        self.log('recall', recall, prog_bar=True)
        self.log('precision', precision, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        
        y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
        y_true = y.cpu().detach().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        
        self.log('acc', acc)
        self.log('f1', f1)
        
        return
    
