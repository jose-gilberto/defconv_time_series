from typing import Any, Tuple
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from lightning.pytorch import LightningModule
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.nn.deformable_conv import PackedDeformableConvolution1d


class DeformableFCN(LightningModule):

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(*[
            nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=30, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=20, padding='same'),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
        ])

        self.defconv = PackedDeformableConvolution1d(
            in_channels=256, out_channels=128, kernel_size=10, padding='same', stride=1,
        )
        
        self.conv2 = nn.Sequential(*[
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
        ])

        self.linear = nn.Linear(in_features=128, out_features=1 if num_classes == 2 else num_classes)

        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()

    def configure_optimizers(self) -> any:
        optimizer = torch.optim.Adam([
            {'params': self.conv_layers.parameters(), 'lr': 1e-4},
            {'params': self.defconv.parameters(), 'lr': 1e-6},
            {'params': self.linear.parameters(), 'lr': 1e-4},
            {'params': self.conv2.parameters(), 'lr': 1e-4},
        ], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-4
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.defconv(x)
        x = self.conv2(x)
        x = torch.mean(x, dim=-1)
        x = self.linear(x)
        return x
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        loss = self.criteria(preds, y.float().to(self.device) if self.num_classes == 2 else y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)            

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_f1', f1, on_epoch=True, on_step=False)
        self.log('train_precision', precision, on_epoch=True, on_step=False)
        self.log('train_recall', recall, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        loss = self.criteria(preds, y.float().to(self.device) if self.num_classes == 2 else y)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)            

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True, on_step=False)
        self.log('val_precision', precision, on_epoch=True, on_step=False)
        self.log('val_recall', recall, on_epoch=True, on_step=False)
        return

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('acc', acc)
        self.log('f1', f1)
        self.log('precision', precision)
        self.log('recall', recall)

        return


class FCN(LightningModule):

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        
        self.num_classes = num_classes

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
        
        self.linear = nn.Linear(in_features=128, out_features=1 if num_classes == 2 else num_classes)

        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()

    def configure_optimizers(self) -> any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=0.0001
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.mean(x, dim=-1)
        x = self.linear(x)
        return x
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        loss = self.criteria(preds, y.float().to(self.device) if self.num_classes == 2 else y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)            

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_f1', f1, on_epoch=True, on_step=False)
        self.log('train_precision', precision, on_epoch=True, on_step=False)
        self.log('train_recall', recall, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        loss = self.criteria(preds, y.float().to(self.device) if self.num_classes == 2 else y)

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)            

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('val_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_f1', f1, on_epoch=True, on_step=False)
        self.log('val_precision', precision, on_epoch=True, on_step=False)
        self.log('val_recall', recall, on_epoch=True, on_step=False)

        return
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('acc', acc)
        self.log('f1', f1)
        self.log('precision', precision)
        self.log('recall', recall)

        return
