from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Tuple, List, Dict
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import functional as F


def noop(x: torch.Tensor) -> torch.Tensor:
    return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.convolution_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=8, padding='same', bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(out_channels)
        self.convolution_2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding='same', bias=False)
        self.batch_norm_2 = nn.BatchNorm1d(out_channels)
        self.convolution_3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False)
        self.batch_norm_3 = nn.BatchNorm1d(out_channels)
        
        self.activation = nn.ReLU()
        
        self.shortcut = nn.BatchNorm1d(in_channels) if in_channels == out_channels else nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same', bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.activation(self.batch_norm_1(self.convolution_1(x)))
        x_ = self.activation(self.batch_norm_2(self.convolution_2(x_)))
        x_ = self.activation(self.batch_norm_3(self.convolution_3(x_)))
        return x_ + self.shortcut(x)
    
class ResNet(LightningModule):

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
        )

        self.linear = nn.Linear(128, 1 if num_classes == 2 else num_classes)
        self.num_classes = num_classes
        
        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.mean(x, dim=-1)
        return self.linear(x)
    
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
