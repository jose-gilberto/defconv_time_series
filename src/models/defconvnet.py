from typing import Any, Tuple
import torch
from torch import nn
from lightning.pytorch import LightningModule
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.nn.deformable_conv import PackedDeformableConvolution1d
from src.models.inception import InceptionModule, InceptionBlock


class ParallelConvolutions(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.large_convolution = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, bias=False, padding='same', stride=1, kernel_size=40)
        self.medium_convolution = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, bias=False, padding='same', stride=1, kernel_size=20)

        self.batch_normalization = nn.BatchNorm1d(num_features=out_channels * 2)
        self.activation = nn.PReLU()
        
    def forward(self, x):
        x = torch.cat([self.large_convolution(x), self.medium_convolution(x)], dim=1)
        x = self.batch_normalization(x)
        x = self.activation(x)
        return x

class DefConvNet(LightningModule):

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(*[
            ParallelConvolutions(in_channels=1, out_channels=32),
            PackedDeformableConvolution1d(
                in_channels=32*2, out_channels=128, kernel_size=20, padding='same', stride=1
            ),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=10, padding='same'),
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(),
        ])

        self.linear = nn.Linear(in_features=128, out_features=1 if num_classes == 2 else num_classes)

        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()

    def configure_optimizers(self) -> any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer

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