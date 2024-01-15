from typing import Any, Tuple
import torch
from torch import nn
from lightning.pytorch import LightningModule
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MLP(LightningModule):
    def __init__(self, in_features: int, in_dim: int, num_classes: int) -> None:
        super().__init__()

        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(*[
            nn.Dropout(p=0.1),
            nn.Linear(in_features=in_features * in_dim, out_features=500),
            nn.ReLU(),
            
            nn.Dropout(p=0.2),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            
            nn.Dropout(p=0.2),
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            
            nn.Linear(in_features=500, out_features=num_classes),
            nn.Dropout(p=0.3),
            nn.Softmax(dim=1),
        ])

        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCELoss()

    def configure_optimizers(self) -> any:
        optimizer = torch.optim.Adadelta(self.parameters(), lr=1e-1, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=200, min_lr=0.1
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)
        
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
        
        y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
        y_true = y.cpu().detach().numpy()
        
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
        # recall = recall_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred)
        
        
        self.log('acc', acc)
        self.log('f1_score', f1)
        # self.log('recall', recall)
        # self.log('precision', precision)
        
        return
    
