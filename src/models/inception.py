from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Tuple, List, Dict
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def noop(x: torch.Tensor) -> torch.Tensor:
    return x

class InceptionModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = [40, 20, 10],
                 bottleneck: bool = True) -> None:
        super().__init__()
        
        self.kernel_sizes = kernel_size
        bottleneck = bottleneck if in_channels > 1 else False
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False, padding='same') if bottleneck else noop
        
        self.convolutions = nn.ModuleList([
            nn.Conv1d(out_channels if bottleneck else in_channels,
                      out_channels,
                      kernel_size=k,
                      padding='same',
                      bias=False) for k in self.kernel_sizes
        ])
        self.maxconv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1),
                                       nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same', bias=False)])
        self.batchnorm = nn.BatchNorm1d(out_channels * 4)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x
        x = self.bottleneck(x)
        x = torch.cat([conv(x) for conv in self.convolutions] + [self.maxconv(x_)], dim=1)
        return self.activation(x)


class InceptionBlock(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 residual: bool = True,
                 depth: int = 3) -> None:
        super().__init__()
        
        self.residual = residual
        self.depth = depth
        
        self.activation = nn.ReLU()
        
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        
        for d in range(depth):
            self.inception.append(InceptionModule(
                in_channels=in_channels if d == 0 else out_channels * 4,
                out_channels=out_channels
            ))
            if self.residual and d % 3 == 2:
                c_in, c_out = in_channels if d == 2 else out_channels * 4, out_channels * 4
                self.shortcut.append(
                    nn.BatchNorm1d(c_in) if c_in == c_out else nn.Sequential(*[
                        nn.Conv1d(c_in, c_out, kernel_size=1, padding='same'),
                        nn.BatchNorm1d(c_out)
                    ])
                )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                res = x = self.activation(x + self.shortcut[d // 3](res))

        return x


class Inception(LightningModule):

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int) -> None:
        super().__init__()

        self.inception_block = InceptionBlock(in_channels, hidden_channels, depth=3)
        self.linear = nn.Linear(hidden_channels * 4, 1 if num_classes == 2 else num_classes)
        self.num_classes = num_classes
        
        self.criteria = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception_block(x)
        x = torch.mean(x, dim=-1)
        return self.linear(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 2:
            proba = nn.functional.sigmoid(self(x))
            proba = torch.hstack([1 - proba, proba])
        else:
            proba = nn.functional.softmax(self(x))
            
        # here the shape of proba is (batch_size, class_num)
        proba = proba / torch.sum(proba, dim=1, keepdim=True)
        return proba
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        
        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = nn.functional.sigmoid(preds).round()
            
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
            y_pred = nn.functional.sigmoid(preds).round()
            
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

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        preds = self(x)

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = nn.functional.sigmoid(preds).round()

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


class InceptionTime:

    def __init__(self, models: List[Inception], num_classes: int, device: torch.device) -> None:
        self.models = models
        self.num_classes = num_classes
        self.device = device

    def _predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = [[] for _ in self.models]

        for i, model in enumerate(self.models):
            proba = model.predict_proba(x)
            logits[i].extend(proba.tolist())

        logits = torch.tensor(logits)
        return torch.mean(logits, dim=0) # Shape: (batch_size, class_num)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return np.array([
            np.random.choice(np.flatnonzero(prob == prob.max())) for prob in self._predict_proba(x).numpy()
        ])

    def test(self, dataloader: DataLoader) -> Dict[str, float]:
        y_true = []
        y_pred = []

        for model in self.models:
            model.eval()

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.predict(x)

                y_true.extend(y.cpu().detach().tolist())
                y_pred.extend(preds.tolist())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        return {
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }
