from lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import numpy as np

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


class HInception(LightningModule):

    def __init__(self, sequence_length, in_channels, num_classes) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Build model
        custom_kernels_sizes = [2, 4, 8, 16, 32, 64]
        custom_convolutions = []
        
        for ks in custom_kernels_sizes:
            filter_ = np.ones(shape=(1, self.in_channels, ks))
            indices_ = np.arange(ks)

            filter_[:, :, indices_ % 2 == 0] *= -1 # increasing detection filter

            custom_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())
            
            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        for ks in custom_kernels_sizes:
            filter_ = np.ones(shape=(1, self.in_channels, ks))
            indices_ = np.arange(ks)
            
            filter_[:,:, indices_ % 2 > 0] *= -1 # decreasing detection filter
            
            custom_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())
            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        for ks in [6, 12, 24, 48, 96]:
            filter_ = np.zeros(shape=(1, self.in_channels, ks + ks // 2))
            x_mesh = np.linspace(start=0, stop=1, num=ks//4 + 1)[1:].reshape((-1, 1, 1))
            
            filter_left = x_mesh ** 2
            filter_right = filter_left[::-1]
            
            filter_left = np.transpose(filter_left, (1, 2, 0))
            filter_right = np.transpose(filter_right, (1, 2, 0))
            
            filter_[:, :, 0:ks//4] = -filter_left
            filter_[:, :, ks//4:ks//2] = -filter_right
            filter_[:, :, ks//2:3*ks//4] = 2 * filter_left
            filter_[:, :, 3*ks//4:ks] = 2 * filter_right
            filter_[:, :, ks:5*ks//4] = -filter_left
            filter_[:, :, 5*ks//4:] = -filter_right
            
            custom_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=1,
                                    kernel_size=ks, bias=False, padding='same')
            custom_conv.weight = nn.Parameter(torch.from_numpy(filter_).float())

            for param in custom_conv.parameters():
                param.requires_grad = False
            custom_convolutions.append(custom_conv)

        self.custom_convolutions = nn.ModuleList(custom_convolutions)
        self.custom_activation = nn.ReLU()

        self.inception_module_1 = InceptionModule(in_channels=1, out_channels=32)
        self.inception_module_2 = InceptionModule(in_channels=145, out_channels=32)
        self.inception_module_3 = InceptionModule(in_channels=32 * 4, out_channels=32)

        self.inception_module_4 = InceptionModule(in_channels=32 * 4, out_channels=32)
        self.inception_module_5 = InceptionModule(in_channels=32 * 4, out_channels=32)
        self.inception_module_6 = InceptionModule(in_channels=32 * 4, out_channels=32)

        self.linear = nn.Linear(in_features=32 * 4, out_features=num_classes)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x

        custom_feature_maps = torch.concat([custom_conv(x) for custom_conv in self.custom_convolutions], dim=1) # Concatenate channel-wise
        custom_feature_maps = self.custom_activation(custom_feature_maps)

        feature_maps_1 = torch.cat([self.inception_module_1(x), custom_feature_maps], dim=1)
        feature_maps_2 = self.inception_module_2(feature_maps_1)
        feature_maps_3 = self.inception_module_3(feature_maps_2)

        # raise ValueError()

        feature_maps_3_ = feature_maps_3 + x_ # First Residual

        feature_maps_4 = self.inception_module_4(feature_maps_3_)
        feature_maps_5 = self.inception_module_5(feature_maps_4)
        feature_maps_6 = self.inception_module_6(feature_maps_5)

        feature_maps_6_ = feature_maps_6 + feature_maps_3_ # Second Residual

        feature_maps = torch.mean(feature_maps_6_, dim=-1)

        return self.linear(feature_maps)

    def configure_optimizers(self) -> any:
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, min_lr=0
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)

        y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
        y_true = y.cpu().detach().numpy()

        loss = self.criteria(preds, y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)            

        acc = accuracy_score(y_true, y_pred)
        # f1 = f1_score(y_true, y_pred, average='macro')
        # precision = precision_score(y_true, y_pred, average='macro')
        # recall = recall_score(y_true, y_pred, average='macro')

        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        # self.log('train_f1', f1, on_epoch=True, on_step=False)
        # self.log('train_precision', precision, on_epoch=True, on_step=False)
        # self.log('train_recall', recall, on_epoch=True, on_step=False)

        # self.train_accs.append(acc)
        # self.train_losses.append(loss.item())

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        
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