import numpy as np

from src.nn.deformable_conv import PackedDeformableConvolution1d

import torch
from torch import nn
from torch.nn import functional as F
from lightning.pytorch import LightningModule

from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class DepthwiseSeparableConvolution1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 2) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, padding='same', bias=False,
                                   dilation=dilation, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, padding='same', bias=False,
                                   dilation=dilation, stride=stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class LITE(LightningModule):

    def __init__(
        self,
        sequence_length: int,
        num_classes: int,
        in_channels: int,
        hidden_channels: int = 32,
    ) -> None:
        """ Create an instance of a LITE [1] model.

        Based on the implementation from https://github.com/MSD-IRIMAS/LITE/blob/main/classifiers/lite.py

        Args:
            sequence_length (int): length of the time series
            num_classes (int): number of classes for the problem
            in_channels (int): number of channels (if the series is univariate than in_channels = 1)
            hidden_channels (int, optional): number of hidden channels to use in inception convolutions. Defaults to 32.

        References:
            [1]
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_channels = hidden_channels

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

        n_convs = 3
        # Hidden Channels
        n_filters = 32
        # Kernel size
        inception_kernel_sizes = [40, 20, 10]
        inception_convolutions = []

        for i in range(len(inception_kernel_sizes)):
            inception_convolutions.append(
                nn.Conv1d(
                    in_channels=self.in_channels, out_channels=n_filters, kernel_size=inception_kernel_sizes[i],
                    stride=1, padding='same', dilation=1, bias=False
                )
            )

        self.inception_convolutions = nn.ModuleList(inception_convolutions)
        self.inception_batchnorm = nn.BatchNorm1d(num_features=96)
        self.inception_activation = nn.ReLU()

        separable_convolutions = []
        separable_kernel_size = 40 // 2

        for i in range(2):
            dilation_rate = 2 ** (i + 1)
            separable_conv = DepthwiseSeparableConvolution1d(
                in_channels=113 if i == 0 else n_filters, out_channels=n_filters,
                kernel_size=separable_kernel_size // (2 ** i), dilation=dilation_rate
            )
            separable_convolutions.append(separable_conv)

        self.separable_convolutions = nn.ModuleList(
            separable_convolutions
            # [
            #     nn.Conv1d(in_channels=96, out_channels=32, kernel_size=10, dilation=2, padding='same', bias=False),
            #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, dilation=4, padding='same', bias=False)
            # ]
        )
        self.separable_batchnorm = nn.BatchNorm1d(num_features=n_filters)
        self.separable_activation = nn.ReLU()
        
        # self.mlp = nn.Sequential(*[
        #     nn.Flatten(),
        #     nn.Linear(in_features=96 * sequence_length, out_features=500),
        #     nn.Linear(in_features=500, out_features=500),
        #     nn.Linear(in_features=500, out_features=num_classes),
        # ])

        self.linear = nn.Linear(in_features=32, out_features=num_classes)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        custom_feature_maps = torch.concat([custom_conv(x) for custom_conv in self.custom_convolutions], dim=1) # Concatenate channel-wise
        custom_feature_maps = self.custom_activation(custom_feature_maps)

        feature_maps = torch.concat([inception_conv(x) for inception_conv in self.inception_convolutions], dim=1) # Concatenate channel-wise again
        feature_maps = self.inception_batchnorm(feature_maps)
        feature_maps = self.inception_activation(feature_maps)

        feature_maps = torch.concat([feature_maps, custom_feature_maps], dim=1)

        # Go over the separable convolution
        for separable_conv in self.separable_convolutions:
            feature_maps = separable_conv(feature_maps) 

            feature_maps = self.separable_batchnorm(feature_maps)
            feature_maps = self.separable_activation(feature_maps)

        feature_maps = torch.mean(feature_maps, dim=-1)

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

        if self.num_classes == 2:
            preds = preds.squeeze(dim=-1)
            y_pred = F.sigmoid(preds).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
            y_true = y.cpu().detach().numpy()

        loss = self.criteria(preds, y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)            

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_f1', f1, on_epoch=True, on_step=False)
        self.log('train_precision', precision, on_epoch=True, on_step=False)
        self.log('train_recall', recall, on_epoch=True, on_step=False)

        # self.train_accs.append(acc)
        # self.train_losses.append(loss.item())

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

        # self.validation_accs.append(acc)
        # self.validation_losses.append(loss.item())

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

