from aeon.datasets import load_classification, load_regression
from torch.utils.data import TensorDataset, DataLoader, Dataset
from typing import Literal, Tuple, Union
import numpy as np
import torch


def load_data(
    name: str,
    task: Literal["classification", "regression"],
    split: Literal["train", "test", "full"] = "train"
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    assert task in ["classification", "regression"], "Task value must be 'classification' or 'regression'."
    assert split in ["train", "test", "full"], "Split value must be 'train', 'test' or 'full'."

    if task == "classification":
        datasets = []

        if split == "train" or split == "full":
            X_train, y_train = load_classification(name=name, split="TRAIN", return_metadata=False)
            y_train = y_train.astype(int)
            
            classes = np.unique(y_train)
            
            #TODO: fix with a dictionary based mapping
            
            for i in range(len(y_train)):
                y_train[i] = np.where(classes == y_train[i])[0][0]

            datasets.append(X_train)
            datasets.append(y_train.astype(int))
            
        if split == "test" or split == "full":
            X_test, y_test = load_classification(name=name, split="TEST", return_metadata=False)
            y_test = y_test.astype(int)
            classes = np.unique(y_test)
            
            #TODO: fix with a dictionary based mapping
            for i in range(len(y_test)):
                y_test[i] = np.where(classes == y_test[i])[0][0]
            
            datasets.append(X_test)
            datasets.append(y_test.astype(int))

        datasets = tuple(datasets)
        return datasets
    else:
        raise NotImplemented
    
def to_torch_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[TensorDataset, TensorDataset]:
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long()
    )
    
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    return train_dataset, test_dataset
    

def to_torch_loader(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = False
) -> Tuple[DataLoader, DataLoader]:

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )
    
    return train_loader, test_loader
