import sys
sys.path.append('/home/gilberto/projects/defconv_time_series')
import warnings
warnings.filterwarnings('ignore')
from src.models import LITE
from src.utils import load_data, to_torch_dataset, to_torch_loader
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import time
import pandas as pd

# Ensure reproducibility
seed_everything(42, workers=True)

DATASETS = [
    'EOGHorizontalSignal'
]

RESULTS_DIR = '../../../results/'
MODELS_DIR = '../../../models/classification/univariate/'

NUMBER_OF_EXPERIMENTS = 5
NUMBER_OF_EPOCHS = 1500

results_data_dir = {
    'model': [],
    'dataset': [],
    'exp': [],
    'acc': [],
    'f1': [],
    'recall': [],
    'precision': [],
    'time': []
}

for dataset in DATASETS:
    print(f'Loading dataset {dataset}...')
    X_train, y_train, X_test, y_test = load_data(name=dataset, task='classification', split='full')
    sequence_len = X_train.shape[-1]
    
    print('Converting the dataset to torch.DataLoader...')
    train_set, test_set = to_torch_dataset(X_train, y_train, X_test, y_test)
    train_loader, test_loader = to_torch_loader(train_dataset=train_set, test_dataset=test_set, batch_size=64)

    num_classes = len(np.unique(y_train))

    for experiment_number in range(NUMBER_OF_EXPERIMENTS):
        model = LITE(in_channels=1, num_classes=num_classes, sequence_length=sequence_len)

        checkpoint = ModelCheckpoint(
            monitor='train_loss',
            dirpath=f'{MODELS_DIR}',
            filename=f'fcn-{dataset}-{experiment_number}',
            save_top_k=1,
            auto_insert_metric_name=False
        )

        logger = CSVLogger('../../../logs/classification', name=f'lite_ucr_subset_{dataset}')

        trainer = Trainer(
            max_epochs=NUMBER_OF_EPOCHS,
            accelerator='gpu',
            callbacks=[checkpoint],
            logger=logger
        )

        start_time = time.time()
        trainer.fit(model, train_dataloaders=train_loader)
        end_time = time.time()

        results = trainer.test(dataloaders=test_loader, ckpt_path='best')

        results_data_dir['dataset'].append(dataset)
        results_data_dir['model'].append('lite')
        results_data_dir['exp'].append(experiment_number)
        results_data_dir['acc'].append(results[0]['acc'])
        results_data_dir['f1'].append(results[0]['f1'])
        results_data_dir['recall'].append(results[0]['recall'])
        results_data_dir['precision'].append(results[0]['precision'])
        results_data_dir['time'].append(end_time - start_time)

        results_df = pd.DataFrame(results_data_dir)
        results_df.to_csv(f'{RESULTS_DIR}lite.csv', index=False)