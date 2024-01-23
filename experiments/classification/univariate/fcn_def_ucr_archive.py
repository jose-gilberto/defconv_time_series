import sys
sys.path.append('/home/gbarbosa/Projects/defconvts')
from src.models import DeformableFCN
import warnings
warnings.filterwarnings('ignore')
from src.utils import load_data, to_torch_dataset, to_torch_loader
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd

# Ensure reproducibility
seed_everything(42, workers=True)

DATASETS = [
    'ArrowHead',
    'BeetleFly',
    'Car',
    'Earthquakes',
    'FaceAll',
    'FordB',
    'Ham',
    'InlineSkate',
    'InsectWingbeatSound',
    'Lightning7',
    'MoteStrain',
    'NonInvasiveFetalECGThorax2',
    'OliveOil',
    'ProximalPhalanxTW',
    'TwoPatterns',
    'Wine',
    'WordSynonyms',
    'Yoga',
    'EOGVerticalSignal',
    'FreezerSmallTrain',
    'GunPointOldVersusYoung',
]

RESULTS_DIR = '../../../results/'
MODELS_DIR = '../../../models/classification/univariate/'

NUMBER_OF_EXPERIMENTS = 1
NUMBER_OF_EPOCHS = 1000

results_data_dir = {
    'model': [],
    'dataset': [],
    'exp': [],
    'acc': [],
    'f1': []
}

for dataset in DATASETS:

    print(f'Loading dataset {dataset}...')
    X_train, y_train, X_test, y_test = load_data(name=dataset, task='classification', split='full')
    
    print('Converting the dataset to torch.DataLoader...')
    train_set, test_set = to_torch_dataset(X_train, y_train, X_test, y_test)
    train_loader, test_loader = to_torch_loader(train_dataset=train_set, test_dataset=test_set)

    num_classes = len(np.unique(y_train))

    for experiment_number in range(NUMBER_OF_EXPERIMENTS):
        model = DeformableFCN(in_dim=1, num_classes=num_classes)

        checkpoint = ModelCheckpoint(
            monitor='train_loss',
            dirpath=f'{MODELS_DIR}',
            filename=f'deffcn-{dataset}-{experiment_number}',
            save_top_k=1,
            auto_insert_metric_name=False
        )
        trainer = Trainer(
            max_epochs=NUMBER_OF_EPOCHS,
            accelerator='gpu',
            callbacks=[checkpoint]
        )
        
        trainer.fit(model, train_dataloaders=train_loader)
        results = trainer.test(dataloaders=test_loader, ckpt_path='best')
        
        results_data_dir['dataset'].append(dataset)
        results_data_dir['model'].append('deffcn')
        results_data_dir['exp'].append(experiment_number)
        results_data_dir['acc'].append(results[0]['acc'])
        results_data_dir['f1'].append(results[0]['f1'])

        results_df = pd.DataFrame(results_data_dir)
        results_df.to_csv(f'{RESULTS_DIR}deffcn.csv', index=False)