import sys
sys.path.append('/home/gilberto/projects/defconv_time_series')
import warnings
warnings.filterwarnings('ignore')
from src.models import Inception, InceptionTime
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
    "ACSF1",
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
]

RESULTS_DIR = '../../../results/'
MODELS_DIR = '../../../models/classification/univariate/'

NUMBER_OF_EXPERIMENTS = 5
NUMBER_OF_EPOCHS = 1000

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

    print('Converting the dataset to torch.DataLoader...')
    train_set, test_set = to_torch_dataset(X_train, y_train, X_test, y_test)
    train_loader, test_loader = to_torch_loader(train_dataset=train_set, test_dataset=test_set)

    num_classes = len(np.unique(y_train))

    for experiment_number in range(NUMBER_OF_EXPERIMENTS):
        models = []
        inception_training_time = 0

        for i in range(5):
            inception = Inception(in_channels=1, hidden_channels=32, num_classes=num_classes)

            checkpoint = ModelCheckpoint(
                monitor='train_loss',
                dirpath=f'{MODELS_DIR}',
                filename=f'inception-{experiment_number}-{i}-{dataset}-{experiment_number}',
                save_top_k=1,
                auto_insert_metric_name=False
            )

            logger = CSVLogger('../../../logs/classification', name=f'inception-{experiment_number}-{i}_ucr_subset_{dataset}')

            trainer = Trainer(
                max_epochs=NUMBER_OF_EPOCHS,
                accelerator='gpu',
                callbacks=[checkpoint],
                logger=logger
            )

            start_time = time.time()
            trainer.fit(inception, train_dataloaders=train_loader)
            end_time = time.time()
            
            inception_training_time += (end_time - start_time)
            
            models.append(inception)
        
        inceptiontime = InceptionTime(models=models, num_classes=num_classes, device=inception.device)
        results = inceptiontime.test(dataloader=test_loader)

        # start_time = time.time()
        # end_time = time.time()

        # results = trainer.test(dataloaders=test_loader, ckpt_path='best')

        results_data_dir['dataset'].append(dataset)
        results_data_dir['model'].append('inceptiontime')
        results_data_dir['exp'].append(experiment_number)
        results_data_dir['acc'].append(results['acc'])
        results_data_dir['f1'].append(results['f1'])
        results_data_dir['recall'].append(results['recall'])
        results_data_dir['precision'].append(results['precision'])
        results_data_dir['time'].append(inception_training_time)

        results_df = pd.DataFrame(results_data_dir)
        results_df.to_csv(f'{RESULTS_DIR}inceptiontime.csv', index=False)