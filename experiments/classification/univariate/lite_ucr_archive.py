import sys
sys.path.append('/home/gilberto/projects/defconv_time_series')
import warnings
warnings.filterwarnings('ignore')
from src.models import LITE, FCN
from src.utils import load_data, to_torch_dataset, to_torch_loader
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import time
import pandas as pd
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

# Ensure reproducibility
seed_everything(81, workers=True)

DATASETS = [
    # "ACSF1",
    # "Adiac",
    # "ArrowHead",
    # "Beef",
    # "BeetleFly",
    # "BirdChicken",
    # "BME",
    # "Car",
    # "CBF",
    # "Chinatown",
    # "ChlorineConcentration",
    # "CinCECGTorso",
    "Coffee",
    # "Computers",
    # "CricketX",
    # "CricketY",
    # "CricketZ",
    # "Crop",
    # "DiatomSizeReduction",
    # "DistalPhalanxOutlineCorrect",
    # "DistalPhalanxOutlineAgeGroup",
    # "DistalPhalanxTW",
    # "Earthquakes",
    # "ECG200",
    # "ECG5000",
    # "ECGFiveDays",
    # "ElectricDevices",
    # "EOGHorizontalSignal",
    # "EOGVerticalSignal",
    # "EthanolLevel",
    # "FaceAll",
    # "FaceFour",
    # "FacesUCR",
    # "FiftyWords",
    # "Fish",
    # "FordA",
    # "FordB",
    # "FreezerRegularTrain",
    # "FreezerSmallTrain",
    # "GunPoint",
    # "GunPointAgeSpan",
    # "GunPointMaleVersusFemale",
    # "GunPointOldVersusYoung",
    # "Ham",
    # "HandOutlines",
    # "Haptics",
    # "Herring",
    # "HouseTwenty",
    # "InlineSkate",
    # "InsectEPGRegularTrain",
    # "InsectEPGSmallTrain",
    # "InsectWingbeatSound",
    # "ItalyPowerDemand",
    # "LargeKitchenAppliances",
    # "Lightning2",
    # "Lightning7",
    # "Mallat",
    # "Meat",
    # "MedicalImages",
    # "MiddlePhalanxOutlineCorrect",
    # "MiddlePhalanxOutlineAgeGroup",
    # "MiddlePhalanxTW",
    # "MixedShapesRegularTrain",
    # "MixedShapesSmallTrain",
    # "MoteStrain",
    # "NonInvasiveFetalECGThorax1",
    # "NonInvasiveFetalECGThorax2",
    # "OliveOil",
    # "OSULeaf",
    # "PhalangesOutlinesCorrect",
    # "Phoneme",
    # "PigAirwayPressure",
    # "PigArtPressure",
    # "PigCVP",
    # "Plane",
    # "PowerCons",
    # "ProximalPhalanxOutlineCorrect",
    # "ProximalPhalanxOutlineAgeGroup",
    # "ProximalPhalanxTW",
    # "RefrigerationDevices",
    # "Rock",
    # "ScreenType",
    # "SemgHandGenderCh2",
    # "SemgHandMovementCh2",
    # "SemgHandSubjectCh2",
    # "ShapeletSim",
    # "ShapesAll",
    # "SmallKitchenAppliances",
    # "SmoothSubspace",
    # "SonyAIBORobotSurface1",
    # "SonyAIBORobotSurface2",
    # "StarLightCurves",
    # "Strawberry",
    # "SwedishLeaf",
    # "Symbols",
    # "SyntheticControl",
    # "ToeSegmentation1",
    # "ToeSegmentation2",
    # "Trace",
    # "TwoLeadECG",
    # "TwoPatterns",
    # "UMD",
    # "UWaveGestureLibraryAll",
    # "UWaveGestureLibraryX",
    # "UWaveGestureLibraryY",
    # "UWaveGestureLibraryZ",
    # "Wafer",
    # "Wine",
    # "WordSynonyms",
    # "Worms",
    # "WormsTwoClass",
    # "Yoga",
]

RESULTS_DIR = '../../../results/'
MODELS_DIR = '../../../models/classification/univariate/'

NUMBER_OF_EXPERIMENTS = 1
NUMBER_OF_EPOCHS = 100

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
    train_loader, test_loader = to_torch_loader(train_dataset=train_set, test_dataset=test_set, batch_size=100)

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
            # callbacks=[checkpoint],
            logger=logger
        )

        # start_time = time.time()
        trainer.fit(model, train_dataloaders=train_loader)
        # torch.save(model.state_dict(), './lite.pth')

        # model.load_state_dict(torch.load('./lite.pth'), strict=True)

        # trainer.test(model, train_loader)
  
        # model.eval()
        # with torch.no_grad():
        #     for x, y in train_loader:
        #         x = x.to(model.device)
        #         y = y.to(model.device)

        #         preds = model(x)

        #         if num_classes == 2:
        #             preds = preds.squeeze(dim=-1)
        #             y_pred = F.sigmoid(preds).round()

        #             y_pred = y_pred.cpu().detach().numpy()
        #             y_true = y.cpu().detach().numpy()
        #         else:
        #             y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
        #             y_true = y.cpu().detach().numpy()

        #         acc = accuracy_score(y_true, y_pred)
        #         print(acc)

        # end_time = time.time()

        results = trainer.test(model, dataloaders=train_loader)

        # results_data_dir['dataset'].append(dataset)
        # results_data_dir['model'].append('lite')
        # results_data_dir['exp'].append(experiment_number)
        # results_data_dir['acc'].append(results[0]['acc'])
        # results_data_dir['f1'].append(results[0]['f1'])
        # results_data_dir['recall'].append(results[0]['recall'])
        # results_data_dir['precision'].append(results[0]['precision'])
        # results_data_dir['time'].append(end_time - start_time)

        # results_df = pd.DataFrame(results_data_dir)
        # results_df.to_csv(f'{RESULTS_DIR}lite.csv', index=False)