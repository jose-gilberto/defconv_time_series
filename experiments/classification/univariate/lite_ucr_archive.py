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
import matplotlib.pyplot as plt
from torch import nn

# Ensure reproducibility
# seed_everything(81, workers=True)

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
    # "Shapelettrain_acc",
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
    train_loader, test_loader = to_torch_loader(train_dataset=train_set, test_dataset=test_set, batch_size=64)

    num_classes = len(np.unique(y_train))

    for experiment_number in range(NUMBER_OF_EXPERIMENTS):
        # model = LITE(in_channels=1, num_classes=num_classes, sequence_length=sequence_len)
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


        torch.autograd.set_detect_anomaly(True)
        # start_time = time.time()
        # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        # end_time = time.time()
        model.to(torch.device('cuda'))
        model.train()

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, min_lr=0
        )

        criteria = nn.CrossEntropyLoss()

        lowest_loss = None

        for epoch in range(1500):
            loss_epoch = []
            acc_epoch = []

            for x, y in train_loader:
                x = x.to(model.device)
                y = y.to(model.device)

                preds = model(x)
                
                y_pred = torch.argmax(preds, dim=1).cpu().detach().numpy()
                y_true = y.cpu().detach().numpy()

                loss = criteria(preds.squeeze(dim=-1), y)
                acc = accuracy_score(y_true, y_pred)

                loss_epoch.append(loss.item())
                acc_epoch.append(acc)

            loss_epoch, acc_epoch = np.array(loss_epoch), np.array(acc_epoch)

            # if lowest_loss is None or lowest_loss > loss_epoch.mean():
            #     lowest_loss = loss_epoch.mean()
            #     torch.save(model.state_dict(), 'lite.pth')

            print(f'Epoch {epoch} - Train Loss: {loss_epoch.mean()} - Acc {acc_epoch.mean()}')

            loss.backward()

            optimizer.step()
            scheduler.step(loss_epoch.mean())


        # model.load_state_dict(torch.load('./lite.pth'))

        test_ypred = []
        test_ytrue = []

        model.eval()
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(model.device)
                y = y.to(model.device)

                preds = model(x)
                
                y_pred = torch.argmax(nn.functional.softmax(preds, dim=1), dim=1).cpu().detach().numpy()
                y_true = y.cpu().detach().numpy()

                print(preds)
                print(y_pred)
                print(y_true)

                test_ypred.extend(y_pred)
                test_ytrue.extend(y_true)

            print(f'Accuracy on Test {accuracy_score(test_ypred, test_ytrue)}')

        # results = trainer.test(model, dataloaders=train_loader)

        # # plt.plot(list(range(NUMBER_OF_EPOCHS)), model.train_losses)
        # # plt.plot(list(range(NUMBER_OF_EPOCHS + 1)), model.validation_losses)
        # # plt.show()

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