import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from src import preprocessing, training
from utils.data_loading import get_my_data
from src.evaluation import evaluate_model
from utils.stratification import stratify_y
import random

# Parameters
showPlot = False
save_predictions = False
use_chromatography_column = False
keep_all_chromatographic_columns_in_preprocessing = False
split_train_test_by_experiment = False


if __name__ == "__main__":
    # Load data
    print("Loading data")
    X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()


    experiments = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0009',
       '0010', '0013', '0014', '0015', '0019', '0020', '0021', '0023',
       '0025', '0027', '0028', '0029', '0030', '0031', '0032', '0033',
       '0034', '0036', '0037', '0038', '0039', '0040', '0041', '0042',
       '0044', '0045', '0046', '0047', '0048', '0049', '0050', '0051',
       '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059',
       '0060', '0065', '0066', '0067', '0068', '0069', '0070', '0071',
       '0072', '0073', '0074', '0075', '0076', '0077', '0078', '0079',
       '0080', '0081', '0082', '0083', '0084', '0085', '0086', '0087',
       '0088', '0089', '0090', '0091', '0092', '0093', '0094', '0095',
       '0096', '0097', '0098', '0099', '0100', '0101', '0102', '0103',
       '0104', '0105', '0106', '0107', '0108', '0109', '0110', '0111',
       '0112', '0113', '0114', '0115', '0116', '0117', '0118', '0119',
       '0120', '0121', '0122', '0123', '0124', '0125', '0126', '0127',
       '0128', '0129', '0130', '0131', '0132', '0133', '0134', '0135',
       '0136', '0137', '0138', '0139', '0140', '0141', '0142', '0143',
       '0144', '0145', '0146', '0147', '0148', '0149', '0150', '0151',
       '0152', '0153', '0154', '0155', '0156', '0157', '0158', '0159',
       '0160', '0161', '0162', '0163', '0164', '0165', '0166', '0167',
       '0168', '0169', '0170', '0171', '0172', '0173', '0174', '0175',
       '0176', '0177', '0178', '0179', '0180', '0181', '0182', '0183',
       '0184', '0185', '0186', '0187', '0188', '0189', '0190', '0191',
       '0192', '0193', '0194', '0195', '0196', '0197', '0198', '0199',
       '0200', '0201', '0202', '0203', '0204', '0205', '0225', '0228',
       '0231', '0232', '0233', '0234', '0235', '0236', '0237',
       '0238', '0239', '0240', '0241', '0242', '0243', '0244', '0245',
       '0246', '0247', '0248', '0249', '0250', '0251', '0252', '0253',
       '0254', '0255', '0256', '0257', '0258', '0259', '0260', '0261',
       '0262', '0263', '0264', '0265', '0266', '0267', '0268', '0269',
       '0270', '0271', '0272', '0273', '0274', '0275', '0276', '0277',
       '0278', '0279', '0280', '0281', '0282', '0283', '0284', '0285',
       '0286', '0287', '0288', '0289', '0290', '0291', '0292', '0293',
       '0294', '0295', '0310', '0311', '0312', '0313', '0314', '0315',
       '0316', '0317', '0318', '0319', '0320', '0321', '0322', '0323',
       '0324', '0325', '0326', '0327', '0328', '0329', '0330', '0331',
       '0332', '0333', '0334', '0335', '0336', '0337', '0338', '0339',
       '0340', '0341', '0342', '0343', '0344', '0345', '0346', '0347',
       '0348', '0349', '0350', '0351', '0352', '0353', '0354', '0355',
       '0356', '0357', '0358', '0359', '0360', '0362', '0363',
       '0364', '0365', '0366', '0367', '0368', '0369', '0370', '0371',
       '0372', '0373', '0374', '0375', '0376', '0377', '0378', '0379',
       '0380', '0381', '0382', '0383', '0384', '0385', '0386', '0387',
       '0388', '0389', '0390', '0391']

    if not use_chromatography_column:
        X = X.sort_values("id")
        number_columns = X["id"].str[0:4].drop_duplicates().values
        experiment_data = {}
        number_molecules = 0
        for value in number_columns:
            experiment = int(X[X["id"].str.startswith(value)].shape[0])
            experiment_data[value] = (number_molecules, number_molecules + experiment)
            number_molecules = number_molecules + experiment

    if experiments != ["all"] and not use_chromatography_column:
        experiment_key_values = [(key, experiment_data.get(key)) for key in experiments]
    elif experiments == ["all"] and not use_chromatography_column:
        experiment_key_values = [(key, value) for key, value in experiment_data.items()]
    else:
        experiment_key_values = [("all", 0)]

    for key, values in experiment_key_values:
        if not use_chromatography_column:
            X_ex = X[values[0]:values[1]]
            y_ex = y[values[0]:values[1]]
        else:
            X_ex = X
            y_ex = y
        # Create results directory if it doesn't exist
        if not os.path.exists('./results'):
            os.makedirs('./results')

        fold = 0
        if split_train_test_by_experiment:
            X_ex["id"] = X_ex["id"].str[0:4]
            unique_experiments = X_ex["id"].unique()
            train_experiments = random.sample(list(unique_experiments), k=int(len(unique_experiments) * 0.8))
            train_data = X_ex[X_ex['id'].isin(train_experiments)]
            test_data = X_ex[~X_ex['id'].isin(train_experiments)]
            train_split_X = train_data.drop(["id", "rt"], axis=1).astype('float32')
            test_split_X = test_data.drop(["id", "rt"], axis=1).astype('float32')
            train_split_y = np.array(train_data["rt"]).astype('float32').flatten()
            test_split_y = np.array(test_data["rt"]).astype('float32').flatten()

        else:
            X_ex = X_ex.drop(["id", "rt"], axis=1).astype('float32')
            y = np.array(y).astype('float32').flatten()
            train_split_X, test_split_X, train_split_y, test_split_y = train_test_split(X_ex, y_ex, test_size=0.2,
                                                                                        random_state=42)

        features_list = ["descriptors", "fingerprints", "all"]
        for features in features_list:
            # Preprocess X
            print(f"experiment-{key}-{features}")
            if not keep_all_chromatographic_columns_in_preprocessing and use_chromatography_column:
                (preprocessed_train_split_X, preprocessed_test_split_X) = preprocessing.preprocess_X_except_chromatography(
                     usp_columns=usp_columns,
                     chromatography_columns=chromatography_columns,
                     descriptors_columns=descriptors_columns,
                     fingerprints_columns=fingerprints_columns,
                     train_X=train_split_X,
                     train_y=train_split_y,
                     test_X=test_split_X,
                     test_y=test_split_y,
                     features=features
                )

            elif keep_all_chromatographic_columns_in_preprocessing and use_chromatography_column:
                preprocessed_train_split_X, preprocessed_test_split_X = preprocessing.preprocess_X_except_usp(
                    usp_columns=usp_columns,
                    chromatography_columns=chromatography_columns,
                    descriptors_columns=descriptors_columns,
                    fingerprints_columns=fingerprints_columns,
                    train_X=train_split_X,
                    train_y=train_split_y,
                    test_X=test_split_X,
                    test_y=test_split_y,
                    features=features
                )
            else:
                preprocessed_train_split_X, preprocessed_test_split_X = preprocessing.preprocess_X(
                    descriptors_columns=descriptors_columns,
                    fingerprints_columns=fingerprints_columns,
                    train_X=train_split_X,
                    train_y=train_split_y,
                    test_X=test_split_X,
                    test_y=test_split_y,
                    features=features
                )

            preprocessed_train_split_y, preprocessed_test_split_y, preproc_y = preprocessing.preprocess_y(
                train_y=train_split_y, test_y=test_split_y
            )
            columns_deleted_preprocessing = [column for column in train_split_X.columns if column not in preprocessed_train_split_X]

            print("Param search")
            trained_dnn = training.optimize_and_train_dnn(preprocessed_train_split_X, preprocessed_train_split_y, showPlot, features)

            if save_predictions:
                y_pred_test = trained_dnn.predict(preprocessed_test_split_X)
                prediction_test = pd.DataFrame({'real': preprocessed_test_split_y.flatten(), 'pred': y_pred_test.flatten()})
                prediction_test.to_csv(f"./results/rt_test_predictions-{features}-{key}.csv", sep=",")

            print("Saving dnn used for this fold")
            # trained_dnn.save(f"./results/dnn-{fold}-{key}-{features}.keras")

            print("Evaluation of the model & saving of the results")
            evaluate_model(trained_dnn, preprocessed_test_split_X, preprocessed_test_split_y, preproc_y, fold,
                           features, key)
