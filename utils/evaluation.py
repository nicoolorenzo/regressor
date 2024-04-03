import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


def evaluate_model(dnn, preprocessed_train_split_X, test_split_y, fold):

    # This dictionary creates the evaluation results
    evaluation_dictionary = {
        'mae': mean_absolute_error(test_split_y, dnn.predict(preprocessed_train_split_X)),
        'medae': median_absolute_error(test_split_y, dnn.predict(preprocessed_train_split_X)),
        'mape': mean_absolute_percentage_error(test_split_y, dnn.predict(preprocessed_train_split_X)),
        'fold': fold
    }

    # Convert results dictionary to DataFrame, so it can be easily saved as a csv file
    results_df = pd.DataFrame([evaluation_dictionary])

    # Save the DataFrame to CSV
    if not os.path.exists("./results/evaluation_results.txt"):
        # If the file doesn't exist, set header=True
        results_df.to_csv("./results/evaluation_results.txt", index=False, header=True)
    else:
        # If the file exists, set header=False
        results_df.to_csv("./results/evaluation_results.txt", index=False, header=False, mode='a')

    print(f"Evaluation results have been saved into ./results/evaluation_results.txt")
