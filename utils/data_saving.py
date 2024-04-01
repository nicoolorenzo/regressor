import pickle


def save_preprocessor_and_dnn(preprocessor, dnn, fold):
    with open(f"./results/blender-preprocessor-{fold}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open(f"./results/blender-{fold}.pkl", "wb") as f:
        pickle.dump(dnn, f)

