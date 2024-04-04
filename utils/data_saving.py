import pickle


def save_dnn(dnn, fold):
    with open(f"./results/blender-{fold}.pkl", "wb") as f:
        pickle.dump(dnn, f)

