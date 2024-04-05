import pickle


def save_dnn(trained_dnn, fold, features):
    with open(f"./results/blender-{fold}-{features}.pkl", "wb") as f:
        #TODO: quitar pickle y salvar con keras
        pickle.dump(trained_dnn, f)

