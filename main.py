import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and informational logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
from utils import config, validate_path
from preprocessing import load_dataset, load_dataset_from_folders
from training import training_models
from analysis import (do_all_analysis)
from interface import build_interface

def retrain_models():
    try:
        training_models(X_train, y_train, X_test, y_test)
        do_all_analysis(X_test, y_test)
    except Exception as e:
         print("Error", str(e))

# Função para carregar os modelos pré-treinados
def use_pretrained_models():
    try:
        do_all_analysis(X_test, y_test)
    except Exception as e:
         print("Error", str(e))

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, class_mapping = load_dataset_from_folders(r"C:\Users\gumar\OneDrive\Área de Trabalho\Pesquisa UBO\DeepShip-main")
    #X_train, y_train, X_test, y_test = load_dataset()
    root = build_interface(
        config=config,
        retrain_models=retrain_models,
        use_pretrained_models=use_pretrained_models,
        validate_path=validate_path,
    )
    root.mainloop()