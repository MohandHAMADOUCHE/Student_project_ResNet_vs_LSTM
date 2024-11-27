import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and informational logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
from utils import config, model_paths, validate_path
from preprocessing import load_dataset
from training import load_and_compile_model, training_models
from analysis import (do_all_analysis)
from interface import build_interface

# Main function
def main(train=True):
    try:
        if train:
            training_models(X_train, y_train, X_test, y_test)

        print("Loading pretrained models...")
        resnet_model = load_and_compile_model(model_paths["resnet"])
        lstm_model = load_and_compile_model(model_paths["lstm"])
        transformer_model = load_and_compile_model(model_paths["transformer"])
        print("Pretrained models loaded and compiled successfully!")

        do_all_analysis(resnet_model, lstm_model, transformer_model, X_test, y_test)

    except Exception as e:
        print("Error", str(e))


def retrain_models():
    main(train=True)  # Passando o argumento 'train' como True para indicar que é re-treinamento

# Função para carregar os modelos pré-treinados
def use_pretrained_models():
    main(train=False)  # Passando o argumento 'train' como False para indicar que deve usar os modelos salvos


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset()
    root = build_interface(
        config=config,
        retrain_models=retrain_models,
        use_pretrained_models=use_pretrained_models,
        validate_path=validate_path,
    )
    root.mainloop()