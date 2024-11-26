import os

# Set TensorFlow environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and informational logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

import time
import json
import pygame
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import History # type: ignore
from preprocessing import load_audio_files, plot_spectrograms_per_class_from_files, process_single_audio
from training import build_resnet_model, build_lstm_model, build_transformer_model
from analysis import (
    evaluate_model_performance,
    plot_metrics_comparison,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_comparison,
    plot_image_models,
    image_list
)

# Default paths and hyperparameters
config = {
    "data_path": r'C:\Users\gumar\OneDrive\Área de Trabalho\Pesquisa UBO\Revisão Literaria\UrbanSound8K\UrbanSound8K\audio',
    "metadata_path": r'C:\Users\gumar\OneDrive\Área de Trabalho\Pesquisa UBO\Revisão Literaria\UrbanSound8K\UrbanSound8K\metadata\UrbanSound8K.csv',
    "epochs": 20,
    "batch_size": 32,
    "test_size": 0.2,
    "random_state": 42,
    "num_classes": 10
}

model_paths = {
                "resnet": "resnet_model.keras",
                "lstm": "lstm_model.keras",
                "transformer": "transformer_model.keras",
            }

# Helper function to validate paths
def validate_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path

from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
from PIL import Image, ImageTk  # Para ícones

# Função para atualizar configuração
def update_config():
    def save_changes():
        try:
            config["data_path"] = validate_path(vars_dict["data_path"].get())
            config["metadata_path"] = validate_path(vars_dict["metadata_path"].get())
            config["epochs"] = int(vars_dict["epochs"].get())
            config["batch_size"] = int(vars_dict["batch_size"].get())
            config["test_size"] = float(vars_dict["test_size"].get())
            config["random_state"] = int(vars_dict["random_state"].get())
            messagebox.showinfo("Success", "Configuration updated successfully!")
            config_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def browse_file(var):
        filepath = filedialog.askopenfilename(title="Select a File")
        if filepath:
            var.set(filepath)

    # Criar janela de configuração
    config_window = tk.Toplevel(root)
    config_window.title("Configuration Panel")
    config_window.geometry("600x450")
    config_window.configure(bg=style.colors.primary)

    # Header estilizado
    header = ttk.Label(config_window, text="Update Configuration", font=("Helvetica", 16, "bold"), anchor="center")
    header.pack(pady=10)

    # Frame para organizar os campos
    form_frame = ttk.Frame(config_window, padding=20)
    form_frame.pack(fill="both", expand=True)

    # Dados de configuração inicial
    fields = [
        ("Data Path", "data_path", True),
        ("Metadata Path", "metadata_path", True),
        ("Epochs", "epochs", False),
        ("Batch Size", "batch_size", False),
        ("Test Size", "test_size", False),
        ("Random State", "random_state", False),
    ]

    vars_dict = {}
    for i, (label, key, is_path) in enumerate(fields):
        ttk.Label(form_frame, text=label, font=("Helvetica", 12)).grid(row=i, column=0, padx=10, pady=10, sticky="w")
        vars_dict[key] = tk.StringVar(value=config.get(key, ""))  # Use config ou um dicionário fictício para testes
        entry = ttk.Entry(form_frame, textvariable=vars_dict[key], width=40)
        entry.grid(row=i, column=1, padx=10, pady=10)
        
        if is_path:  # Botão de navegação para campos de caminho
            browse_btn = ttk.Button(form_frame, text="Browse", command=lambda var=vars_dict[key]: browse_file(var))
            browse_btn.grid(row=i, column=2, padx=5)

    # Botão de salvar
    save_btn = ttk.Button(config_window, text="Save Changes", style="success.TButton", command=save_changes)
    save_btn.pack(fill="x", padx=20, pady=10)  # Botão ocupa 100% da largura

def load_class_names(file_path="classes.json"):
    """
    Carrega as classes a partir de um arquivo JSON.
    """
    try:
        with open(file_path, 'r') as file:
            class_names = json.load(file)
        return {int(k): v for k, v in class_names.items()}  # Converte as chaves para inteiros
    except Exception as e:
        raise ValueError(f"Error loading class names: {e}")

def load_audio_and_classify():
    """
    Permite carregar um arquivo de áudio e classificá-lo usando os modelos treinados.
    """
    try:
        # Abrir diálogo para selecionar um arquivo de áudio
        audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if not audio_file:
            print("No file selected.")
            return

        print(f"Selected audio file: {audio_file}")

        # Processar o arquivo de áudio usando a função process_single_audio
        audio_data, _ = process_single_audio(audio_file, None)  # O class_label não é necessário aqui
        if audio_data is None:
            raise ValueError("Error processing the audio file.")

        audio_data = audio_data.reshape(1, audio_data.shape[0], 1)  # Ajustar formato para o modelo

        # Tocar o áudio usando pygame
        pygame.mixer.init()  # Inicializa o mixer do pygame
        pygame.mixer.music.load(audio_file)  # Carrega o arquivo de áudio
        pygame.mixer.music.play()  # Toca o áudio

        # Aguardar o término da reprodução do áudio (opcional)
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Espera até que o áudio termine de tocar

        resnet_model = load_and_compile_model(model_paths["resnet"])
        lstm_model = load_and_compile_model(model_paths["lstm"])
        transformer_model = load_and_compile_model(model_paths["transformer"])

        # Fazer previsões com os modelos carregados
        resnet_pred_id = resnet_model.predict(audio_data).argmax(axis=1)[0]
        lstm_pred_id = lstm_model.predict(audio_data).argmax(axis=1)[0]
        transformer_pred_id = transformer_model.predict(audio_data).argmax(axis=1)[0]

        class_names = load_class_names("classes.json")

        # Mapear IDs para nomes das classes
        resnet_pred = class_names.get(resnet_pred_id, "Unknown")
        lstm_pred = class_names.get(lstm_pred_id, "Unknown")
        transformer_pred = class_names.get(transformer_pred_id, "Unknown")

        # Exibir os resultados
        print(f"ResNet Prediction: {resnet_pred}")
        print(f"LSTM Prediction: {lstm_pred}")
        print(f"Transformer Prediction: {transformer_pred}")

       

        # Exibir resultados em uma MessageBox
        messagebox.showinfo(
            "Classification Results",
            f"ResNet Prediction: {resnet_pred}\n"
            f"LSTM Prediction: {lstm_pred}\n"
            f"Transformer Prediction: {transformer_pred}"
        )

    except Exception as e:
        messagebox.showerror("Error", f"Error classifying the audio file: {e}")

def retrain_models():
    main(train=True)  # Passando o argumento 'train' como True para indicar que é re-treinamento

# Função para carregar os modelos pré-treinados
def use_pretrained_models():
    main(train=False)  # Passando o argumento 'train' como False para indicar que deve usar os modelos salvos

# Função para salvar o modelo
def save_model(model, file_path):
    model.save(file_path)
    print(f"Model saved to {file_path}")

# Após carregar o modelo, recompilá-lo
def load_and_compile_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def dict_to_history(history_dict):
    if not history_dict:
        return None
    history = History()
    history.history = history_dict
    return history

def save_training_times(training_times, history_resnet=None, history_lstm=None, history_transformer=None, training_params=None, filename="training_times.json"):
    """
    Salva os tempos de treinamento, o histórico dos modelos e os parâmetros de treinamento no arquivo JSON.
    """
    try:
        data_to_save = {
            "training_params": training_params,
            "training_times": training_times,
            "history_resnet": history_resnet.history if history_resnet else None,  # Salva o histórico como dicionário
            "history_lstm": history_lstm.history if history_lstm else None,
            "history_transformer": history_transformer.history if history_transformer else None
        }

        with open(filename, 'w') as file:
            json.dump(data_to_save, file, indent=4)
        print(f"Training times, history, and params saved to {filename}")
    except Exception as e:
        print(f"Error saving training times: {e}")

def load_training_data(filename="training_times.json"):
    """
    Carrega os tempos de treinamento, histórico e parâmetros de um arquivo JSON.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            training_times = data.get("training_times", {})
            history_resnet = dict_to_history(data.get("history_resnet", {}))
            history_lstm = dict_to_history(data.get("history_lstm", {}))
            history_transformer = dict_to_history(data.get("history_transformer", {}))
            training_params = data.get("training_params", {})
        print("Training data loaded successfully!")
        return training_times, history_resnet, history_lstm, history_transformer, training_params
    except Exception as e:
        print(f"Error loading training data: {e}")
        return {}, None, None, None, {}

# Main function
def main(train=True):
    try:
        # Validate paths
        data_path = validate_path(config["data_path"])
        metadata_path = validate_path(config["metadata_path"])

        # Load metadata
        metadata = pd.read_csv(metadata_path)

        # Load and process audio files
        print("Loading and processing audio files...")
        X, y = load_audio_files(data_path, metadata)

        # Plot spectrograms
        print("Plotting spectrograms...")
        plot_spectrograms_per_class_from_files(data_path, metadata, [f"Class {i}" for i in range(config["num_classes"])])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])

        # Prepare labels
        y_train = to_categorical(y_train, num_classes=config["num_classes"])
        y_test = to_categorical(y_test, num_classes=config["num_classes"])

        # Reshape data for model input
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Armazenar tempos de treinamento
        training_times = {}
        if not train:
            history_resnet, history_lstm, history_transformer = None, None, None


        # Parâmetros de treinamento
        training_params = {
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "test_size": config["test_size"],
            "random_state": config["random_state"],
            "num_classes": config["num_classes"]
        }

        if train:
            # Treinamento dos modelos, salvando os tempos
            print("Training ResNet...")
            start_time_resnet = time.time()
            resnet_model = build_resnet_model((X_train.shape[1], 1), config["num_classes"])
            resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history_resnet = resnet_model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(X_test, y_test))
            training_times["resnet"] = time.time() - start_time_resnet

            # Train and evaluate LSTM
            print("Training LSTM...")
            start_time_lstm = time.time()
            lstm_model = build_lstm_model((X_train.shape[1], 1), config["num_classes"])
            lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history_lstm = lstm_model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(X_test, y_test))
            training_times["lstm"] = time.time() - start_time_lstm

            # Train and evaluate Transformer
            print("Training Transformer...")
            start_time_transformer = time.time()
            transformer_model = build_transformer_model((X_train.shape[1], 1), config["num_classes"])
            transformer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history_transformer = transformer_model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(X_test, y_test))
            training_times["transformer"] = time.time() - start_time_transformer

            # Salvar os modelos
            # Salvar os modelos usando o formato .keras
            save_model(resnet_model, "resnet_model.keras")
            save_model(lstm_model, "lstm_model.keras")
            save_model(transformer_model, "transformer_model.keras")

            print("Models trained and saved successfully!")

            # Salvar os tempos de treinamento em um arquivo JSON
            save_training_times(training_times, history_resnet, history_lstm, history_transformer, training_params)
        else:
            # Carregar os modelos previamente salvos
            print("Loading pretrained models...")
            resnet_model = load_and_compile_model(model_paths["resnet"])
            lstm_model = load_and_compile_model(model_paths["lstm"])
            transformer_model = load_and_compile_model(model_paths["transformer"])
            print("Pretrained models loaded and compiled successfully!")

            training_times, history_resnet, history_lstm, history_transformer, training_params = load_training_data()


        start_classification_resnet = time.time()
        y_pred_resnet = resnet_model.predict(X_test).argmax(axis=1)
        classification_time_resnet = time.time() - start_classification_resnet

        start_classification_lstm = time.time()
        y_pred_lstm = lstm_model.predict(X_test).argmax(axis=1)
        classification_time_lstm = time.time() - start_classification_lstm

        start_classification_transformer = time.time()
        y_pred_transformer = transformer_model.predict(X_test).argmax(axis=1)
        classification_time_transformer = time.time() - start_classification_transformer

        # Evaluate models
        print("Evaluating models...")

        # Exemplo de uso
        plot_image_models("models.png")

        y_true = y_test.argmax(axis=1)
        report_resnet, auc_resnet = evaluate_model_performance(y_true, y_pred_resnet, [f"Class {i}" for i in range(config["num_classes"])])
        report_lstm, auc_lstm = evaluate_model_performance(y_true, y_pred_lstm, [f"Class {i}" for i in range(config["num_classes"])])
        report_transformer, auc_transformer = evaluate_model_performance(y_true, y_pred_transformer, [f"Class {i}" for i in range(config["num_classes"])])

        plot_metrics_comparison(
            report_resnet, 
            report_lstm, 
            report_transformer, 
            auc_resnet, 
            auc_lstm, 
            auc_transformer, 
            [f"Class {i}" for i in range(config["num_classes"])]
        )

        # Plot confusion matrices
        print("Plotting confusion matrices...")
        plot_confusion_matrix(y_true, y_pred_resnet, [f"Class {i}" for i in range(config["num_classes"])], "ResNet")
        plot_confusion_matrix(y_true, y_pred_lstm, [f"Class {i}" for i in range(config["num_classes"])], "LSTM")
        plot_confusion_matrix(y_true, y_pred_transformer, [f"Class {i}" for i in range(config["num_classes"])], "Transformer")
        
        # Plot ROC curves
        print("Plotting ROC curves...")
        plot_multiclass_roc(y_true, y_pred_resnet, [f"Class {i}" for i in range(config["num_classes"])], "ResNet")
        plot_multiclass_roc(y_true, y_pred_lstm, [f"Class {i}" for i in range(config["num_classes"])], "LSTM")
        plot_multiclass_roc(y_true, y_pred_transformer, [f"Class {i}" for i in range(config["num_classes"])], "Transformer")

        # Plot accuracy and time comparison
        print("Generating accuracy and time comparison plot...")
        plot_comparison(
            training_times.get("resnet", 0),  # Usando o tempo do dicionário
            training_times.get("lstm", 0),    # Usando o tempo do dicionário
            training_times.get("transformer", 0),  # Usando o tempo do dicionário  # Adicione o tempo de treinamento do Transformer
            classification_time_resnet, 
            classification_time_lstm, 
            classification_time_transformer,  # Adicione o tempo de classificação do Transformer
            history_resnet = history_resnet, 
            history_lstm = history_lstm, 
            history_transformer = history_transformer # Adicione a precisão do Transformer
        )
        # Display all generated plots interactively
        def display_images():
            if not image_list:
                print("No images were generated.")
                return

            current_image_index = 0

            def update_image():
                plt.clf()
                plt.imshow(image_list[current_image_index])
                plt.axis('off')
                plt.title(f"Image {current_image_index + 1} of {len(image_list)}")
                plt.draw()

            def on_key(event):
                nonlocal current_image_index
                if event.key == 'right':
                    current_image_index = (current_image_index + 1) % len(image_list)
                elif event.key == 'left':
                    current_image_index = (current_image_index - 1) % len(image_list)
                update_image()

            fig, ax = plt.subplots()
            fig.canvas.mpl_connect('key_press_event', on_key)
            update_image()
            plt.show()

        print("Displaying generated plots...")
        display_images()

    except Exception as e:
        messagebox.showerror("Error", str(e))

print("Loading interface...")

# Inicializar Tkinter root
root = tk.Tk()
root.title("Audio Classification Control Panel")
root.geometry("400x600")  # Altura maior para testes
root.configure(bg="#e0f7fa")

# Inicializar o estilo ttkbootstrap
style = Style("darkly")  # Escolha um tema: 'cosmo', 'darkly', 'flatly', etc.

# Criar um estilo personalizado para os botões com fonte maior
style.configure("large.TButton", font=("Helvetica", 16))


# Inicializar o estilo ttkbootstrap uma vez

buttons = [
    ("⚙️ --- [ Update Configuration] --- ⚙️", update_config),
    ("📚 ------ [Retrain  Models] ------ 📚", retrain_models),
    ("🤖 --- [Use Pretrained Models] --- 🤖", use_pretrained_models),
    ("🎵 ------ [Classify  Audio] ------ 🎵", load_audio_and_classify),
]

# Configurar grid layout para expandir botões
root.rowconfigure([0, 1, 2, 3], weight=1)  # Cada botão ocupa uma linha com altura proporcional
root.columnconfigure(0, weight=1)         # Coluna única, ocupando toda a largura

# Criar botões estilizados que ocupam 100% da largura e altura com letras maiores
for i, (text, command) in enumerate(buttons):
    btn = ttk.Button(
        root,
        text=text,
        style="large.TButton",  # Usando o estilo personalizado
        command=command,
    )
    btn.grid(row=i, column=0, sticky="nsew", padx=10, pady=10)  # Expandir em todas as direções

# Start the GUI
root.mainloop()