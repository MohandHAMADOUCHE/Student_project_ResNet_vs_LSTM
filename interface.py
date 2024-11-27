import tkinter as tk
from ttkbootstrap import Style
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
from preprocessing import load_class_names, process_single_audio
from training import load_and_compile_model
from utils import model_paths


def load_audio_and_classify(class_names_file="classes.json"):
    """Carrega e classifica um arquivo de áudio selecionado pelo usuário."""
    try:
        # Abrir diálogo para selecionar um arquivo de áudio
        audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if not audio_file:
            print("No file selected.")
            return

        print(f"Selected audio file: {audio_file}")

        # Processar o arquivo de áudio
        audio_data, _ = process_single_audio(audio_file, None)  # O class_label não é necessário aqui
        if audio_data is None:
            raise ValueError("Error processing the audio file.")

        audio_data = audio_data.reshape(1, audio_data.shape[0], 1)  # Ajustar formato para o modelo

        # Reproduzir o áudio usando pygame
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Esperar o término da reprodução
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Carregar modelos
        resnet_model = load_and_compile_model(model_paths["resnet"])
        lstm_model = load_and_compile_model(model_paths["lstm"])
        transformer_model = load_and_compile_model(model_paths["transformer"])

        # Fazer previsões
        resnet_pred_id = resnet_model.predict(audio_data).argmax(axis=1)[0]
        lstm_pred_id = lstm_model.predict(audio_data).argmax(axis=1)[0]
        transformer_pred_id = transformer_model.predict(audio_data).argmax(axis=1)[0]

        # Carregar nomes das classes
        class_names = load_class_names(class_names_file)

        # Mapear IDs para nomes das classes
        resnet_pred = class_names.get(resnet_pred_id, "Unknown")
        lstm_pred = class_names.get(lstm_pred_id, "Unknown")
        transformer_pred = class_names.get(transformer_pred_id, "Unknown")

        # Exibir os resultados
        messagebox.showinfo(
            "Classification Results",
            f"ResNet Prediction: {resnet_pred}\n"
            f"LSTM Prediction: {lstm_pred}\n"
            f"Transformer Prediction: {transformer_pred}"
        )

    except Exception as e:
        messagebox.showerror("Error", f"Error classifying the audio file: {e}")

def update_config(config, validate_path, root_style):
    """Janela para atualizar a configuração."""
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

    config_window = tk.Toplevel()
    config_window.title("Configuration Panel")
    config_window.geometry("600x450")
    config_window.configure(bg=root_style.colors.primary)

    header = ttk.Label(config_window, text="Update Configuration", font=("Helvetica", 16, "bold"))
    header.pack(pady=10)

    form_frame = ttk.Frame(config_window, padding=20)
    form_frame.pack(fill="both", expand=True)

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
        vars_dict[key] = tk.StringVar(value=config.get(key, ""))
        entry = ttk.Entry(form_frame, textvariable=vars_dict[key], width=40)
        entry.grid(row=i, column=1, padx=10, pady=10)

        if is_path:
            browse_btn = ttk.Button(form_frame, text="Browse", command=lambda var=vars_dict[key]: browse_file(var))
            browse_btn.grid(row=i, column=2, padx=5)

    save_btn = ttk.Button(config_window, text="Save Changes", style="success.TButton", command=save_changes)
    save_btn.pack(fill="x", padx=20, pady=10)



def build_interface(config, retrain_models, use_pretrained_models, validate_path):
    print("Loading interface...")

    """Constrói e inicializa a interface principal com design aprimorado."""
    # Inicializar a janela principal
    root = tk.Tk()
    root.title("Audio Classification Control Panel")
    root.geometry("500x500")  # Tamanho maior para visualização melhor
    root.configure(bg="#f0f0f0")  # Cor de fundo clara para contraste

    # Inicializar o estilo
    style = Style("darkly")  # Escolha um tema ('darkly', 'flatly', etc.)
    style.configure("large.TButton", font=("Helvetica", 14), padding=10)

    # Título principal
    title = ttk.Label(
        root,
        text="Audio Classification Tool UBO & USP",
        font=("Helvetica", 18, "bold"),
        anchor="center",
        background=style.colors.primary,
        foreground=style.colors.light,
        padding=10,
    )
    title.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(10, 20))  # Ocupa toda a largura

    # Configurar layout do grid
    root.rowconfigure([1, 2, 3, 4], weight=1)  # Cada botão ocupa a mesma altura
    root.columnconfigure(0, weight=1)         # Única coluna que ocupa toda a largura

    # Botões estilizados
    buttons = [
        ("⚙️ --- [ Update Configuration] --- ⚙️", lambda: update_config(config, validate_path, style)),
        ("📚 ------- [Retrain  Models] ------- 📚", retrain_models),
        ("🤖 --- [Use Pretrained Models] --- 🤖", use_pretrained_models),
        ("🎵 -------- [Classify  Audio] -------- 🎵", lambda: load_audio_and_classify()),
    ]

    for i, (text, command) in enumerate(buttons, start=1):
        btn = ttk.Button(
            root,
            text=text,
            style="large.TButton",  # Usando o estilo personalizado
            command=command,
        )
        btn.grid(row=i, column=0, sticky="nsew", padx=10, pady=10)  # Expandir em largura e altura

    # Rodapé
    footer = ttk.Label(
        root,
        text="Developed in Université de Bretagne Occidentale © 2024",
        font=("Helvetica", 10),
        anchor="center",
        foreground=style.colors.dark,
    )
    footer.grid(row=5, column=0, sticky="nsew", pady=(20, 10))  # Rodapé ocupa toda a largura

    print("Interface loaded...")

    return root
