import tkinter as tk
from ttkbootstrap import Style
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
from preprocessing import preprocessing, load_preprocessed, process_audio_file
from training import train
from analysis import do_all_analysis


def load_audio_and_classify(config, class_names_file="classes.json"):
    """Carrega e classifica um arquivo de áudio selecionado pelo usuário."""
    try:
        # Abrir diálogo para selecionar um arquivo de áudio
        audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if not audio_file:
            print("No file selected.")
            return

        print(f"Selected audio file: {audio_file}")

        # Processar o arquivo de áudio
        audio_data, _ = process_audio_file(audio_file, None)  # O class_label não é necessário aqui
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



        for model_name, is_selected in config["selected_models"].items():
            if not is_selected:
                continue
            model = load_and_compile_model(f"{model_name}_model.keras")
            model_pred_id = model.predict(audio_data).argmax(axis=1)[0]
        
        # Carregar nomes das classes
        class_names = "" #load_class_names(class_names_file)

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

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def update_config(config, root_style):
    """Janela para atualizar a configuração com layout em abas."""

    def save_changes():
        """Salva alterações na configuração."""
        try:
            # Atualiza valores na configuração
            for section, sub_dict in vars_dict.items():
                for key, var in sub_dict.items():
                    if isinstance(config[section][key], bool):
                        config[section][key] = var.get() == "1"
                    elif isinstance(config[section][key], int):
                        config[section][key] = int(var.get())
                    elif isinstance(config[section][key], float):
                        config[section][key] = float(var.get())
                    else:
                        config[section][key] = var.get()
            messagebox.showinfo("Success", "Configuration updated successfully!")
            config_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def browse_folder(var):
        """Abre dialog para seleção de pasta."""
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if folder_path:
            var.set(folder_path)

    config_window = tk.Toplevel()
    config_window.title("Configuration Panel")
    config_window.geometry("800x600")
    config_window.configure(bg=root_style.colors.primary)

    header = ttk.Label(config_window, text="Update Configuration", font=("Helvetica", 16, "bold"))
    header.pack(pady=10)

    # Criar notebook para abas
    notebook = ttk.Notebook(config_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    vars_dict = {}

    # Criar uma aba para cada seção de configuração
    for section, settings in config.items():
        if isinstance(settings, dict):
            section_frame = ttk.Frame(notebook, padding=10)
            notebook.add(section_frame, text=section.replace("_", " ").title())
            vars_dict[section] = {}

            row = 0
            for key, value in settings.items():
                label = ttk.Label(section_frame, text=key.replace("_", " ").title(), font=("Helvetica", 12))
                label.grid(row=row, column=0, sticky="w", padx=10, pady=5)

                # Checkbutton para booleanos
                if isinstance(value, bool):
                    vars_dict[section][key] = tk.StringVar(value="1" if value else "0")
                    checkbox = ttk.Checkbutton(section_frame, variable=vars_dict[section][key])
                    checkbox.grid(row=row, column=1, sticky="w", padx=10, pady=5)

                # Entrada padrão para outros tipos de dados
                else:
                    vars_dict[section][key] = tk.StringVar(value=str(value))
                    entry = ttk.Entry(section_frame, textvariable=vars_dict[section][key], width=40)
                    entry.grid(row=row, column=1, padx=10, pady=5)

                    # Botão de navegação para seleção de pasta
                    if key in ["data_path", "save_dir"]:
                        browse_btn = ttk.Button(section_frame, text="Browse",
                                                command=lambda var=vars_dict[section][key]: browse_folder(var))
                        browse_btn.grid(row=row, column=2, padx=5)

                row += 1

    save_btn = ttk.Button(config_window, text="Save Changes", style="success.TButton", command=save_changes)
    save_btn.pack(fill="x", padx=20, pady=10)

def retrain(config, image_list):
    X_train, y_train, X_test, y_test, class_to_index = preprocessing(config, image_list)
    train(config, X_train, y_train, X_test, y_test, class_to_index)
    do_all_analysis(X_test, y_test, config, class_to_index, image_list)

def utilize_pretrained_model(config, image_list):
    X_test, y_test, class_to_index = load_preprocessed(config, image_list)
    do_all_analysis(X_test, y_test, config, class_to_index, image_list)

def build_interface(config, image_list):
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

    image_list = []

    # Botões estilizados
    buttons = [
        ("⚙️ --- [ Update Configuration] --- ⚙️", lambda: update_config(config, style)),
        ("📚 ------- [Retrain  Models] ------- 📚", lambda: retrain(config, image_list) ),
        ("🤖 --- [Use Pretrained Models] --- 🤖", lambda: utilize_pretrained_model(config, image_list) ),
        ("🎵 -------- [Classify  Audio] -------- 🎵", lambda: load_audio_and_classify(config)),
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
