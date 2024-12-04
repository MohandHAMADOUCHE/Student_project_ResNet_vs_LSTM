import tkinter as tk
from ttkbootstrap import Style
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
from preprocessing import load_class_names, process_single_audio
from training import load_and_compile_model
from utils import model_paths, config


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

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def update_config(config, validate_path, root_style):
    """Janela para atualizar a configuração com layout em três colunas e barra de rolagem."""
    def save_changes():
        try:
            for key, var in vars_dict.items():
                value = var.get()
                if isinstance(config[key], int):
                    config[key] = int(value)
                elif isinstance(config[key], float):
                    config[key] = float(value)
                elif isinstance(config[key], str):
                    config[key] = validate_path(value)
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
    config_window.geometry("700x500")  # Ajuste para caber os parâmetros
    config_window.configure(bg=root_style.colors.primary)

    header = ttk.Label(config_window, text="Update Configuration", font=("Helvetica", 16, "bold"))
    header.pack(pady=10)

    # Frame com barra de rolagem
    canvas = tk.Canvas(config_window)
    scroll_y = tk.Scrollbar(config_window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scroll_y.set)

    # Frame interno que vai conter os parâmetros
    form_frame = ttk.Frame(canvas, padding=20)

    # Adicionando barra de rolagem
    scroll_y.pack(side="right", fill="y")
    canvas.pack(fill="both", expand=True)
    canvas.create_window((0, 0), window=form_frame, anchor="nw")
    form_frame.bind(
        "<Configure>", 
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    vars_dict = {}
    row = 0

    for i, (key, value) in enumerate(config.items()):
        if key in ["data_path", "metadata_path"]:
            # Para os caminhos, usar linha completa
            label = key.replace("_", " ").title()
            ttk.Label(form_frame, text=label, font=("Helvetica", 12)).grid(row=row, column=0, padx=10, pady=10, sticky="w", columnspan=3)
            vars_dict[key] = tk.StringVar(value=value)
            entry = ttk.Entry(form_frame, textvariable=vars_dict[key], width=60)
            entry.grid(row=row, column=1, padx=10, pady=10, columnspan=3)

            browse_btn = ttk.Button(form_frame, text="Browse", command=lambda var=vars_dict[key]: browse_file(var))
            browse_btn.grid(row=row, column=4, padx=5)
            row += 1
        else:
            # Para outros parâmetros, usar três colunas
            col = (i - 2) % 3  # Alternar entre 0, 1 e 2
            if col == 0 and i > 2:
                row += 1  # Nova linha a cada três colunas

            label = key.replace("_", " ").title()
            ttk.Label(form_frame, text=label, font=("Helvetica", 12)).grid(row=row, column=col * 2, padx=10, pady=10, sticky="w")
            vars_dict[key] = tk.StringVar(value=value)
            entry = ttk.Entry(form_frame, textvariable=vars_dict[key], width=20)
            entry.grid(row=row, column=col * 2 + 1, padx=10, pady=10)

    save_btn = ttk.Button(config_window, text="Save Changes", style="success.TButton", command=save_changes)
    save_btn.pack(fill="x", padx=20, pady=10)

    # Ajustando as colunas do frame
    for col in range(5):  # Considerando 5 colunas (1 para cada campo e 4 para botões)
        form_frame.grid_columnconfigure(col, weight=1, uniform="equal")

def select_models(callback):
    """
    Exibe uma janela para selecionar os modelos que o usuário deseja usar.
    Os modelos são carregados diretamente da configuração (model_paths).
    Salva a seleção dos modelos na variável de configuração.
    """
    def confirm_selection():
        selected_models = [var.get() for var in model_vars if var.get()]
        if not selected_models:
            messagebox.showwarning("No Selection", "Please select at least one model.")
        else:
            # Salvar a seleção de modelos na configuração
            config["selected_models"] = selected_models
            callback()
            model_selection_window.destroy()

    # Criar janela para seleção de modelos
    model_selection_window = tk.Toplevel()
    model_selection_window.title("Select Models")
    model_selection_window.geometry("500x300")  # Tamanho ajustado para ser proporcional ao conteúdo
    model_selection_window.configure(bg="#2E3B4E")  # Cor de fundo mais escura

    # Estilo do título
    header = ttk.Label(
        model_selection_window,
        text="Select Models to Use",
        font=("Helvetica", 16, "bold"),
        anchor="center",
        foreground="white",
        background="#0A74DA",  # Cor de fundo que combina com a interface principal
        padding=10
    )
    header.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")

    # Configurar layout para ocupar toda a largura
    model_selection_window.columnconfigure(0, weight=1)  # Coluna única ocupa toda a largura

    # Usando model_paths do utils.py para obter os modelos disponíveis
    model_vars = []
    for i, model_name in enumerate(model_paths.keys(), start=1):
        var = tk.StringVar(value=model_name)  # Definir todos como selecionados por padrão
        checkbox = ttk.Checkbutton(
            model_selection_window,
            text=model_name.capitalize(),
            variable=var,
            onvalue=model_name,
            offvalue="",
            style="primary.TCheckbutton"  # Estilo mais bonito e moderno
        )
        checkbox.grid(row=i, column=0, padx=20, pady=5, sticky="ew")  # Ocupa toda a largura
        model_vars.append(var)

    # Botão de confirmação
    confirm_btn = ttk.Button(
        model_selection_window,
        text="Confirm Selection",
        command=confirm_selection,
        style="large.TButton"
    )
    confirm_btn.grid(row=len(model_paths) + 1, column=0, pady=20, padx=20, sticky="ew")  # Botão ocupa toda a largura

    # Ajustar o estilo
    style = Style("darkly")
    style.configure("large.TButton", font=("Helvetica", 12), padding=10)
    style.configure("primary.TCheckbutton", font=("Helvetica", 12), padding=10, background="#0A74DA")

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
        ("📚 ------- [Retrain  Models] ------- 📚", lambda: select_models(retrain_models)),
        ("🤖 --- [Use Pretrained Models] --- 🤖", lambda: select_models(use_pretrained_models)),
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
