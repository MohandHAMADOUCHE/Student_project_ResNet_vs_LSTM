import os
import pygame
import logging
import tkinter as tk
from training import train
from ttkbootstrap import Style
from analysis import do_all_analysis
from tkinter import ttk, filedialog, messagebox
from preprocessing import preprocessing, load_preprocessed, classify_audio

def update(config, root_style):
    """
    Creates a window for updating the configuration, organized into tabs.

    Parameters:
        config (dict): The current configuration dictionary to update.
        root_style (ThemedStyle): The styling object for the interface.
    """

    def save_changes():
        """
        Saves the changes made in the configuration window.

        Updates the `config` dictionary based on user inputs and closes the window
        if successful. Displays an error message if input validation fails.
        """
        try:
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
            logging.info("Configuration updated successfully.")
            config_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            logging.error(f"Failed to update configuration: {e}")

    def browse_folder(var):
        """
        Opens a dialog for selecting a folder and updates the associated variable.

        Parameters:
            var (tk.StringVar): The variable to update with the selected folder path.
        """
        folder_path = filedialog.askdirectory(title="Select a Folder")
        if folder_path:
            var.set(folder_path)
            logging.info(f"Folder selected: {folder_path}")

    # Create a new window for the configuration panel
    config_window = tk.Toplevel()
    config_window.title("Configuration Panel")
    config_window.geometry("650x610")
    config_window.configure(bg=root_style.colors.primary)
    logging.info("Configuration panel initialized.")

    # Header for the window
    header = ttk.Label(
        config_window,
        text="Update Configuration",
        font=("Helvetica", 16, "bold")
    )
    header.pack(pady=10)

    # Notebook widget to organize settings into tabs
    notebook = ttk.Notebook(config_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    vars_dict = {}  # Dictionary to hold variables for each configuration setting

    # Loop through configuration sections to create tabs
    for i, (section, settings) in enumerate(config.items()):
        if isinstance(settings, dict):
            # Create a frame for each section
            section_frame = ttk.Frame(notebook, padding=10)
            notebook.add(section_frame, text=section.replace("_", " ").title())
            vars_dict[section] = {}

            row = 0

            # Special layout for the first section (e.g., file paths)
            if i == 0:
                # Frame for path fields
                path_frame = ttk.Frame(section_frame)
                path_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
                row += 1

                for key in ["data_path", "processed_data_path"]:
                    if key in settings:
                        label = ttk.Label(path_frame, text=key.replace("_", " ").title(), font=("Helvetica", 12))
                        label.grid(row=row, column=0, sticky="w", padx=5, pady=3)

                        vars_dict[section][key] = tk.StringVar(value=str(settings[key]))
                        entry = ttk.Entry(path_frame, textvariable=vars_dict[section][key], width=50)
                        entry.grid(row=row, column=1, padx=5, pady=3, sticky="w")

                        browse_btn = ttk.Button(
                            path_frame,
                            text="Browse",
                            command=lambda var=vars_dict[section][key]: browse_folder(var)
                        )
                        browse_btn.grid(row=row, column=2, padx=5, pady=3, sticky="w")

                        row += 1

                # Visual separator
                ttk.Separator(section_frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
                row += 1

                # Frame for other fields in the section
                data_frame = ttk.Frame(section_frame)
                data_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
                col = 0

                for key, value in settings.items():
                    if key in ["data_path", "processed_data_path"]:
                        continue  # Skip already displayed fields

                    label = ttk.Label(data_frame, text=key.replace("_", " ").title(), font=("Helvetica", 12))
                    label.grid(row=row, column=col * 2, sticky="w", padx=5, pady=3)

                    if isinstance(value, bool):
                        vars_dict[section][key] = tk.StringVar(value="1" if value else "0")
                        checkbox = ttk.Checkbutton(data_frame, variable=vars_dict[section][key])
                        checkbox.grid(row=row, column=(col * 2) + 1, sticky="w", padx=5, pady=3)
                    else:
                        vars_dict[section][key] = tk.StringVar(value=str(value))
                        entry = ttk.Entry(data_frame, textvariable=vars_dict[section][key], width=25)
                        entry.grid(row=row, column=(col * 2) + 1, padx=5, pady=3)

                    col += 1
                    if col >= 2:  # Alternate between columns
                        col = 0
                        row += 1

            # Standard layout for other sections
            else:
                for key, value in settings.items():
                    label = ttk.Label(section_frame, text=key.replace("_", " ").title(), font=("Helvetica", 12))
                    label.grid(row=row, column=0, sticky="w", padx=5, pady=3)

                    if isinstance(value, bool):
                        vars_dict[section][key] = tk.StringVar(value="1" if value else "0")
                        checkbox = ttk.Checkbutton(section_frame, variable=vars_dict[section][key])
                        checkbox.grid(row=row, column=1, sticky="w", padx=5, pady=3)
                    else:
                        vars_dict[section][key] = tk.StringVar(value=str(value))
                        entry = ttk.Entry(section_frame, textvariable=vars_dict[section][key], width=25)
                        entry.grid(row=row, column=1, padx=5, pady=3)

                    row += 1

    # Save button
    save_btn = ttk.Button(
        config_window,
        text="Save Changes",
        style="success.TButton",
        command=save_changes
    )
    save_btn.pack(fill="x", padx=20, pady=10)
    logging.info("Configuration panel ready for user interaction.")

def retrain(config, image_list):
    """
    Orchestrates the retraining process by preparing data, training the model,
    and performing analysis on the results.

    Parameters:
        config (dict): Configuration settings for preprocessing and training.
        image_list (list): List of images to be used for training and analysis.

    Returns:
        None
    """
    try:
        # Reset the image list
        image_list = []
        logging.info("Image list reset for retraining.")

        # Preprocessing step: Split data into training and testing sets
        X_train, y_train, X_test, y_test, class_to_index = preprocessing(config, image_list)
        logging.info("Data preprocessing completed. Training and testing sets prepared.")

        # Model training step
        train(config, X_train, y_train, X_test, y_test, class_to_index)
        logging.info("Model training completed successfully.")

        # Perform all analysis on the test set
        do_all_analysis(X_test, y_test, config, class_to_index, image_list)
        logging.info("Analysis completed. Results and insights generated.")
    except Exception as e:
        logging.error(f"An error occurred during retraining: {e}")
        raise

def utilize_pretrained_model(config, image_list):
    """
    Utilizes a pretrained model to perform analysis on preprocessed data.

    Parameters:
        config (dict): Configuration settings for the analysis process.
        image_list (list): List of images to store visualization outputs.

    Returns:
        None
    """
    try:
        # Reset the image list for a clean start
        image_list = []
        logging.info("Image list reset for utilizing pretrained model.")

        # Load preprocessed data for testing
        X_test, y_test, class_to_index = load_preprocessed(config, image_list)
        logging.info("Preprocessed data loaded successfully for testing.")

        # Perform all analyses using the pretrained model
        do_all_analysis(X_test, y_test, config, class_to_index, image_list)
        logging.info("Analysis using the pretrained model completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred while utilizing the pretrained model: {e}")
        raise

def select_and_classify_audio(config):
    """
    Allows the user to select an audio file, play it, and classify it using selected models.

    Parameters:
        config (dict): Configuration settings, including model paths and general configurations.

    Returns:
        None
    """
    try:
        # Select an audio file through a file dialog
        logging.info("Prompting the user to select an audio file.")
        audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if not audio_file:
            logging.warning("No audio file selected by the user.")
            return

        logging.info(f"Audio file selected: {audio_file}")

        # Initialize and play the selected audio file
        logging.info("Initializing pygame mixer to play the selected audio file.")
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        logging.info("Audio playback started.")

        # Retrieve class mapping from the data directory
        logging.info("Retrieving class mappings from the data directory.")
        data_path = config["general"]["data_path"]
        class_mapping = {
            class_name: idx
            for idx, class_name in enumerate(sorted(os.listdir(data_path)))
            if os.path.isdir(os.path.join(data_path, class_name))
        }
        logging.info(f"Detected classes: {class_mapping}")

        # Load paths for selected models
        logging.info("Loading paths for selected models.")
        model_paths = {
            model_name: f"{model_name}_model.keras"
            for model_name, is_model_selected in config["selected_models"].items()
            if is_model_selected
        }
        logging.info(f"Selected model paths: {model_paths}")

        # Perform audio classification
        logging.info("Classifying the selected audio file using the models.")
        results = classify_audio(
            file_path=audio_file,
            config=config,
            model_paths=model_paths,
            class_mapping=class_mapping
        )

        # Display classification results
        model_results = results["model_results"]
        final_prediction = results["final_prediction"]
        logging.info(f"Final prediction: {final_prediction}")

        details = "\n".join([
            f"Model: {model}, Predicted Class: {res['predicted_class']}"
            for model, res in model_results.items()
        ])

        messagebox.showinfo(
            "Classification Result",
            f"Final Prediction (Mode): {final_prediction}\n\nModel Details:\n{details}"
        )

        # Stop audio playback
        pygame.mixer.music.stop()
        logging.info("Audio playback stopped.")
    
    except Exception as e:
        # Handle any errors and ensure the audio playback stops
        pygame.mixer.music.stop()
        logging.error(f"An error occurred: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")

def build_interface(config, image_list):
    """
    Builds and initializes the main user interface for the audio classification tool.

    Parameters:
        config (dict): Configuration dictionary containing application parameters.
        image_list (list): List to store images generated during processing.

    Returns:
        root (Tk): The main Tkinter application window.
    """

    # Initialize the main window
    root = tk.Tk()
    root.title("Audio Classification Control Panel")
    root.geometry("500x500")  
    root.configure(bg="#f0f0f0")  
    
    # Apply a themed style for a modern look
    style = Style("darkly")
    style.configure("large.TButton", font=("Helvetica", 14), padding=10)

    # Main title
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

    # Configure grid layout for consistent spacing
    root.rowconfigure([1, 2, 3, 4], weight=1) 
    root.columnconfigure(0, weight=1)         

    image_list = []

    # Buttons with actions
    buttons = [
        ("⚙️ --- [ Update Configuration] --- ⚙️", lambda: update(config, style)),
        ("📚 ------- [Retrain  Models] ------- 📚", lambda: retrain(config, image_list) ),
        ("🤖 --- [Use Pretrained Models] --- 🤖", lambda: utilize_pretrained_model(config, image_list) ),
        ("🎵 -------- [Classify  Audio] -------- 🎵", lambda: select_and_classify_audio(config)),
    ]

    # Create and place buttons
    for i, (text, command) in enumerate(buttons, start=1):
        btn = ttk.Button(
            root,
            text=text,
            style="large.TButton", 
            command=command,
        )
        btn.grid(row=i, column=0, sticky="nsew", padx=10, pady=10) 

    # Footer with credits
    footer = ttk.Label(
        root,
        text="Developed in Université de Bretagne Occidentale © 2024",
        font=("Helvetica", 10),
        anchor="center",
        foreground=style.colors.dark,
    )
    footer.grid(row=5, column=0, sticky="nsew", pady=(20, 10)) 

    return root