# =======================
# Environment Configuration
# =======================

import os
# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import logging
from interface import build_interface
from colorama import Fore, Style

# =======================
# Logging Configuration
# =======================

def configure_logging():
    """
    Configures the logging format for terminal output.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL}: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )

# =======================
# Main Workflow
# =======================

def main():
    """
    Entry point of the application. Configures the model parameters and launches the interface.
    """
    # Initialize logging
    configure_logging()

    logging.info("Starting the application...")

    # Configuration dictionary for model, preprocessing, and training parameters
    config = {
        "general": {
            "data_path": r"C:\Users\Gustavo\Desktop\UrbanSound8K\organized",
            "processed_data_path": r"C:\Users\Gustavo\Desktop\UrbanSound8K\processed_features",
            "epochs": 100,
            "visual_feedback": True,
            "audio_duration": 20,
            "mix_classes": False,
            "batch_size": 32,
            "add_noisy": True,
            "test_size": 0.2,
            "reduction_NCA": False,
            "random_state": 42,
            "learning_rate": 0.01,
            "dense_units": 128,
            "dropout_rate": 0.3,
            "patience": 100,
            "frame_length": 2048,
            "hop_length": 512,
            "max_frames": 100,
            "n_features": 13,
            "sample_rate": 22050,
        },
        "lstm": {
            "lstm_units": 64,
            "lstm_dropout": 0.2,
        },
        "resnet": {
            "resnet_filters": 32,
            "resnet_kernel_size": 3,
            "resnet_blocks": 3,
        },
        "transformer": {
            "transformer_heads": 4,
            "transformer_ff_dim": 128,
            "transformer_layers": 2,
            "transformer_epsilon": 1e-6,
        },
        "resnet18": {
            "initial_filters": 64,
            "blocks_per_stage": 2,
            "momentum": 0.9,
        },
        "preprocessing_methods": {
            "MFCC": True,
            "GFCC": True,
            "CQT": True,
            "LOFAR": True,
            "DELTAS": True,
        },
        "selected_models": {
            "resnet": True,
            "lstm": True,
            "transformer": True,
            "resnet18": True,
        },
    }

    logging.info("Configuration loaded successfully.")

    # Placeholder for storing images generated during processing
    image_list = []

    logging.info("Launching the graphical user interface...")
    root = build_interface(config, image_list)

    logging.info("Interface is now running.")
    root.mainloop()

    logging.info("Application has been closed. Goodbye!")

if __name__ == "__main__":
    main()

