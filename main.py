import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and informational logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
from interface import build_interface

# ======================
# Main Workflow
# ======================

def main():

    config = {
        "general_config": {
            "data_path": r"D:\deepship\Original",
            "processed_data_path": r"D:\deepship\processed_features",
            "epochs": 100,
            "audio_duration": 20,
            "batch_size": 32,
            "test_size": 0.2,
            "random_state": 42,
            "learning_rate": 0.01,
            "dense_units": 128,
            "dropout_rate": 0.3,
            "patience": 100,
            "mix_classes": False,
            "visual_feedback": True,
        },
        "lstm_config":{
            "lstm_units": 64,
            "lstm_dropout": 0.2,
        },
        "resnet_config":{
            "resnet_filters": 32,
            "resnet_kernel_size": 3,
            "resnet_blocks": 3,
        },
        "transformer_config":{
            "transformer_heads": 4,
            "transformer_ff_dim": 128,
            "transformer_layers": 2,
            "transformer_epsilon": 1e-6,
        },
        "resnet18_config": {
            "initial_filters": 64,
            "blocks_per_stage": 2,
            "momentum": 0.9
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

    image_list = []

    root = build_interface(config, image_list)
    root.mainloop()

if __name__ == "__main__":
    main()
