# Configuração global
import os

from matplotlib import pyplot as plt


config = {
    "data_path": r"C:\Users\Gustavo\Desktop\UrbanSound8K\audio",
    "metadata_path": r"C:\Users\Gustavo\Desktop\UrbanSound8K\metadata\UrbanSound8K.csv",
    "epochs": 20,
    "batch_size": 32,
    "test_size": 0.2,
    "random_state": 42,
    "num_classes": 10,
}

model_paths = {
                "resnet": "resnet_model.keras",
                "lstm": "lstm_model.keras",
                "transformer": "transformer_model.keras",
            }

# Display all generated plots interactively
def display_images(image_list):
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


def validate_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path
