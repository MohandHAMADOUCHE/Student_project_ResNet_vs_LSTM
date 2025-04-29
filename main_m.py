import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and informational logs
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from gammatone.gtgram import gtgram
from sklearn.metrics import confusion_matrix

from audio_processing import extract_features, visualize_features, visualize_features_ext


from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis

def reduce_and_visualize_with_nca(features, labels, threshold=0.3, chunk_size=5000):
    print("Filtering features based on threshold using NCA...")

    num_samples, num_features, num_frames = features.shape
    print(f"Input shape: {features.shape}")

    # Reshape features to (num_samples * num_frames, num_features)
    reshaped_features = features.reshape(-1, num_features)
    print(f"Reshaped shape: {reshaped_features.shape}")
    reshaped_labels = np.repeat(labels, num_frames)
    print(f"Reshaped labels shape: {reshaped_labels.shape}")

    # Initialize variables for storing results
    filtered_features = []
    nca = None

    # Process the data in chunks
    for i in range(0, reshaped_features.shape[0], chunk_size):
        print(f"Processing block {i // chunk_size + 1}...")
        start = i
        end = min(i + chunk_size, reshaped_features.shape[0])
        
        chunk_features = reshaped_features[start:end]
        chunk_labels = reshaped_labels[start:end]

        nca = NeighborhoodComponentsAnalysis(random_state=42)
        
        try:
            nca.fit(chunk_features, chunk_labels)
        except MemoryError:
            print("MemoryError: Reduce the dataset size or optimize memory usage.")
            return None, None

        # Debug: Visualize nca.components_
        plt.figure(figsize=(12, 8))
        plt.imshow(nca.components_, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f"NCA Components (Block {i // chunk_size + 1})")
        plt.xlabel("Feature Index")
        plt.ylabel("Component Index")
        plt.tight_layout()
        plt.show()

        # Extract feature weights
        feature_weights = np.linalg.norm(nca.components_, axis=0)

        # Debug: Visualize feature weights
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(feature_weights)), feature_weights, 'bo-')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        plt.title(f"Feature Weights (Block {i // chunk_size + 1})")
        plt.xlabel("Feature Index")
        plt.ylabel("Weight")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Screen features based on the threshold
        important_features_mask = feature_weights >= threshold
        filtered_chunk = chunk_features[:, important_features_mask]

        print(f"Block {i // chunk_size + 1}:")
        print(f" - Number of features before filtering: {num_features}")
        print(f" - Number of features after filtering: {filtered_chunk.shape[1]}")

        filtered_features.append(filtered_chunk)

    # Combine all filtered chunks into a single array
    final_filtered_features = np.vstack(filtered_features)

    # Reshape back to (num_samples, num_filtered_features, num_frames)
    num_filtered_features = final_filtered_features.shape[1]
    final_filtered_features = final_filtered_features.reshape(num_samples, num_frames, num_filtered_features).transpose(0, 2, 1)

    return final_filtered_features, nca


# Function to process a single audio file
def process_audio_file(args):
    """
    Processes a single audio file to extract features.

    Args:
        args (tuple): Contains the file path, class index, frame length, hop length, 
                      maximum frames, number of features, and save directory.

    Returns:
        tuple: A tuple containing the extracted features and class index.
    """
    file_path, class_index, frame_length, hop_length, n_features, save_dir, show_example, show_shapes = args  
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path) + '.npy')
   # print(f"n_features: {n_features}") ; exit
    mgcl_delta_features = extract_features(file_path=file_path, frame_length=frame_length, hop_length=hop_length, n_features=n_features, save_path=save_path, show_example=show_example, show_shapes=show_shapes)
   # print(file_path)
    #  print(f"features shape: {mgcl_delta_features.shape}")
    return mgcl_delta_features, class_index

# Function to load a dataset organized in subdirectories using ThreadPoolExecutor
def load_dataset(dataset_path, frame_length=2048, hop_length=512, n_features=12, save_dir=None, show_example=None, show_shapes=None):
    """
    Loads a dataset organized in subdirectories, processes the audio files in parallel, 
    and extracts their features.

    Args:
        dataset_path (str): Path to the root directory containing the dataset.
        frame_length (int): Length of each frame for feature extraction.
        hop_length (int): Hop length for feature extraction.
        n_features (int): Number of features to extract.
        save_dir (str, optional): Directory to save the extracted features.

    Returns:
        tuple: A tuple containing the features matrix (X), the labels (y), and 
               a dictionary mapping class names to indices.
    """
    X, y = [], []
    classes = sorted(os.listdir(dataset_path))  # List all class folders
    class_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}  # Map class names to indices

    tasks = []
    for cls_name in classes:
        cls_path = os.path.join(dataset_path, cls_name)
        if os.path.isdir(cls_path):  # Check if it is a directory
            for file_name in os.listdir(cls_path):  # List all files in the class folder
                file_path = os.path.join(cls_path, file_name)
                if file_name.endswith(('.wav', '.mp3')):  # Check if the file is an audio file
                    tasks.append((file_path, class_to_index[cls_name], frame_length, hop_length, n_features, save_dir, show_example, show_shapes))

    # Process the files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_audio_file, tasks), total=len(tasks), desc="Processing audio files"))
        for features, label in results:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y), class_to_index

from sklearn.metrics.pairwise import cosine_similarity


def main():
    version = "V_5000"
    dataset_path_train = fr'/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train_DeepShip_Segments_5000/'
    dataset_path_test = fr'/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/test_DeepShip_Segments_3500/'
    save_dir_train = fr'/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/trainDeepShip_{version}_processed_features'
    save_dir_test = fr'/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/testDeepShip_{version}_processed_features'
    model_save_path = fr"/tools/mohand_postdoc/datasets/models/resnet18_model_{version}_.h5"

    # Charger les datasets train et test     
    print("Chargement des données d'entraînement...")
    X_train_raw, y_train, class_to_index = load_dataset(dataset_path_train, save_dir=save_dir_train, show_example=False, show_shapes=False)
    print(f"shape of X_train_raw: {X_train_raw.shape}")
    print(f"shape of y_train: {y_train.shape}")
    print(f"class_to_index: {class_to_index}")
     
    print("Chargement des données de test...")
    X_test_raw, y_test, _ = load_dataset(dataset_path_test, save_dir=save_dir_test)
    print("X_train_raw[1].shape", X_train_raw[1].shape)
    visualize_features_ext(X_train_raw[1000] , 32000)
    # Réduction de dimensionnalité et visualisation (appliquée uniquement sur train)

    print("calcul de la matrice de similarité")
    X_flat = X_train_raw.reshape(20000, -1) 
    print("shape de X",X_flat.shape)
    similarity_matrix = cosine_similarity(X_flat)
    np.save('cosine_similarity_matrix.npy', similarity_matrix)


    print("Réduction de dimensionnalité sur les données d'entraînement...")
    # X_train_reduced, nca_model = reduce_and_visualize_with_nca(X_train_raw, y_train)
    
    # Appliquer la transformation NCA sur les données de test
    num_samples_test = X_test_raw.shape[0]
    num_features_test = np.prod(X_test_raw.shape[1:])
    flattened_X_test = X_test_raw.reshape(num_samples_test, num_features_test)
    X_test_reduced = nca_model.transform(flattened_X_test)
    X_test_reduced = X_test_reduced.reshape(num_samples_test, *X_train_reduced.shape[1:])


if __name__ == "__main__":
    main()

