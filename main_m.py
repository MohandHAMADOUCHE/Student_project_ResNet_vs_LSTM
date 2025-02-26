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

from audio_processing import extract_features, visualize_features, extract_features_v1


from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import numpy as np

def reduce_and_visualize_with_nca(features, labels, threshold=0.3, chunk_size=1000):
    """
    Filters features based on their importance weights using NCA and a threshold, with block processing.
    """
    print("Filtering features based on threshold using NCA...")

    num_samples, num_features, num_frames = features.shape
    print(f"Input shape: {features.shape}")

    # Step 1: Subsample frames to reduce temporal dimension
    subsampled_features = np.mean(features[:, :, ::4], axis=2)  # Reduce frames by averaging every 4 frames
    print(f"Subsampled features shape: {subsampled_features.shape}")

    # Step 2: Flatten the data for dimensionality reduction
    flattened_features = subsampled_features.reshape(subsampled_features.shape[0], -1)
    print(f"Flattened features shape: {flattened_features.shape}")

    # Step 3: Apply Incremental PCA to reduce dimensionality further
    ipca = IncrementalPCA(n_components=150)  # Reduce to a manageable number of dimensions
    reduced_features = ipca.fit_transform(flattened_features)
    print(f"Reduced features shape: {reduced_features.shape}")

    # Initialize variables for storing results
    filtered_chunks = []
    nca = None

    # Process the data in chunks with NCA
    for i in range(0, reduced_features.shape[0], chunk_size):
        print(f"Processing block {i // chunk_size + 1}...")
        start = i
        end = min(i + chunk_size, reduced_features.shape[0])
        
        chunk_features = reduced_features[start:end]
        chunk_labels = labels[start:end]

        nca = NeighborhoodComponentsAnalysis(random_state=42)
        try:
            nca.fit(chunk_features, chunk_labels)
        except MemoryError:
            print("MemoryError: Reduce the dataset size or optimize memory usage.")
            return None, None

        feature_weights = np.linalg.norm(nca.components_, axis=0)
        important_features_mask = feature_weights >= threshold
        filtered_chunk = chunk_features[:, important_features_mask]

        filtered_chunks.append(filtered_chunk)

    final_filtered_features = np.vstack(filtered_chunks)

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
    file_path, class_index, frame_length, hop_length, n_features, save_dir = args  
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path) + '.npy')
    
    # Extract features from the audio file def extract_features(file_path, frame_length=2048, hop_length=512, n_features=13, save_path=None, show_example=None, show_shapes=None):
    mgcl_delta_features = extract_features_v1(file_path=file_path, frame_length=frame_length, hop_length=hop_length, n_features=n_features, save_path=save_path, show_example=False, show_shapes=True)
    print(file_path)
    print(f"features shape: {mgcl_delta_features.shape}")
    return mgcl_delta_features, class_index

# Function to load a dataset organized in subdirectories using ThreadPoolExecutor
def load_dataset(dataset_path, frame_length=2048, hop_length=512, n_features=12, save_dir=None):
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
                    tasks.append((file_path, class_to_index[cls_name], frame_length, hop_length, n_features, save_dir))

    # Process the files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_audio_file, tasks), total=len(tasks), desc="Processing audio files"))
        for features, label in results:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y), class_to_index

def main():
    version = "V_5000"
    dataset_path_train = fr'/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train_DeepShip_Segments_5000/'
    dataset_path_test = fr'/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/test_DeepShip_Segments_3500/'
    save_dir_train = fr'/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/trainDeepShip_{version}_processed_features'
    save_dir_test = fr'/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/testDeepShip_{version}_processed_features'
    model_save_path = fr"/tools/mohand_postdoc/datasets/models/resnet18_model_{version}_.h5"

    # Charger les datasets train et test
    print("Chargement des données d'entraînement...")
    X_train_raw, y_train, class_to_index = load_dataset(dataset_path_train, save_dir=save_dir_train)
    print(f"shape of X_train_raw: {X_train_raw.shape}")
    print(f"shape of y_train: {y_train.shape}")
    print(f"class_to_index: {class_to_index}")
    
    print("Chargement des données de test...")
    X_test_raw, y_test, _ = load_dataset(dataset_path_test, save_dir=save_dir_test)
    print(X_train_raw[1].shape)
    visualize_features(X_train_raw[1] , 32000)
    # Réduction de dimensionnalité et visualisation (appliquée uniquement sur train)
    print("Réduction de dimensionnalité sur les données d'entraînement...")
    X_train_reduced, nca_model = reduce_and_visualize_with_nca(X_train_raw, y_train)
    
    # Appliquer la transformation NCA sur les données de test
    num_samples_test = X_test_raw.shape[0]
    num_features_test = np.prod(X_test_raw.shape[1:])
    flattened_X_test = X_test_raw.reshape(num_samples_test, num_features_test)
    X_test_reduced = nca_model.transform(flattened_X_test)
    X_test_reduced = X_test_reduced.reshape(num_samples_test, *X_train_reduced.shape[1:])


if __name__ == "__main__":
    main()

