import os
import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Fonction pour traiter un fichier audio
def process_single_audio(file_path, class_label, n_mfcc=20):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        n_fft = 512 if len(audio) < 2048 else 2048
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=20)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled, class_label
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")
        return None

# Fonction pour charger les fichiers audio et extraire les caractéristiques
def load_audio_files(data_path, metadata, n_mfcc=20):
    X = []
    y = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(metadata)):
            file_path = os.path.join(data_path, 'fold' + str(metadata.fold[i]), metadata.slice_file_name[i])
            class_label = metadata['classID'][i]
            futures.append(executor.submit(process_single_audio, file_path, class_label, n_mfcc))
        
        for future in futures:
            result = future.result()
            if result:
                mfcc_scaled, class_label = result
                X.append(mfcc_scaled)
                y.append(class_label)
    
    return np.array(X), np.array(y)

# Fonction principale de prétraitement des données
def preprocess_data(metadata_path, data_path):
    metadata = pd.read_csv(metadata_path)
    X, y = load_audio_files(data_path, metadata)
    return X, y
