
import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

# Chemin du dossier contenant les sous-dossiers des classes de navires
base_path = "/tools/mohand_postdoc/datasets/DeepShip/DeepShip_organized_V2"
output_path = "/tools/mohand_postdoc/datasets/DeepShip/DeepShip_SegmentsV1"


# Classes de navires
ship_classes = ["Cargo", "Passengership", "Tanker", "Tug"]

# Durée du segment en secondes
segment_duration = 3

# Dictionnaire pour stocker les informations sur les segments créés
segments_info = {}

# Parcours des classes de navires
for ship_class in ship_classes:
    class_path = os.path.join(base_path, ship_class)
    output_class_path = os.path.join(output_path, ship_class)
    os.makedirs(output_class_path, exist_ok=True)
    
    segments_count = 0
    
    # Vérifier si le dossier existe
    if not os.path.exists(class_path):
        print(f"Dossier non trouvé : {class_path}")
        continue
    
    # Parcourir les fichiers audio du dossier
    for file in os.listdir(class_path):
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(class_path, file)
            file_name_without_ext = os.path.splitext(file)[0]
            
            try:
                # Charger le fichier audio avec librosa
                y, sr = librosa.load(file_path, sr=None)
                
                # Calculer le nombre de segments complets de 3 secondes
                num_samples_per_segment = int(segment_duration * sr)
                num_segments = len(y) // num_samples_per_segment
                
                # Créer et sauvegarder chaque segment complet de 3 secondes
                for i in range(num_segments):
                    start = i * num_samples_per_segment
                    end = start + num_samples_per_segment
                    segment = y[start:end]
                    
                    # Vérifier que le segment fait exactement 3 secondes
                    if len(segment) == num_samples_per_segment:
                        # Générer un nom de fichier unique pour le segment
                        segment_filename = f"{file_name_without_ext}_segment_{segments_count:04d}.wav"
                        segment_path = os.path.join(output_class_path, segment_filename)
                        
                        # Sauvegarder le segment
                        sf.write(segment_path, segment, sr)
                        
                        segments_count += 1
                
            except Exception as e:
                print(f"Erreur lors du traitement de {file}: {e}")
    
    segments_info[ship_class] = segments_count

# Création d'un DataFrame pour affichage
df_results = pd.DataFrame.from_dict(segments_info, orient="index", columns=["Nombre de segments"])

# Affichage dans la console
print(df_results)

# Sauvegarde des résultats dans un fichier CSV
df_results.to_csv("resultats_mini_datasetsegmenté.csv", index=True)

print("\nLe mini-dataset a été créé dans le dossier 'mini_dataset'.")
print("Les résultats ont été enregistrés dans 'resultats_mini_dataset.csv'.")