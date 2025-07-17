import os
import librosa
import soundfile as sf
import shutil

# Chemins
base_path = "/tools/mohand_postdoc/datasets/DeepShip/DeepShip_organized_V2"
mini_dataset_path = "/tools/mohand_postdoc/datasets/DeepShip/mini_dataset"
raw_data_path = os.path.join(mini_dataset_path, "raw_data")
segment_data_path = os.path.join(mini_dataset_path, "segment_data")

# Noms des fichiers sélectionnés par classe
selected_files = {
    "Tug": ["9.wav", "40.wav", "49.wav"],
    "Cargo": ["15.wav", "38.wav", "62.wav"],
    "Tanker": ["10.wav", "18.wav", "24.wav"],
    "Passengership": ["9.wav", "16.wav", "29.wav"]
}

# Paramètres de segmentation
sampling_rate = 32000  # 32 kHz
window_size = 256  # 256 échantillons (16 ms)
hop_size = 128  # 128 échantillons (8 ms)

# Dictionnaire pour stocker le nombre total de segments par classe
segment_counts = {}

# Étape 1: Création des dossiers nécessaires
for ship_class in selected_files.keys():
    raw_class_path = os.path.join(raw_data_path, ship_class)
    segment_class_path = os.path.join(segment_data_path, ship_class)

    os.makedirs(raw_class_path, exist_ok=True)
    os.makedirs(segment_class_path, exist_ok=True)

# Étape 2: Copier les fichiers originaux dans `raw_data`
for ship_class, file_list in selected_files.items():
    class_source_path = os.path.join(base_path, ship_class)
    class_dest_path = os.path.join(raw_data_path, ship_class)

    for file in file_list:
        src_file_path = os.path.join(class_source_path, file)
        dest_file_path = os.path.join(class_dest_path, file)

        if os.path.exists(src_file_path):
            shutil.copy(src_file_path, dest_file_path)
            print(f"✔️ Copié : {file} → {class_dest_path}")
        else:
            print(f"⚠️ Fichier introuvable : {src_file_path}")

# Étape 3: Découper chaque fichier en segments et enregistrer dans `segment_data`
for ship_class, file_list in selected_files.items():
    raw_class_path = os.path.join(raw_data_path, ship_class)
    segment_class_path = os.path.join(segment_data_path, ship_class)

    total_segments_class = 0  # Compteur de segments pour cette classe

    for file in file_list:
        file_path = os.path.join(raw_class_path, file)
        
        if os.path.exists(file_path):
            try:
                # Charger le fichier audio
                audio, sr = librosa.load(file_path, sr=sampling_rate)

                # Segmenter l'audio
                num_samples = len(audio)
                segment_index = 0
                filename_base = os.path.splitext(file)[0]  # Retirer l'extension .wav

                for start in range(0, num_samples - window_size, hop_size):
                    end = start + window_size
                    segment = audio[start:end]

                    # Sauvegarder le segment avec le nom du fichier d'origine
                    segment_filename = f"{filename_base}_segment_{segment_index:04d}.wav"
                    segment_path = os.path.join(segment_class_path, segment_filename)

                    # Vérifier si le segment existe déjà
                    if os.path.exists(segment_path):
                        print(f"⚠️ Segment déjà existant, ignoré : {segment_filename}")
                    else:
                        sf.write(segment_path, segment, samplerate=sampling_rate)
                        total_segments_class += 1

                    segment_index += 1

                print(f"✔️ {file} → {total_segments_class} nouveaux segments créés dans {segment_class_path}")
            
            except Exception as e:
                print(f"❌ Erreur avec {file}: {e}")
        else:
            print(f"⚠️ Fichier introuvable : {file_path}")

    segment_counts[ship_class] = total_segments_class  # Enregistrer le nombre total de segments créés par classe

# Affichage récapitulatif
print("\n📊 Nombre total de nouveaux segments créés par classe :")
for ship_class, count in segment_counts.items():
    print(f"  - {ship_class} : {count} segments")

print("\n✅ Mini-dataset créé avec les fichiers bruts dans 'raw_data' et les segments dans 'segment_data'.")
