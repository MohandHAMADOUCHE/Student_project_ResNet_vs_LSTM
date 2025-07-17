import os
import librosa
import pandas as pd

# Chemin du dossier contenant les fichiers audio
base_path = "/tools/mohand_postdoc/datasets/DeepShip/DeepShip_organized_V2"

# Noms des fichiers sélectionnés dans chaque classe (avec .wav)
selected_files = {
    "Tug": ["9.wav", "40.wav", "49.wav"],
    "Cargo": ["15.wav", "38.wav", "62.wav"],
    "Tanker": ["10.wav", "18.wav", "24.wav"],
    "Passengership": ["9.wav", "16.wav", "29.wav"]
}

# Paramètres de segmentation
sampling_rate = 32000  # 16 kHz
window_size = 256  # 256 échantillons (16 ms)
hop_size = 128  # 128 échantillons (8 ms)

# Dictionnaire pour stocker les résultats
results = {}

# Traitement des fichiers
for ship_class, file_list in selected_files.items():
    class_path = os.path.join(base_path, ship_class)

    if not os.path.exists(class_path):
        print(f"❌ Dossier non trouvé : {class_path}")
        continue

    print(f"\n📂 Traitement des fichiers de {ship_class}...")

    total_duration = 0
    num_files = 0
    total_segments = 0  # Compteur des segments

    # Vérifier et traiter chaque fichier sélectionné
    for file in file_list:
        file_path = os.path.join(class_path, file)
        
        if os.path.exists(file_path):
            try:
                # Charger le fichier audio et obtenir sa durée
                duration = librosa.get_duration(path=file_path)
                total_duration += duration
                num_files += 1

                # Calcul du nombre de segments avec fenêtre de 256 et hop de 128
                num_segments = int(((duration * sampling_rate) - window_size) / hop_size) + 1
                total_segments += num_segments

                print(f"✔️ {file} - Durée : {round(duration, 2)} s - Segments : {num_segments}")

            except Exception as e:
                print(f"❌ Erreur lors de la lecture de {file}: {e}")
        else:
            print(f"⚠️ Fichier non trouvé : {file_path}")

    # Stockage des résultats
    results[ship_class] = {
        "Nombre de fichiers": num_files,
        "Durée totale (s)": round(total_duration, 2),
        "Nombre total de segments": total_segments
    }

# Création d'un DataFrame pour affichage
df_results = pd.DataFrame.from_dict(results, orient="index")

# Affichage des résultats
print("\n📊 Résultats finaux :")
print(df_results)

# Sauvegarde des résultats dans un fichier CSV
df_results.to_csv("resultats_segments_fenetres.csv", index=True)

print("\n✅ Les résultats des segments sélectionnés ont été enregistrés dans 'resultats_segments_fenetres.csv'.")
