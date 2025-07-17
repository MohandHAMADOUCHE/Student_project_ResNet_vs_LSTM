import os
import librosa
import pandas as pd

# Chemin du dossier contenant les sous-dossiers des classes de navires
base_path = "/tools/mohand_postdoc/datasets/DeepShip/DeepShip_organized_V2"

# Classes de navires
ship_classes = ["Cargo", "Passengership", "Tanker", "Tug"]

# Dictionnaire pour stocker les durées et le nombre de segments
results = {}

# Fenêtres de segmentation en secondes
segment_sizes = [2.5, 3]

# Parcours des classes de navires
for ship_class in ship_classes:
    class_path = os.path.join(base_path, ship_class)
    total_duration = 0  # Durée totale en secondes
    num_files = 0       # Nombre de fichiers audio dans la classe
    
    # Vérifier si le dossier existe
    if not os.path.exists(class_path):
        print(f"Dossier non trouvé : {class_path}")
        continue
    
    # Parcourir les fichiers audio du dossier
    for file in os.listdir(class_path):
        if file.endswith(".wav") or file.endswith(".mp3"):  # Ajuste selon tes formats
            file_path = os.path.join(class_path, file)
            
            try:
                # Charger le fichier audio avec librosa pour obtenir la durée
                duration = librosa.get_duration(path=file_path)
                total_duration += duration
                num_files += 1
            except Exception as e:
                print(f"Erreur lors de la lecture de {file}: {e}")

    # Calcul du nombre de segments pour chaque fenêtre
    num_segments_25 = int(total_duration // segment_sizes[0])
    num_segments_30 = int(total_duration // segment_sizes[1])
    
    # Stockage des résultats
    results[ship_class] = {
        "Nombre de fichiers": num_files,
        "Durée totale (s)": round(total_duration, 2),
        f"Segments ({segment_sizes[0]}s)": num_segments_25,
        f"Segments ({segment_sizes[1]}s)": num_segments_30
    }
# Création d'un DataFrame pour affichage
df_results = pd.DataFrame.from_dict(results, orient="index")

# Affichage dans la console
print(df_results)

# Sauvegarde des résultats dans un fichier CSV
df_results.to_csv("resultats_segments.csv", index=True)

print("\nLes résultats ont été enregistrés dans 'resultats_segments.csv'.")
