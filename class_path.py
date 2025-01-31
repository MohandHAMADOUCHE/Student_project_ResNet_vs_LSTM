import os
import shutil
import pandas as pd

# Chemins des fichiers et répertoires
csv_path = "/tools/mohand_postdoc/datasets/urban/UrbanSound8K/metadata/UrbanSound8K.csv"
audio_root_path = "/tools/mohand_postdoc/datasets/urban/UrbanSound8K/audio"
destination_root_path = "/tools/mohand_postdoc/datasets/urban/UrbanSound8K/classified_audio"  # Dossier de sortie

# Charger le fichier CSV
df = pd.read_csv(csv_path)

# Vérifier et créer le dossier racine de destination
if not os.path.exists(destination_root_path):
    os.makedirs(destination_root_path)

# Traiter chaque ligne du fichier CSV
for _, row in df.iterrows():
    file_name = row["slice_file_name"]
    fold = row["fold"]
    class_id = row["classID"]  # Identifiant de la classe sonore

    # Chemin du fichier source (dans son dossier "foldX" d'origine)
    original_path = os.path.join(audio_root_path, f"fold{fold}", file_name)
    
    # Dossier de destination basé sur la classe sonore
    class_folder = os.path.join(destination_root_path, f"class_{class_id}")
    destination_path = os.path.join(class_folder, file_name)

    # Créer le dossier de la classe s'il n'existe pas
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # Vérifier si le fichier existe avant de le déplacer
    if os.path.exists(original_path):
        shutil.move(original_path, destination_path)
        print(f"✔️ Mové: {original_path} -> {destination_path}")
    else:
        print(f"❌ Fichier introuvable: {original_path}")

print("✅ Organisation par classe sonore terminée !")
