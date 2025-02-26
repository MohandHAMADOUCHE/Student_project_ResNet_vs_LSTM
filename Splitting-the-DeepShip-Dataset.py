import os
import shutil

# Définition des chemins
base_path = "/tools/mohand_postdoc/datasets/DeepShip/DeepShip_organized_V2"
train_list = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train.txt"
test_list = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/test.txt"
train_output_dir = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train"
test_output_dir = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/test"

def copy_files(list_path, output_dir):
    if not os.path.exists(list_path):
        print(f"⚠️ Fichier liste introuvable : {list_path}")
        return

    with open(list_path, 'r') as f:
        files = f.read().splitlines()

    for file_path in files:
        source_path = os.path.join(base_path, file_path)
        destination_path = os.path.join(output_dir, file_path)
        
        # Création du répertoire de destination (y compris les sous-dossiers)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
            print(f"✅ Copié : {source_path} -> {destination_path}")
        else:
            print(f"⚠️ Fichier introuvable : {source_path}")

    print(f"✅ Copie des fichiers vers {output_dir} terminée.")

# Copie des fichiers d'entraînement
copy_files(train_list, train_output_dir)

# Copie des fichiers de test
copy_files(test_list, test_output_dir)
