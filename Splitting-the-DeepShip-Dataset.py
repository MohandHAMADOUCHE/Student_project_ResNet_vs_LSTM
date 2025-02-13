import os
from scipy.io import wavfile

# Définition des chemins
base_path = "/tools/mohand_postdoc/datasets/DeepShip/DeepShip_organized_V2"
train_list = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train_list"
test_list  = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/test_list"

# Vérification de l'existence des fichiers
if not os.path.exists(train_list):
    print(f"❌ Fichier introuvable : {train_list}")
    exit(1)
if not os.path.exists(test_list):
    print(f"❌ Fichier introuvable : {test_list}")
    exit(1)
if not os.path.exists(base_path):
    print(f"❌ Dossier introuvable : {base_path}")
    exit(1)

# Lecture des fichiers train et test
train_lines = open(train_list).read().splitlines()
test_lines  = open(test_list).read().splitlines()

# Paramètres d'extraction audio (sans chevauchement)
audlen = 32000 * 3  # 3 secondes d'audio (si échantillonnage à 32 kHz)
audstr = audlen  # Pas de chevauchement (on avance de 3 sec à chaque segment)

# Traitement des fichiers d'entraînement
for idx_train, file_train in enumerate(train_lines):
    file_train_path = os.path.join(base_path, file_train)
    
    if not os.path.exists(file_train_path):
        print(f"⚠️ Fichier introuvable : {file_train_path}")
        continue

    # Chargement du fichier audio
    fs_train, aud_train = wavfile.read(file_train_path)

    # Définition du répertoire de sortie
    writedir_train = file_train_path.replace('/DeepShip/', '/DeepShip_train/')
    os.makedirs(writedir_train, exist_ok=True)  # Création automatique du répertoire

    # Découpage et enregistrement des segments audio (sans chevauchement)
    for st_train in range(0, len(aud_train) - audlen, audstr):
        output_file = f"{writedir_train}/%05d.wav" % (st_train // fs_train)
        wavfile.write(output_file, fs_train, aud_train[st_train:st_train + audlen])

    print(f"[TRAIN] {idx_train+1}/{len(train_lines)} : {file_train}")

# Traitement des fichiers de test
for idx_test, file_test in enumerate(test_lines):
    file_test_path = os.path.join(base_path, file_test)
    
    if not os.path.exists(file_test_path):
        print(f"⚠️ Fichier introuvable : {file_test_path}")
        continue

    # Chargement du fichier audio
    fs_test, aud_test = wavfile.read(file_test_path)

    # Définition du répertoire de sortie
    writedir_test = file_test_path.replace('/DeepShip/', '/DeepShip_test/')
    os.makedirs(writedir_test, exist_ok=True)  # Création automatique du répertoire

    # Découpage et enregistrement des segments audio (sans chevauchement)
    for st_test in range(0, len(aud_test) - audlen, audstr):
        output_file = f"{writedir_test}/%05d.wav" % (st_test // fs_test)
        wavfile.write(output_file, fs_test, aud_test[st_test:st_test + audlen])

    print(f"[TEST] {idx_test+1}/{len(test_lines)} : {file_test}")
