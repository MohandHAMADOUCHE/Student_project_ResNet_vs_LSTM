
import os
import random

# Chemin du dossier contenant les sous-dossiers des classes de navires
base_path = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train_DeepShip_Segments_5000"

# Classes de navires
ship_classes = ["Cargo", "Passengership", "Tanker", "Tug"]

# Nombre de segments à conserver par classe
target_segments = 5000

for ship_class in ship_classes:
    class_path = os.path.join(base_path, ship_class)
    
    # Liste tous les fichiers de segments dans le dossier de la classe
    segments = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    
    # Si le nombre de segments est supérieur à la cible
    if len(segments) > target_segments:
        # Sélectionne aléatoirement les segments à supprimer
        segments_to_remove = random.sample(segments, len(segments) - target_segments)
        
        # Supprime les segments en trop
        for segment in segments_to_remove:
            os.remove(os.path.join(class_path, segment))
        
        print(f"{ship_class}: {len(segments_to_remove)} segments supprimés. Maintenant à {target_segments} segments.")
    else:
        print(f"{ship_class}: Déjà à {len(segments)} segments, pas de suppression nécessaire.")

print("Dataset équilibré avec au maximum 8800 segments par classe")
