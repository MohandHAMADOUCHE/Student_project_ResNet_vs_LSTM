import os

# Dictionnaire des noms de classes basé sur UrbanSound8K
class_labels = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

# Dossier contenant les dossiers class_X
classified_audio_path = "/tools/mohand_postdoc/datasets/urban/UrbanSound8K/classified_audio"

# Vérifier si le dossier existe
if not os.path.exists(classified_audio_path):
    print("❌ Le dossier des classes n'existe pas. Exécute d'abord le script de tri.")
else:
    # Parcourir tous les dossiers class_X
    for class_id, real_name in class_labels.items():
        old_folder = os.path.join(classified_audio_path, f"class_{class_id}")
        new_folder = os.path.join(classified_audio_path, real_name)

        # Renommer si l'ancien dossier existe
        if os.path.exists(old_folder):
            os.rename(old_folder, new_folder)
            print(f"✅ Renommé : {old_folder} → {new_folder}")
        else:
            print(f"⚠️ Dossier introuvable : {old_folder}")

    print("🎉 Renommage terminé !")


