
from audio_processing import extract_features
from audio_processing import visualize_features
import os

if __name__ == "__main__":
    # Remplacez par le chemin de votre fichier audio
    # file_path = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train_DeepShip_Segments_5000/Cargo/99_segment_3060.wav"
    file_path = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/test_DeepShip_Segments_3500/Tug/020446_segment_1086.wav"

    if not os.path.exists(file_path):
        print("Fichier audio introuvable.")
        exit()

    print("Extraction des caractéristiques...")

    # Extraction des caractéristiques avec visualisation activée
    mgcl_delta_features, features = extract_features(

        file_path=file_path,
        frame_length=2048,
        hop_length=512,
        n_features=12,
        show_example=False,  # Activer la visualisation des caractéristiques
        show_shapes=True  # Afficher les dimensions des caractéristiques
    )

    # Afficher les dimensions des caractéristiques fusionnées
    print(f"Dimensions des caractéristiques fusionnées (MGCL-Delta) : {mgcl_delta_features.shape}")
    print(type(mgcl_delta_features))
    print(type(features))
    

    # Afficher les caractéristiques fusionnées
    visualize_features(features)
