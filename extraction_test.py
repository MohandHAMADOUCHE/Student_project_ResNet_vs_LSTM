
from audio_processing import extract_features
from audio_processing import visualize_features
import os
import numpy as np



def Compare_results_diff_files():
    def count_differences(matrix1, matrix2):
        if matrix1.shape != matrix2.shape:
            raise ValueError("Les matrices doivent avoir les mêmes dimensions")
        return np.sum(matrix1 != matrix2)
    
    # Chemins des deux fichiers audio à comparer
    file_path1 = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/test_DeepShip_Segments_3500/Tug/020446_segment_1088.wav"
    file_path2 = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train_DeepShip_Segments_5000/Cargo/99_segment_3060.wav"

    # Vérification de l'existence des fichiers
    if not os.path.exists(file_path1) or not os.path.exists(file_path2):
        print("Un ou les deux fichiers audio sont introuvables.")
        exit()

    print("Extraction des caractéristiques pour le premier fichier...")
    mgcl_delta_features1, features1 = extract_features(
        file_path=file_path1,
        frame_length=2048,
        hop_length=512,
        n_features=12,
        show_example=False,
        show_shapes=True
    )

    print("\nExtraction des caractéristiques pour le deuxième fichier...")
    mgcl_delta_features2, features2 = extract_features(
        file_path=file_path2,
        frame_length=2048,
        hop_length=512,
        n_features=12,
        show_example=False,
        show_shapes=True
    )

    # Affichage des dimensions des caractéristiques fusionnées pour les deux fichiers
    print(f"\nDimensions des caractéristiques fusionnées (MGCL-Delta) du fichier 1 : {mgcl_delta_features1.shape}")
    print(f"Dimensions des caractéristiques fusionnées (MGCL-Delta) du fichier 2 : {mgcl_delta_features2.shape}")

    # Comparaison des matrices MGCL-Delta
    if mgcl_delta_features1.shape == mgcl_delta_features2.shape:
        diff_count = count_differences(mgcl_delta_features1, mgcl_delta_features2)
        print(f"\nNombre d'éléments différents entre les matrices MGCL-Delta : {diff_count}")
        print(f"Pourcentage de différence : {(diff_count / mgcl_delta_features1.size) * 100:.2f}%")
    else:
        print("\nLes matrices MGCL-Delta ont des dimensions différentes et ne peuvent pas être comparées directement.")

    # Visualisation des caractéristiques (optionnel)
    visualize_features(features1)
    visualize_features(features2)
if __name__ == "__main__":
    compare = False
    if compare:
        Compare_results_diff_files()
        exit()
    else:
        # Remplacez par le chemin de votre fichier audio
        # 
        file_path = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train_DeepShip_Segments_5000/Cargo/99_segment_3060.wav"
        #  file_path = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/test_DeepShip_Segments_3500/Tug/020446_segment_1088.wav"
        #  file_path = "5084.220822235505.flac"
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
