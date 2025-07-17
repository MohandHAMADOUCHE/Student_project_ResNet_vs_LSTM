import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from preprocessing import load_audio_files

# Fonction pour prédire avec le modèle
def predict(model, X_test):
    if model is None:
        print("Le modèle n'est pas chargé.")
        return None
    # Renvoyer les probabilités des classes (et non les indices)
    y_pred_prob = model.predict(X_test)
    return y_pred_prob

# Fonction pour évaluer les performances du modèle
def evaluate_model_performance(y_true, y_pred_prob, class_names):
    if y_pred_prob is None:
        print("Erreur : Aucune prédiction n'a été effectuée.")
        return None, None
    report = classification_report(y_true, np.argmax(y_pred_prob, axis=1), target_names=class_names, output_dict=True)
    try:
        # Calculer l'AUC avec les probabilités (pour multi-classe)
        auc_score = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovr')
    except ValueError:
        auc_score = None
    return report, auc_score

# Fonction pour sauvegarder les données prétraitées
def save_preprocessed_data(X, y, output_dir='preprocessed_data'):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    print(f"Caractéristiques sauvegardées dans {output_dir}")

# Fonction pour charger les données prétraitées
def load_preprocessed_data(output_dir='preprocessed_data'):
    if os.path.exists(os.path.join(output_dir, 'X.npy')) and os.path.exists(os.path.join(output_dir, 'y.npy')):
        print("Données prétraitées trouvées, chargement...")
        X = np.load(os.path.join(output_dir, 'X.npy'))
        y = np.load(os.path.join(output_dir, 'y.npy'))
        return X, y
    else:
        print("Aucune donnée prétraitée trouvée, traitement en cours...")
        return None, None

# Charger les données et préparer les tests
def load_data_for_inference(metadata_path, data_path, test_size=0.2, output_dir='preprocessed_data'):
    # Vérification si les données prétraitées existent
    X, y = load_preprocessed_data(output_dir)
    
    if X is None or y is None:
        metadata = pd.read_csv(metadata_path)
        # Charger les fichiers audio et extraire les caractéristiques
        X, y = load_audio_files(data_path, metadata)  # Utilisez la fonction de prétraitement existante
        save_preprocessed_data(X, y, output_dir)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Préparer les labels pour l'inférence
    y_test = to_categorical(y_test, num_classes=10)

    # Redimensionner les données pour l'entrée
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"Données de test préparées, taille de X_test : {X_test.shape}, taille de y_test : {y_test.shape}")
    return X_test, y_test

def main():
    # Chemins des données
    data_path = r'E:\Post_doc\Gustavo\UrbanSound8K\UrbanSound8K\audio'
    metadata_path = r'E:\Post_doc\Gustavo\UrbanSound8K\UrbanSound8K\metadata\UrbanSound8K.csv'

    # Charger les données de test
    X_test, y_test = load_data_for_inference(metadata_path, data_path)

    # Charger les modèles déjà entraînés
    print("Chargement des modèles...")
    resnet_model = load_model(r'E:\Post_doc\Gustavo\UrbanSound8K\Mohand\saved_models\resnet.h5')
    lstm_model = load_model(r'E:\Post_doc\Gustavo\UrbanSound8K\Mohand\saved_models\lstm.h5')  # Assurez-vous que le modèle LSTM est sauvegardé avec ce nom

    # Vérification si les modèles ont été chargés correctement
    if resnet_model is None:
        print("Erreur lors du chargement du modèle ResNet.")
        return
    if lstm_model is None:
        print("Erreur lors du chargement du modèle LSTM.")
        return

    # Prédictions avec le modèle ResNet
    print("Prédictions avec ResNet...")
    y_pred_resnet_prob = predict(resnet_model, X_test)
    if y_pred_resnet_prob is None:
        print("Erreur lors des prédictions avec ResNet.")
        return

    # Prédictions avec le modèle LSTM
    print("Prédictions avec LSTM...")
    y_pred_lstm_prob = predict(lstm_model, X_test)
    if y_pred_lstm_prob is None:
        print("Erreur lors des prédictions avec LSTM.")
        return

    # Évaluation des performances pour ResNet
    report_resnet, auc_resnet = evaluate_model_performance(np.argmax(y_test, axis=1), y_pred_resnet_prob, class_names=['Class {}'.format(i) for i in range(10)])
    if report_resnet is not None:
        print("Rapport ResNet:")
        print(report_resnet)
        print("AUC ResNet:", auc_resnet)

    # Évaluation des performances pour LSTM
    report_lstm, auc_lstm = evaluate_model_performance(np.argmax(y_test, axis=1), y_pred_lstm_prob, class_names=['Class {}'.format(i) for i in range(10)])
    if report_lstm is not None:
        print("Rapport LSTM:")
        print(report_lstm)
        print("AUC LSTM:", auc_lstm)

if __name__ == "__main__":
    main()
