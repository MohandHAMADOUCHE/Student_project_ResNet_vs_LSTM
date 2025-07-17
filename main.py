import numpy as np  
import time
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from preprocessing import preprocess_data
from training import train_model
from inference import predict, evaluate_model_performance

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

# Fonction pour sauvegarder le modèle
def save_model(model, model_name='model', output_dir='saved_models'):
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, f'{model_name}.h5'))
    print(f"Modèle sauvegardé dans {os.path.join(output_dir, f'{model_name}.h5')}")

# Fonction pour charger le modèle sauvegardé
def load_saved_model(model_name='model', output_dir='saved_models'):
    model_path = os.path.join(output_dir, f'{model_name}.h5')
    if os.path.exists(model_path):
        print(f"Chargement du modèle sauvegardé depuis {model_path}...")
        return load_model(model_path)
    else:
        print(f"Aucun modèle sauvegardé trouvé dans {model_path}, réentraîner le modèle...")
        return None

def main():
    # Chemins des données
    data_path = r'E:\Post_doc\Gustavo\UrbanSound8K\UrbanSound8K\audio'
    metadata_path = r'E:\Post_doc\Gustavo\UrbanSound8K\UrbanSound8K\metadata\UrbanSound8K.csv'
    
    # Dossier de sortie pour les données prétraitées
    output_dir = 'preprocessed_data'

    # Vérification si les données sont déjà prétraitées
    X, y = load_preprocessed_data(output_dir)
    
    # Si les données n'ont pas été chargées, effectuer le prétraitement et sauvegarder
    if X is None or y is None:
        X, y = preprocess_data(metadata_path, data_path)
        save_preprocessed_data(X, y, output_dir)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Préparer les labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Redimensionner les données pour l'entrée
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Vérification si le modèle ResNet est déjà sauvegardé
    resnet_model = load_saved_model(model_name='resnet', output_dir='saved_models')
    if resnet_model is None:
        # Entraîner le modèle ResNet
        print("Entraînement de ResNet...")
        start_time = time.time()
        resnet_model, history_resnet = train_model(X_train, y_train, X_test, y_test, model_type='resnet')
        resnet_time = time.time() - start_time
        print(f"Temps d'entraînement de ResNet: {resnet_time:.2f} secondes")
        save_model(resnet_model, model_name='resnet', output_dir='saved_models')

    # Vérification si le modèle LSTM est déjà sauvegardé
    lstm_model = load_saved_model(model_name='lstm', output_dir='saved_models')
    if lstm_model is None:
        # Entraîner le modèle LSTM
        print("Entraînement de LSTM...")
        start_time = time.time()
        lstm_model, history_lstm = train_model(X_train, y_train, X_test, y_test, model_type='lstm')
        lstm_time = time.time() - start_time
        print(f"Temps d'entraînement de LSTM: {lstm_time:.2f} secondes")
        save_model(lstm_model, model_name='lstm', output_dir='saved_models')

    # Prédictions et évaluation
    y_pred_resnet = predict(resnet_model, X_test)
    y_pred_lstm = predict(lstm_model, X_test)

    # Évaluation des performances du modèle ResNet
    report_resnet, auc_resnet = evaluate_model_performance(np.argmax(y_test, axis=1), y_pred_resnet, class_names=['Class {}'.format(i) for i in range(10)])
    print("Rapport ResNet:", report_resnet)
    print("AUC ResNet:", auc_resnet)

    # Évaluation des performances du modèle LSTM
    report_lstm, auc_lstm = evaluate_model_performance(np.argmax(y_test, axis=1), y_pred_lstm, class_names=['Class {}'.format(i) for i in range(10)])
    print("Rapport LSTM:", report_lstm)
    print("AUC LSTM:", auc_lstm)

if __name__ == "__main__":
    main()
