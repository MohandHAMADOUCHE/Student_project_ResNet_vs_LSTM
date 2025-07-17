import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from gammatone.gtgram import gtgram

def pre_emphasize(signal, alpha=0.97):
    """Applique un pré-accentuation pour renforcer les hautes fréquences."""
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def frame_signal(signal, frame_length=256, hop_length=128):
    """Découpe le signal en trames."""
    num_frames = (len(signal) - frame_length) // hop_length + 1
    frames = np.stack([signal[i * hop_length: i * hop_length + frame_length] for i in range(num_frames)])
    return frames

def compute_gfcc(audio, sr, n_filters=64, frame_length=256, hop_length=128, f_min=50):
    """Calcule les coefficients GFCC en utilisant un filtrage Gammatone."""

    # Vérifier la taille du signal d'entrée
    if len(audio) < frame_length:
        print(f"⚠ Avertissement: la taille du signal {len(audio)} est inférieure à frame_length={frame_length}")
        return np.zeros((n_filters, 1))  # Retourne un vecteur vide pour éviter l'erreur

    # Calcul de GFCC
    try:
        print(f"📏 Calcul GFCC : audio={len(audio)}, frame_length={frame_length}, hop_length={hop_length}")
        gammatone_features = gtgram(audio, sr, frame_length / sr, hop_length / sr, n_filters, f_min)
        gfcc = np.log(np.abs(gammatone_features) + 1e-8)  # Log compression
        return gfcc
    except ValueError as e:
        print(f"❌ Erreur dans gtgram: {e}")
        return np.zeros((n_filters, 1))  # Évite la rupture de programme

def extract_features_from_frames(frames, sr, n_mfcc=13):
    """Applique l'extraction des caractéristiques sur chaque trame."""
    mfcc_features = np.array([librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc) for frame in frames])
    gfcc_features = np.array([compute_gfcc(frame, sr) for frame in frames])
    cqt_features = np.array([np.abs(librosa.cqt(y=frame, sr=sr, n_bins=84)) for frame in frames])
    lofar_features = np.array([librosa.amplitude_to_db(np.abs(librosa.stft(frame, n_fft=256))[:n_mfcc, :], ref=np.max) for frame in frames])

    return mfcc_features, gfcc_features, cqt_features, lofar_features

if __name__ == "__main__":
    # Charger un fichier audio
    audio_file = "/tools/mohand_postdoc/datasets/DeepShip/DeepShip_organized_V2/Cargo/15.wav"  # Remplace par ton fichier
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Le fichier {audio_file} n'existe pas.")

    y, sr = librosa.load(audio_file, sr=None)

    # Étape 1 : Pré-accentuation
    emphasized_audio = pre_emphasize(y)

    # Étape 2 : Découpage en trames
    frames = frame_signal(emphasized_audio)

    # Affichage du nombre de frames générées
    num_frames = frames.shape[0]
    print(f"📌 Nombre de trames générées : {num_frames}")

    # Étape 3 : Sauvegarder les trames
    np.save("frames.npy", frames)
    print(f"✅ Trames sauvegardées avec une forme : {frames.shape}")

    # Étape 4 : Charger les trames et extraire les caractéristiques
    frames_loaded = np.load("frames.npy")
    mfcc, gfcc, cqt, lofar = extract_features_from_frames(frames_loaded, sr)

    # Afficher les formes des caractéristiques extraites
    print(f"🔹 MFCC Shape: {mfcc.shape}")
    print(f"🔹 GFCC Shape: {gfcc.shape}")
    print(f"🔹 CQT Shape: {cqt.shape}")
    print(f"🔹 LOFAR Shape: {lofar.shape}")

    # Sauvegarder les caractéristiques
    np.save("mfcc.npy", mfcc)
    np.save("gfcc.npy", gfcc)
    np.save("cqt.npy", cqt)
    np.save("lofar.npy", lofar)

    print("✅ Caractéristiques extraites et sauvegardées avec succès !")
