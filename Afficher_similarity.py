import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger la matrice de similarité
similarity_matrix = np.load('cosine_similarity_matrix.npy')
print("Shape de la matrice :", similarity_matrix.shape)

# 2. Afficher une sous-matrice (ex : 100 premiers échantillons)
def afficher_heatmap(subset_size=100):
    subset = similarity_matrix[:subset_size, :subset_size]
    plt.figure(figsize=(10, 8))
    sns.heatmap(subset, cmap='viridis')
    plt.title(f"Heatmap des similarités cosinus ({subset_size} premiers échantillons)")
    plt.xlabel("Échantillon")
    plt.ylabel("Échantillon")
    plt.show()

# 3. Afficher les similarités d’un échantillon particulier
def afficher_similarites_echantillon(i, top_k=5):
    similarities = similarity_matrix[i].copy()
    similarities[i] = -1  # Ignore la similarité avec soi-même
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    print(f"\nIndices des {top_k} plus similaires à l'échantillon {i} :", top_indices)
    print("Scores :", similarities[top_indices])

# 4. Statistiques globales
def afficher_stats_globales():
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    all_similarities = similarity_matrix[mask]
    print("\nStatistiques globales sur les similarités (hors diagonale) :")
    print("Min      :", np.min(all_similarities))
    print("Max      :", np.max(all_similarities))
    print("Moyenne  :", np.mean(all_similarities))
    print("Médiane  :", np.median(all_similarities))
    plt.figure(figsize=(8, 5))
    plt.hist(all_similarities, bins=50, color='skyblue')
    plt.title("Distribution des similarités cosinus (hors diagonale)")
    plt.xlabel("Similarité cosinus")
    plt.ylabel("Nombre de paires")
    plt.show()

# ==============================
# Utilisation
# ==============================

if __name__ == "__main__":
    # Afficher la heatmap des 100 premiers échantillons
    afficher_heatmap(subset_size=1000)

    # Afficher les 5 plus similaires à l'échantillon 0 (modifiable)
    afficher_similarites_echantillon(i=0, top_k=5)

    # Afficher les statistiques globales et l'histogramme
    afficher_stats_globales()