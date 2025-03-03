import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis

def reduce_and_visualize_with_nca(features, labels, threshold=0.3, chunk_size=5000):
    print("Filtering features based on threshold using NCA...")

    num_samples, num_features, num_frames = features.shape
    print(f"Input shape: {features.shape}")

    # Reshape features to (num_samples * num_frames, num_features)
    reshaped_features = features.reshape(-1, num_features)
    reshaped_labels = np.repeat(labels, num_frames)

    # Initialize variables for storing results
    filtered_features = []
    nca = None

    # Process the data in chunks
    for i in range(0, reshaped_features.shape[0], chunk_size):
        print(f"Processing block {i // chunk_size + 1}...")
        start = i
        end = min(i + chunk_size, reshaped_features.shape[0])
        
        chunk_features = reshaped_features[start:end]
        chunk_labels = reshaped_labels[start:end]

        nca = NeighborhoodComponentsAnalysis(random_state=42)
        
        try:
            nca.fit(chunk_features, chunk_labels)
        except MemoryError:
            print("MemoryError: Reduce the dataset size or optimize memory usage.")
            return None, None

        # Debug: Visualize nca.components_
        plt.figure(figsize=(12, 8))
        plt.imshow(nca.components_, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f"NCA Components (Block {i // chunk_size + 1})")
        plt.xlabel("Feature Index")
        plt.ylabel("Component Index")
        plt.tight_layout()
        plt.show()

        # Extract feature weights
        feature_weights = np.linalg.norm(nca.components_, axis=0)

        # Debug: Visualize feature weights
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(feature_weights)), feature_weights, 'bo-')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        plt.title(f"Feature Weights (Block {i // chunk_size + 1})")
        plt.xlabel("Feature Index")
        plt.ylabel("Weight")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Screen features based on the threshold
        important_features_mask = feature_weights >= threshold
        filtered_chunk = chunk_features[:, important_features_mask]

        print(f"Block {i // chunk_size + 1}:")
        print(f" - Number of features before filtering: {num_features}")
        print(f" - Number of features after filtering: {filtered_chunk.shape[1]}")

        filtered_features.append(filtered_chunk)

    # Combine all filtered chunks into a single array
    final_filtered_features = np.vstack(filtered_features)

    # Reshape back to (num_samples, num_filtered_features, num_frames)
    num_filtered_features = final_filtered_features.shape[1]
    final_filtered_features = final_filtered_features.reshape(num_samples, num_frames, num_filtered_features).transpose(0, 2, 1)

    return final_filtered_features, nca



import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis

# Génération de données synthétiques
def generate_synthetic_data(num_samples=100, num_features=5, num_frames=10):
    np.random.seed(42)
    
    # Création de labels (2 classes)
    labels = np.random.randint(0, 2, size=num_samples)
    
    # Création de caractéristiques importantes (les 2 premières)
    important_feat1 = np.random.normal(loc=labels*1.0 - 0.5, scale=0.5, size=(num_samples, num_frames))
    important_feat2 = np.random.normal(loc=labels*0.8 - 0.4, scale=0.6, size=(num_samples, num_frames))
    
    # Création de caractéristiques non importantes (bruit)
    noise_features = np.random.normal(size=(num_samples, num_features-2, num_frames))
    
    # Combinaison des caractéristiques
    features = np.concatenate([
        important_feat1[:, np.newaxis, :],
        important_feat2[:, np.newaxis, :],
        noise_features
    ], axis=1).transpose(0, 2, 1)  # Shape (num_samples, num_frames, num_features)
    
    return features.transpose(0, 2, 1), labels  # Retourne shape (num_samples, num_features, num_frames)

# Chargement des données synthétiques
features, labels = generate_synthetic_data(num_samples=100, num_features=5, num_frames=10)

# Affichage de la structure des données
print("Shape des caractéristiques:", features.shape)
print("Shape des labels:", labels.shape)

# Exécution de la fonction avec visualisation debug
filtered_features, nca = reduce_and_visualize_with_nca(
    features=features,
    labels=labels,
    threshold=0.5,
    chunk_size=30
)

# Affichage des résultats finaux
print("\nRésultats finaux:")
print("Shape des caractéristiques filtrées:", filtered_features.shape)
print("Nombre de caractéristiques conservées:", filtered_features.shape[1])

# Visualisation des distributions des caractéristiques
plt.figure(figsize=(12, 6))
for i in range(features.shape[1]):
    plt.hist(features[:, i, :].flatten(), bins=30, alpha=0.5, label=f'Feature {i+1}')
plt.title("Distribution des caractéristiques originales")
plt.xlabel("Valeur")
plt.ylabel("Fréquence")
plt.legend()
plt.show()
