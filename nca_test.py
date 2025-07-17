import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Charger un jeu de données (par exemple, Iris)
X, y = load_iris(return_X_y=True)

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraîner un modèle NCA
nca = NeighborhoodComponentsAnalysis(random_state=42)
nca.fit(X_scaled, y)

# Calculer les poids des caractéristiques (somme des carrés des coefficients)
feature_weights = np.sum(nca.components_**2, axis=0)

# Visualiser les poids des caractéristiques
plt.figure(figsize=(10, 6))
plt.scatter(range(len(feature_weights)), feature_weights, color='red', s=20)
plt.title("NCA Feature Screening Diagram")
plt.xlabel("Feature Index")
plt.ylabel("Feature Weight")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
