import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis

# Simulated dataset (replace with actual dataset)
num_samples = 1000  # Number of samples
num_features = 272  # Number of features
num_frames = 188  # Number of frames

# Generate random features and labels (replace with actual data)
features = np.random.rand(num_samples, num_features, num_frames)
labels = np.random.randint(0, 4, size=num_samples)  # 4 classes

# Step 1: Collapse the temporal dimension by averaging across frames
mean_features = np.mean(features, axis=2)  # Shape: (num_samples, num_features)

# Step 2: Apply NCA to learn feature weights
nca = NeighborhoodComponentsAnalysis(random_state=42)
nca.fit(mean_features, labels)

# Step 3: Extract feature weights
feature_weights = np.linalg.norm(nca.components_, axis=0)  # Shape: (num_features,)

# Step 4: Visualize feature weights
plt.figure(figsize=(12, 6))
plt.scatter(range(len(feature_weights)), feature_weights, color='red', s=20)
#plt.bar(range(len(feature_weights)), feature_weights, color='blue', alpha=0.7)
plt.title("Feature Weights Before Reduction (NCA)")
plt.xlabel("Feature Index")
plt.ylabel("Weight")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
