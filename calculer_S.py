# Étape 2 : Calcul de la matrice de corrélation des réactions (Reaction Correlation Matrix S)

import numpy as np
from sklearn.decomposition import PCA

# Supposons que C_data est déjà chargé depuis l'étape 1 (dimensions : [temps x espèces x expériences])
# Nous allons "aplatir" les données : empiler toutes les expériences dans une seule matrice 2D
timepoints, variables, experiments = C_data.shape

# Reshape : [temps*expériences, variables]
data_flat = C_data.transpose(2, 0, 1).reshape(experiments * timepoints, variables)

# Normalisation colonne par colonne (par maximum absolu)
data_norm = data_flat / np.max(np.abs(data_flat), axis=0)

# PCA pour réduire l'espace d'états
n_components = 7  # Nombre de composantes principales (comme dans l'article)
pca = PCA(n_components=n_components)
scores = pca.fit_transform(data_norm)
coefficients = pca.components_.T

# Reconstruction de la matrice de corrélation S
S = coefficients * np.max(np.abs(data_flat), axis=0)[:, np.newaxis]

print("Matrice de corrélation S (dimensions):", S.shape)

# Sauvegarder la matrice S pour utilisation ultérieure
np.save('reaction_correlation_matrix_S.npy', S)

# ➡️ Étape suivante : Définir les modèles de réseaux de neurones (FFNN et LSTM)
