import scipy.io as sio
import numpy as np
import pandas as pd

# Charger un fichier .mat MATLAB
mat_data = sio.loadmat('data/HEK293_data.mat')  # Remplace par le bon chemin

# Liste des variables disponibles
print("Clés disponibles dans le fichier:", mat_data.keys())

# Supposons que 'C_data' contient les concentrations [temps x espèces x expériences]
C_data = mat_data['C_data']  # Remplace par la bonne clé si nécessaire

# Exemple : convertir en DataFrame pandas si besoin
n_experiments = C_data.shape[2]
all_data = []

for i in range(n_experiments):
    df = pd.DataFrame(C_data[:, :, i], columns=[f'Var_{j+1}' for j in range(C_data.shape[1])])
    df['Experiment'] = i + 1
    df['Time'] = np.arange(C_data.shape[0])  # ou charger à partir des données réelles
    all_data.append(df)

full_data = pd.concat(all_data, ignore_index=True)

# Aperçu des données
print(full_data.head())

# Enregistrer en CSV pour vérification
full_data.to_csv('HEK293_data_extracted.csv', index=False)

# Cette étape prépare les données à utiliser pour la modélisation et l'entraînement

# ➡️ Étape suivante : Calcul de la matrice de corrélation des réactions (Reaction Correlation Matrix S)
