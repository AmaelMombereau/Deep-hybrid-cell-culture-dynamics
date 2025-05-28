# Étape 6 : Évaluation et visualisation des résultats

import matplotlib.pyplot as plt

def plot_predictions(times, C_true, C_pred, species_names=None):
    plt.figure(figsize=(12, 6))
    n_species = C_true.shape[1]

    for i in range(n_species):
        plt.subplot(n_species // 3 + 1, 3, i+1)
        plt.plot(times, C_true[:, i], label='Données réelles')
        plt.plot(times, C_pred[:, i], '--', label='Prédiction modèle')
        plt.xlabel('Temps (h)')
        plt.ylabel(f'Concentration {species_names[i] if species_names else f"Var{i+1}"}')
        plt.legend()
        plt.tight_layout()

    plt.show()

# Exemple d'appel après l'entraînement :
# model = ffnn_model ou lstm_model
# Après entraînement avec train_hybrid_model()

# Générer les prédictions finales
with torch.no_grad():
    def r_func(t):
        input_C = torch.tensor([C0], dtype=torch.float32)
        r_pred = model(input_C.unsqueeze(0))
        return r_pred.squeeze().numpy()

sol = solve_ivp(lambda t, C: odes(t, C, r_func, S), [0, 240], C0, t_eval=np.linspace(0, 240, 100))
C_pred_final = sol.y.T

# true_C_data = ... (charger ou récupérer les vraies données)
# times = np.linspace(0, 240, 100)
# plot_predictions(times, true_C_data, C_pred_final, species_names=[...])

# 🎯 Fin du pipeline hybride !

