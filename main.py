# main.py
from modele import FFNN, LSTMModel
from train_hybrid import train_hybrid_model
from odes import odes
from evaluation import plot_predictions
import numpy as np

# Charger S et données
S = np.load('data/reaction_correlation_matrix_S.npy')
C0 = np.load('data/C0.npy')  # À adapter selon tes données
true_C_data = np.load('data/true_C_data.npy')  # À adapter

# Définir modèle
model = LSTMModel(input_size=27, hidden_size=16, output_size=7)

# Entraîner modèle
model = train_hybrid_model(model, S, C0, [0, 240], np.linspace(0, 240, 100), true_C_data)

# Prédire et visualiser
times = np.linspace(0, 240, 100)
with torch.no_grad():
    def r_func(t):
        input_C = torch.tensor([C0], dtype=torch.float32)
        r_pred = model(input_C.unsqueeze(0))
        return r_pred.squeeze().numpy()

from scipy.integrate import solve_ivp
sol = solve_ivp(lambda t, C: odes(t, C, r_func, S), [0, 240], C0, t_eval=times)
C_pred_final = sol.y.T

plot_predictions(times, true_C_data, C_pred_final)

