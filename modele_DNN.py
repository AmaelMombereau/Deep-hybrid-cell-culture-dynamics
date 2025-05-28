# Étape 3 : Définir les modèles de réseaux de neurones (FFNN et LSTM)

import torch
import torch.nn as nn

# Modèle FFNN (Feedforward Neural Network)
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNN, self).__init__()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())  # ou Tanh, Sigmoid selon les besoins
            input_size = h
        layers.append(nn.Linear(input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Modèle LSTM (Long Short-Term Memory)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Exemples d'initialisation
input_size = 27  # Nombre de variables (par exemple 27 espèces)
output_size = 7  # Nombre de taux réactionnels (comme dans l'article)

ffnn_model = FFNN(input_size=input_size, hidden_sizes=[16, 8], output_size=output_size)
lstm_model = LSTMModel(input_size=input_size, hidden_size=16, output_size=output_size)

# ➡️ Étape suivante : Définir les équations différentielles (ODEs)
