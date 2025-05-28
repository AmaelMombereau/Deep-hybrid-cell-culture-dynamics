# Étape 5 : Boucle d'entraînement (couplage DNN + ODEs)

import torch
import torch.optim as optim

# Exemple simplifié de boucle d'entraînement

def train_hybrid_model(model, S, C0, t_span, times, true_C_data, n_epochs=50, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Fonction pour obtenir r(t) depuis le modèle (interpolé)
        def r_func(t):
            with torch.no_grad():
                t_tensor = torch.tensor([[t]], dtype=torch.float32)
                # Remplacer l'entrée par des concentrations ou autres si nécessaire
                input_C = torch.tensor([C0], dtype=torch.float32)  # Simplification
                r_pred = model(input_C.unsqueeze(0))  # Batch x Time x Features
                return r_pred.squeeze().numpy()
        
        # Résolution des ODEs
        sol = solve_ivp(lambda t, C: odes(t, C, r_func, S), t_span, C0, t_eval=times)
        
        C_pred = torch.tensor(sol.y.T, dtype=torch.float32)
        C_true = torch.tensor(true_C_data, dtype=torch.float32)

        loss = loss_fn(C_pred, C_true)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return model

# Exemple d'appel :
# model = ffnn_model ou lstm_model
# train_hybrid_model(model, S, C0, [0, 240], np.linspace(0, 240, 100), true_C_data)

# ➡️ Étape suivante : Évaluation et visualisation des résultats
