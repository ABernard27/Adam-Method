from .models import MultinomialRegressionModel, NeuralNetwork
from .models import reset_weights
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Création d'un dictionnaire pour regrouper tous les modèles utilisés et
# la perte associée
dict = {'multinomial': {'model': [MultinomialRegressionModel() for i in
                                  range(3)],
                        'loss': nn.NLLLoss()},
        'neural_network': {'model': [NeuralNetwork() for i in range(3)],
                           'loss': nn.CrossEntropyLoss()}
        }


def get_loss(data, model='multinomial', betas=(0.9, 0.999), eps=1e-8):
    """
    Cette fonction calcule la perte de toutes les méthodes pour un ensemble 
    de données en utilisantle modèle spécifié.

    Paramètres:
    - data (Dataloader): L'ensemble de données pour lequel calculer la perte.
    - which (list): Le type de modèle à utiliser pour calculer la perte.
      Doit être dans ['multinomial', 'neural_network'].
    - betas (tuple): Le paramètre beta pour Adam.
    - eps (float): Le paramètre epsilon pour Adam.

    Retourne:
    - stock (ndarray): Un tableau contenant les valeurs de perte calculées par
      chaque méthode d'optimisation.
    """
    # Vérification de la valeur fournie
    if model not in ['multinomial', 'neural_network']:
        raise ValueError("La valeur fournie n'est pas valide.")

    # Initialisation des modèles et des optimiseurs
    model_a, model_grad, model_rms = dict[model]['model']
    loss = dict[model]['loss']
    adam = optim.Adam(model_a.parameters(), lr=0.001, betas=betas, eps=eps)
    rms = optim.RMSprop(model_rms.parameters(), lr=0.001)
    adagrad = optim.Adagrad(model_grad.parameters(), lr=0.001)

    # Initialisation du tableau de stockage
    liste = [(adam, model_a), (rms, model_rms), (adagrad, model_grad)]
    err = np.zeros((3, len(data)))

    # Boucle de calcul
    for batch, (X, y) in enumerate(data):
        # Calcul de la perte pour chaque méthode et mise à jour des paramètres
        for i, (optimizer, model) in enumerate(liste):
            # Front propagation
            output = model(X.view(-1, 28*28))
            res = loss(output.squeeze(), y)

            # Nettoyage des gradients
            optimizer.zero_grad()

            # Back propagation
            res.backward()
            optimizer.step()

            # Stockage de la perte
            err[i, batch] = res.item()

    # Réinitialisation des paramètres pour de nouvelles utilisations
    reset_weights(model_a)
    reset_weights(model_rms)
    reset_weights(model_grad)
    
    return err
