from .models import LogisticRegressionModel, NeuralNetwork
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Création d'un dictionnaire pour regrouper tous les modèles utilisés et
# la perte associée
dict = {'multinomial': {'model': [LogisticRegressionModel() for i in range(3)],
                        'loss': nn.NLLLoss()},
        'neural_network': {'model': [NeuralNetwork() for i in range(3)],
                           'loss': nn.CrossEntropyLoss()}
        }


def get_loss(data, model='multinomial'):
    """
    Cette fonction calcule la perte pour un ensemble de données en utilisant
    le modèle spécifié.

    Paramètres:
    - data (Dataloader): L'ensemble de données pour lequel calculer la perte.
    - which (list): Le type de modèle à utiliser pour calculer la perte.
      Doit être dans ['multinomial', 'neural_network'].

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
    adam = optim.Adam(model_a.parameters(), lr=0.01)
    rms = optim.RMSprop(model_rms.parameters(), lr=0.01)
    adagrad = optim.Adagrad(model_grad.parameters(), lr=0.01)

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

    return err
