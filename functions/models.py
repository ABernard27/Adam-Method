import torch.nn as nn

class MultinomialRegressionModel(nn.Module):
    def __init__(self, n_features=28*28, n_classes=10):
        """
        Initialise une nouvelle instance de la classe
        MultinomialRegressionModel.

        Paramètres:
            self: l'objet courant
            n_features: nombre de caractéristiques
            n_classes: nombre de classes

        Renvoie:
            None
        """
        super(MultinomialRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Fonction forward de la classe MultinomialRegressionModel.

        Paramètres:
            x (Tensor): Tenseur de taille (batch_size, n_features).

        Renvoie:
            predictions (Tensor): Tenseur de taille (batch_size, n_classes).
        """
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Initialise une nouvelle instance de la classe NeuralNetwork.

        Paramètres:
            None

        Renvoie:
            None
        """
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        Fonction forward de la classe NeuralNetwork.

        Paramètres:
            x (torch.Tensor): Le tenseur d'entrée à traiter.

        Renvoie:
            torch.Tensor: Le tenseur après avoir appliqué la série
            d'opérations.
        """
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

def reset_weights(model):
    """
    Réinitialise les poids d'un modèle.
    
    Paramètres:
        model (nn.Module): Le modèle dont les poids doivent être réinitialisés.
        
    Renvoie:
        None
    """
    for param in model.parameters():
        if len(param.shape) > 1:  # Vérifie si le paramètre est une matrice (poids)
            nn.init.xavier_uniform_(param)
        else:  # Vérifie si le paramètre est un vecteur (biais)
            nn.init.zeros_(param)
