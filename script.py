# %% ------------------------------------------------------ #
#                   Importation de base
# --------------------------------------------------------- #
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from functions.comparison import get_loss
import matplotlib.pyplot as plt

# %% ------------------------------------------------------ #
#               Chargement du dataset MNIST
# --------------------------------------------------------- #

dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())

# Construction du dataloader pour avoir des sous échantillons
data = DataLoader(dataset, batch_size=128, shuffle=True)

# Exemples d'image de MNIST
image = MNIST(root="./data", train=True, download=True)

# Affichage des 8 premoères images en deux lignes
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    img, label = image[i]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.suptitle("Exemples d'images de MNIST", fontsize=20)
plt.tight_layout()
plt.savefig("./graph/mnist.png")
plt.show()

# %% ------------------------------------------------------ #
#               Régression multinomiale
# --------------------------------------------------------- #

# Calcul et affichage pour la régression multinomial
betas = (0.9, 0.999)
eps = 1e-8
err_mult = get_loss(data, "multinomial", betas, eps)

plt.figure(figsize=(10, 5))
plt.plot(err_mult[0], color="red")
plt.plot(err_mult[1], color="darkcyan")
plt.plot(err_mult[2], color="purple")
plt.legend(["Adam", "RMSprop", "Adagrad"])
plt.xlabel("Timestep")
plt.ylabel("Valeur de la négative log-vraisemblance")
plt.title(
    "Comparaison des méthodes d'optimisation pour une régression multinomiale"
)

# Amélioration affichage
plt.ylim(0, 2)
plt.xlim(0, 200)
plt.grid()

# Enregistrement du graphe
plt.savefig("./graph/multi_graph.png")

plt.show()

# %% ------------------------------------------------------ #
#                   Réseau de neurone
# --------------------------------------------------------- #

# Calcul et affichage pour le réseau de neurone
betas = (0.9, 0.999)
eps = 1e-8
err_nn = get_loss(data, "neural_network", betas, eps)

plt.figure(figsize=(10, 5))
plt.plot(err_nn[0], color="red")
plt.plot(err_nn[1], color="darkcyan")
plt.plot(err_nn[2], color="purple")
plt.legend(["Adam", "RMSprop", "Adagrad"])
plt.xlabel("Timestep")
plt.ylabel("Valeur de la négative log-vraisemblance")
plt.title("Comparaison des méthodes d'optimisation sur un réseau de neurone")

# Amélioration affichage
plt.ylim(0, 2)
plt.xlim(0, 200)
plt.grid()

# Enregistrement du graphe
plt.savefig("./graph/nn_graph.png")

plt.show()
