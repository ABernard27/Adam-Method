# %%
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from functions.comparison import get_loss
import matplotlib.pyplot as plt


# %%

# Instantiate the MNIST dataset
dataset = MNIST(root='./data',
                train=True,
                download=True,
                transform=ToTensor())

# Extraire les caractéristiques et les étiquettes
data = DataLoader(dataset, batch_size=128, shuffle=True)

# %%
# Calcul et affichage pour la régression multinomial
err_mult = get_loss(data, 'multinomial')

plt.figure(figsize=(10, 5))
plt.plot(err_mult[0], color='red')
plt.plot(err_mult[1], color='darkcyan')
plt.plot(err_mult[2], color='purple')
plt.ylim(0, 2)
plt.xlim(0, 200)
plt.grid()
plt.legend(['Adam', 'RMSprop', 'Adagrad'])
plt.xlabel("Timestep")
plt.ylabel("Valeur de la négative log-vraisemblance")
plt.title("Comparaison des méthodes d'optimisation pour une régression multinomiale")
plt.show()
# %%
# Calcul et affichage pour le réseau de neurone
err_nn = get_loss(data, 'neural_network')

plt.figure(figsize=(10, 5))
plt.plot(err_nn[0], color='red')
plt.plot(err_nn[1], color='darkcyan')
plt.plot(err_nn[2], color='purple')
plt.ylim(0, 2)
plt.xlim(0, 200)
plt.grid()
plt.legend(['Adam', 'RMSprop', 'Adagrad'])
plt.xlabel("Timestep")
plt.ylabel("Valeur de la négative log-vraisemblance")
plt.title("Comparaison des méthodes d'optimisation sur un réseau de neurone")
plt.show()

#%%
import matplotlib.pyplot as plt
from torchvision import datasets

# Chargement du jeu de données MNIST
mnist_data = datasets.MNIST(root='./data', train=True, download=True)

# Affichage des 10 premières images en deux lignes
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    img, label = mnist_data[i]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.suptitle("Exemples d'images de MNIST", fontsize=20)

plt.tight_layout()
plt.show()
plt.savefig("mnist.png")