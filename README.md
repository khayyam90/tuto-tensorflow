# Tutoriel TensorFlow

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-API-D00000?logo=keras&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

Exemples d'architectures de réseaux de neurones construits avec TensorFlow/Keras, tous entraînés sur le dataset **MNIST**.

---

## Contenu

| Fichier | Architecture | Description |
|---|---|---|
| [1.py](1.py) | MLP | Réseau dense avec une ou plusieurs couches cachées |
| [2.py](2.py) | CNN | Réseau de convolution |
| [3-gan.py](3-gan.py) | GAN | Generative Adversarial Network |
| [autoencoder.py](autoencoder.py) | Autoencodeur | Débruitage d'images |

---

## Exemples de résultats

### GAN — génération de chiffres

![Animation GAN](animation-gan.gif)

### Surapprentissage

![Surapprentissage](animation-surapprentissage.gif)

### Autoencodeur — débruitage

![Débruitage](https://raw.githubusercontent.com/khayyam90/tuto-tensorflow/master/denoise.png)

---

## Prérequis

```bash
pip install tensorflow matplotlib
```

## Utilisation

```bash
python 1.py          # MLP
python 2.py          # CNN
python 3-gan.py      # GAN
python autoencoder.py  # Autoencodeur
```
