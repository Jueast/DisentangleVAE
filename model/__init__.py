try:    # Works for python 3
    from model.abstract_VAE import *
    from model.flat_VAE import NaiveVAE, BetaVAE, MMDVAE
    from model.VLAE import VLAE, MMDVLAE, CNNVLAE
    from model.VAEGAN import VAEGAN
except: # Works for python 2
    from abstract_VAE import *
    from flat_VAE import NaiveVAE, BetaVAE
    from VLAE import VLAE, CNNVLAE