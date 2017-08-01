try:    # Works for python 3
    from model.abstract_VAE import *
    from model.naive_VAE import NaiveVAE, BetaVAE
    from model.VLAE import VLAE
except: # Works for python 2
    from abstract_VAE import *
    from naive_VAE import NaiveVAE, BetaVAE
    from VLAE import VLAE