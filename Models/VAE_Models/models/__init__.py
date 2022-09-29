from Models.VAE_Models.models.base import *
from Models.VAE_Models.models.vanilla_vae import *
from Models.VAE_Models.models.gamma_vae import *
from Models.VAE_Models.models.beta_vae import *
from Models.VAE_Models.models.wae_mmd import *
from Models.VAE_Models.models.cvae import *
from Models.VAE_Models.models.hvae import *
from Models.VAE_Models.models.vampvae import *
from Models.VAE_Models.models.iwae import *
from Models.VAE_Models.models.dfcvae import *
from Models.VAE_Models.models.mssim_vae import MSSIMVAE
from Models.VAE_Models.models.fvae import *
from Models.VAE_Models.models.cat_vae import *
from Models.VAE_Models.models.joint_vae import *
from Models.VAE_Models.models.info_vae import *
# from .twostage_vae import *
from Models.VAE_Models.models.lvae import LVAE
from Models.VAE_Models.models.logcosh_vae import *
from Models.VAE_Models.models.swae import *
from Models.VAE_Models.models.miwae import *
from Models.VAE_Models.models.vq_vae import *
from Models.VAE_Models.models.betatc_vae import *
from Models.VAE_Models.models.dip_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {'HVAE':HVAE,
              'LVAE':LVAE,
              'IWAE':IWAE,
              'SWAE':SWAE,
              'MIWAE':MIWAE,
              'VQVAE':VQVAE,
              'DFCVAE':DFCVAE,
              'DIPVAE':DIPVAE,
              'BetaVAE':BetaVAE,
              'InfoVAE':InfoVAE,
              'WAE_MMD':WAE_MMD,
              'VampVAE': VampVAE,
              'GammaVAE':GammaVAE,
              'MSSIMVAE':MSSIMVAE,
              'JointVAE':JointVAE,
              'BetaTCVAE':BetaTCVAE,
              'FactorVAE':FactorVAE,
              'LogCoshVAE':LogCoshVAE,
              'VanillaVAE':VanillaVAE,
              'ConditionalVAE':ConditionalVAE,
              'CategoricalVAE':CategoricalVAE}
