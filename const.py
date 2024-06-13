#Fichier contenant les constantes utiles au problème:
#Pour coder les constantes on utilisera les constantes disponibles dans la bibliothèque scipy
import scipy.constants as const
import numpy as np


celer = const.c
pi = const.pi
k_b= const.Boltzmann
q_e = const.e
m_p = const.physical_constants['proton mass'][0]
m_e = const.physical_constants['electron-proton mass ratio'][0]*m_p
sigma_T = (8 *const.pi / 3 ) *  const.physical_constants['classical electron radius'][0]**2
epsi_e = 0.14
epsi_b = 4.12*(1/np.power(10,5))