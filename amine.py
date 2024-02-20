import numpy as np 
import matplotlib.pyplot as plt
import scipy.constants as const
import numba as nb
import cython as cy
from tools import min, max, choix_1, choix_2

gamma_chapeau_prime = 1 #TODO : Find the value of gamma_chapeau_prime
sigma_T = (8 *const.Pi / 3 ) *  const.physical_constants['classical electron radius'][0]**2
P_prime = 1 #Gas pressure #TODO: Find the value of P_prime
e_prime_i = P_prime / (gamma_chapeau_prime - 1) 
def B_prime(epsilon_B):
    return np.sqrt(8 * const.pi * epsilon_B * e_prime_i) #TODO : Find e_prime_i
beta_u = 1 #TODO : Find the value of beta_u
beta_d = 1 #TODO : Find the value of beta_d

p = (3*beta_u - 2*beta_u * beta_d**2 + beta_d**3) / (beta_u - beta_d) - 2 

gamma_maj = 1 #TODO : Find the value of gamma_maj
def gamma_prime_c(epsilon_B, t):
    return (3 * const.electron_mass * const.c * gamma_maj) / (4 * sigma_T * epsilon_B * e_prime_i * t) #TODO : t ???

epsilon_e = 0.14 #TODO : Verify the value of epsilon_e

gamma_prime_m = max(1, (p - 2) / (p - 1) * const.proton_mass / const.electron_mass * epsilon_e *(gamma_maj -1))

def n_prime(no):
    return no * ((gamma_chapeau_prime * gamma_maj +1)/ (gamma_chapeau_prime -1))


def n_prime_R(no) 
    return n_prime(no) * min(1, (p - 2) / (p - 1) * const.proton_mass / const.electron_mass * epsilon_e *(gamma_maj -1))

def epsilon_prime_nu_prime_p(p, epsilon_B, no):
    return 0.88 * 256 / 27 * (p - 1) / (3*p -1) * const.elementary_charge**3 / (const.electron_mass * const.c**2) * n_prime_R(no) * B_prime(epsilon_B)
TODO : #verify "elementary charge"

def nu_prime_m(epsilon_B):
    return 3/16 * gamma_prime_m**2 * const.elementary_charge * B_prime(epsilon_B) / (const.electron_mass * const.c)

def nu_prime_c(epsilon_B, t):
    return 3/16 * gamma_prime_c(epsilon_B, t)**2 * const.elementary_charge * B_prime(epsilon_B) / (const.electron_mass * const.c)



def epsilon_prime_nu_prime(epsilon_B, no, t, p, nu_prime):
    if nu_prime_m <= nu_prime_c: 
        return epsilon_prime_nu_prime_p(p, epsilon_B, no) * choix_1(nu_prime, nu_prime_m(epsilon_B), nu_prime_c(epsilon_B, t))
    else:
        return epsilon_prime_nu_prime_p(p, epsilon_B, no) * choix_2(nu_prime, nu_prime_c(epsilon_B, t), nu_prime_m(epsilon_B))   

