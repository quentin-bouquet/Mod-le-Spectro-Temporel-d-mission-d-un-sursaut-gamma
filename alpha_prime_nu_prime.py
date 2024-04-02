import const 
from scipy.special import gamma





def min(a, b):
    if a < b:
        return a
    else:
        return b
#TODO: implement p_tildt when y'm  < y'c
#todo : beta's
def p_tildt(gamma_prime_c, gamma_prime_m): 
    if gamma_prime_m >= gamma_prime_c:
        return 2 
    else:
        return (3 * beta_u * - 2 * beta_u * beta_d ** 2 + beta_d ** 3) / (beta_u - beta_d) - 2

#TODO: n'R à rajouter, et b_prime  
def c_11(nu_prime, nu_prime_c, nu_prime_m, gamma_prime_m, gamma_prime_c):
    p_tildt = p_tildt(gamma_prime_c, gamma_prime_m)
    nu_min = min(nu_prime_m, nu_prime_c)
    gamma_min = min(gamma_prime_m, gamma_prime_c)
    return 2 ** 6 *const.pi ** (5/6) * (p_tildt + 2) * (p_tildt - 1) * const.q_e * n_prime_r / (15 * (3*p_tildt + 2) * gamma(5 / 6) * gamma_min ** 5 * b_prime) * (nu_prime / nu_min) ** (-5/3)

#TODO: n'R et b_prime à rajouter
def c_14(nu_prime, nu_prime_c, nu_prime_m, gamma_prime_m, gamma_prime_c):
    p_tildt = p_tildt(gamma_prime_c, gamma_prime_m)
    nu_min = min(nu_prime_m, nu_prime_c)
    gamma_min = min(gamma_prime_m, gamma_prime_c)
    return (2 ** ((3*p_tildt + 8) / 2) * (p_tildt - 1) * gamma(3/2 + p_tildt/4) * gamma(1/6 + p_tildt/4) * const.q_e * n_prime_r ) / (3**(3/2) * const.pi **((p_tildt + 1 )/2) * gamma(2 + p_tildt/4) * gamma_min ** 5 *b_prime) *(nu_prime / nu_min) ** (-(p_tildt + 4) / 2)

def nu_prime_zero(nu_prime_min): 
    return ((5*(3 * p_tildt +2) * gamma(3/2 + p_tildt/4) *gamma(11/6 + p_tildt/4) * gamma(1/6 + p_tildt/4) * gamma(5/6)) / (p_tildt + 2) * gamma(2 + p_tildt/4))** (6 / (3*p_tildt +2)) * ((2**( 3 * (3 * p_tildt -4))) / (27 * const.pi ** (3*p_tildt + 8) )) ** (1 / (3*p_tildt + 2)) * nu_prime_min

def alpha_prime_nu_prime(nu_prime, nu_prime_c, nu_prime_m, gamma_prime_m, gamma_prime_c):
    p_tildt = p_tildt(gamma_prime_c, gamma_prime_m)
    nu_min = min(nu_prime_m, nu_prime_c)
    nu_zero = nu_prime_zero(nu_min)
    if nu_prime < nu_zero:
        alpha_prime_nu_prime_zero = c_11(nu_zero, nu_prime_c, nu_prime_m, gamma_prime_m, gamma_prime_c)
        eta =  -5/3
    else : 
        alpha_prime_nu_prime_zero = c_14(nu_zero, nu_prime_c, nu_prime_m, gamma_prime_m, gamma_prime_c)
        eta = -(p_tildt + 4) / 2
    return alpha_prime_nu_prime_zero * (nu_prime / nu_zero) ** eta