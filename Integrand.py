import numpy as np 
import matplotlib.pyplot as plt
import numba as nb
import cython as cy
import scipy
import const 
from scipy.special import gamma
from scipy.special import kv
from scipy import *
from scipy.integrate import odeint
import math as m
import mpmath

# Defining global variables 
n0 = 1 * 10 ** 3 
epsilon_B = 0.0001 
tetav_c = 0.387 
sh_max=100
En=10**43
mu_val = 0 # mu_val va dépendre de teta et phi 

def min(a, b):
    if a < b:
        return a
    else:
        return b

#Définition des fonctions de Bessel de première espèce (avec une limite finie en 0)
def K1(x) :
    return kv(1, x)

def K2(x) :
    return kv(2, x)

def K3(x) :
    return kv(3, x)

# Développement limité des fonctions de bessel pour x proche de 0
'''
def f1(x):
    if x < 650:
        return K3(x)/K2(x)
    else :
        return 1 + 2.5/x
'''

def f1(x):

    res = np.zeros_like(x)
    res[x < 650] = K3(x)/K2(x)
    res[x>=650] = 1 + 2.5/x
    return res

# Développement limité des fonctions de bessel pour x proche de 0
def f2(x):
    if x < 650:
        return K2(x)/(3*K3(x) + K1(x) - 4*K2(x))
    else: 
        #return x/(3 + np.power(x,2) - 4*x)
        return 5 / 3


# facteur de lorentz 
def gamma_maj(ze): 
    return f1(ze) - 1/ze

# Modified Lorentz factor 
def gamma_chapeau_prime(ze):
    return 1 + 4*f2(ze)/ze

# Vectorizing the functions
gamma_maj =np.vectorize(gamma_maj)
gamma_chapeau_prime = np.vectorize(gamma_chapeau_prime)



# Depend on tetav which is a global constant
def mu(teta, phi):
    return np.sin(teta) * np.sin(tetav_c) * np.cos(phi) + np.cos(teta) * np.cos(tetav_c)


# Depend on distibution of E, n0 is a global constant 
def C_BM(E):
    return np.sqrt(17 * E / (8 * np.pi * n0 * const.m_p * const.celer**5))

def C_ST(E):
    return (2/5) * 1.15 * (E / (n0 * const.m_p * const.celer**5))**(1/5)

def radius(T, mu_val):
    C_BM_val = C_BM(En)
    C_ST_val = C_ST(En)
    mu_val
    def f(R, t):
        t =T+ mu_val * R / const.celer
        t_safe = np.where(t>0, t, 0.000001)
        mine = np.minimum((C_BM_val**2) * (t_safe) ** (-3) + (C_ST_val**2) * (t_safe) ** (-6 / 5), sh_max**2)
        return np.sqrt(mine / (1 + mine)) * const.celer
    
    Rp = -T*const.celer/mu_val
    Rs = 1e25
    cpt = 0
    while abs(Rp - Rs)/abs(Rs) > 10 ** (-2) and cpt < 20:
        cpt += 1
        #print(Rs)
        time1 = [0,T+mu_val*Rs/const.celer]
        sol = odeint(f,y0=[-T*const.celer/mu_val], t=time1)
        Rp = Rs
        Rs = sol[1][0]
    return Rs

# Retourne beta_sh
def shock_speed(E, T, teta, phi):
    # mu_val est une variable globale qui est modifiée dans le calcul de l'intégrante. 
    mu_val = mu(teta, phi)
    radius_value = radius(T, mu_val)
    t = T + mu_val * radius_value / const.celer
    mine = min((C_BM(E)**2) * (t) ** (-3) + (C_ST(E)**2) * (t) ** (-6/5), sh_max**2)
    beta_sh = np.sqrt(mine / (1 + mine))
    return beta_sh


# Détermination de Ksi, car il est défini implicitement ici. 
#Find the rpoblem with ksi ici
from scipy.optimize import fsolve

def eq(gamma_sh,ze):
    a = (gamma_maj(ze) + 1)*(1 + gamma_chapeau_prime(ze)*(gamma_maj(ze) - 1))**2
    b = 2 + gamma_chapeau_prime(ze)*(2 - gamma_chapeau_prime(ze)*(gamma_maj(ze) - 1))
    return gamma_sh**2 - a/b

def sol(gamma_sh): #Sol is zeta
    def func(ze):
        return eq(gamma_sh, ze)
    solve = fsolve(func, 1)
    return solve[0]

sol_vect = np.vectorize(sol)
g_m = np.vectorize(gamma_maj)
g_c = np.vectorize(gamma_chapeau_prime)


#We can now calculate values of beta's

def beta_u(E, t, teta, phi):
    return shock_speed(E, t, teta, phi)

def beta(E, t, teta, phi):
    beta_sh = shock_speed(E, t, teta, phi)
    zeta = sol(beta_sh)
    return np.sqrt(1 - 1/(gamma_maj(zeta) ** 2))

def beta_d(E, T, teta, phi):
    beta_sh = shock_speed(E, T, teta, phi)
    beta_val = beta(E, T, teta, phi)
    return (beta_sh - beta_val) / (1 - beta_val * beta_sh)

def p(E, T, teta, phi):
    beta_u_val = beta_u(E, T, teta, phi)
    beta_d_val = beta_d(E, T, teta, phi)
    a = 3 * beta_u_val - 2 * beta_u_val * ( beta_d_val ** 2  ) + beta_d_val ** 3
    b = beta_u_val - beta_d_val
    return a/b - 2

def gamma_shf(E, n0, sh_max,t, teta, tetav, phi):
    beta_sh = shock_speed(E, n0, sh_max, t, teta, tetav, phi)
    return np.sqrt(1/(1 - beta_sh**2))

def n_prime(gamma_sh):
    return 4*gamma_maj(sol(gamma_sh))*n0

def e_i_p(gamma_sh):
    return const.m_p*((const.celer)**2) * (gamma_maj(sol(gamma_sh)) - 1)*n_prime(gamma_sh)


def gamma_prime_c( gamma_sh,t):
    return (3 * const.m_e * const.celer* gamma_maj(sol(gamma_sh)))/ (4 * const.sigma_T * epsilon_B *e_i_p( gamma_sh) * t*n0)


def gamma_prime_m( gamma_sh, p):
    arg = (((p - 2)*const.m_p)/((p - 1)*const.m_e))*const.epsi_e*(gamma_maj(sol(gamma_sh)) - 1)
    return max(1, arg)

def B_prime( gamma_sh):
        return (np.sqrt(8*const.pi*epsilon_B*e_i_p(gamma_sh)))

def nu_prime_m(gamma_sh, p):
    return 3/16 * gamma_prime_m(gamma_sh, p)**2 * const.q_e*B_prime(gamma_sh) / (const.m_e * const.celer)

def nu_prime_c( t, gamma_sh):
    return 3/16 * gamma_prime_c(gamma_sh, t)**2 * const.q_e *B_prime(gamma_sh) / (const.m_e * const.celer)

def power_I(x,y):
    return 1/np.power(x,y)

#Rapport de comparaison pour les refroidissemts entre nu_prim_c et nu_prim_m
def Kappa(t,gamma_sh,p):
    return nu_prime_c(t,gamma_sh)/nu_prime_m(gamma_sh, p)

def n_prime_r(gamma_sh, p):
    n_prime_val = n_prime(gamma_sh)
    min_val = min(1,(((p - 2)*const.m_p)/((p - 1)*const.m_e))*const.epsi_e*(gamma_maj(sol(gamma_sh)) - 1))
    result = n_prime_val * min_val
    #print(f"n_prime_r calculation: n_prime={n_prime_val}, min_val={min_val}, result={result}")
    return n_prime(gamma_sh)*min(1,(((p - 2)*const.m_p)/((p - 1)*const.m_e))*const.epsi_e*(gamma_maj(sol(gamma_sh)) - 1))

def FastcEmission(x,p,t,gamma_sh):
    if x > Kappa(t,gamma_sh,p):
        return np.power(Kappa(t,gamma_sh,p),-1/3)*np.power(x,1/3)
    elif x >= Kappa(t,gamma_sh,p) and x <1:
        return np.power(Kappa(t,gamma_sh,p),1/2)*power_I(x,-1/2)
    else:
         return power_I(Kappa(t,gamma_sh,p),-1/2)*power_I(x,-p/2)

#Cas du slow-cooling
def SlowcEmission(x,p,t,gamma_sh):
    if x < 1:
        return np.power(x,1/3)
    elif x>=1 and x < Kappa(t,gamma_sh,p):
        return power_I(x,-(p-1)/2)
    else:
        return power_I(Kappa(t,gamma_sh,p),-1/2)*power_I(x,-p/2)

#Fonction d'émission
def Emission_function(x,p,t,gamma_sh):
    if Kappa(t,gamma_sh,p) > 1:
        return SlowcEmission(x,p,t,gamma_sh)
    else:
        return FastcEmission(x,p,t,gamma_sh)


def fact_denorm(p,gamma_sh):
    return 0.88 * ((256 * (p - 1) * (const.q_e ** 3)) / (27 * (3 * p - 1) * const.m_e * (const.celer ** 2))) * n_prime_r(gamma_sh, p) * B_prime(gamma_sh)



def p_tildt(p_val, gamma_prime_m, gamma_prime_c, beta_u_val, beta_d_val):
    if gamma_prime_m >= gamma_prime_c:
        return 2 
    else:
        return (3 * beta_u_val - 2 * beta_u_val * beta_d_val ** 2 + beta_d_val ** 3) / (beta_u_val - beta_d_val) - 2



def c_11(nu_prime, n_prime_r_val, b_prime, nu_prime_c_val, nu_prime_m_val, gamma_prime_m_val, gamma_prime_c_val, p_tildt_val):
    nu_min = min(nu_prime_m_val, nu_prime_c_val)
    gamma_min = min(gamma_prime_m_val, gamma_prime_c_val)

    # Définition des termes 
    a = np.power(2,6)*np.power(const.pi,5/6)*(p_tildt_val + 2)*(p_tildt_val - 1)*const.q_e*n_prime_r_val
    b = 15*(3*p_tildt_val + 2)*gamma(5/6)*np.power(gamma_min,5)*b_prime
    result = (a/b)*np.power(nu_prime/nu_min, -5/3)
    #print(f"c_11 calculation: a={a}, b={b}, nu_min={nu_min}, result={result}")
    return (a/b)*np.power(nu_prime/nu_min, -5/3)

def c_14(nu_prime, n_prime_r_val, b_prime, nu_prime_c_val, nu_prime_m_val, gamma_prime_m_val, gamma_prime_c_val, p_tildt_val):
    nu_min = min(nu_prime_m_val, nu_prime_c_val)
    gamma_min = min(gamma_prime_m_val, gamma_prime_c_val)

    #Définitions des termes 
    a = np.power(2, ((3 * p_tildt_val + 8) / 2))
    b = (p_tildt_val - 1) * gamma(3 / 2 + p_tildt_val / 4) * gamma(11 / 6 + p_tildt_val / 4) *\
          gamma(1 / 6 + p_tildt_val / 4) * const.q_e * n_prime_r_val
    c = np.power(3, 3 /2) * np.power(const.pi, (p_tildt_val + 1) / 2) *\
          gamma(2 + p_tildt_val / 4) * np.power(gamma_min, 5) * b_prime 
    result = (a * b) / c * np.power(nu_prime / nu_min, -(p_tildt_val + 4) / 2)
    #print(f"c_14 calculation: a={a}, b={b}, c={c}, nu_min={nu_min}, result={result}")
    return (a * b) / c * np.power(nu_prime / nu_min, -(p_tildt_val + 4) / 2)
 
def nu_prime_zero(nu_prime_c_val, nu_prime_m_val, p_tildt_val):
    nu_prime_min = min(nu_prime_m_val, nu_prime_c_val)
    a = 5*(3*p_tildt_val + 2)*gamma(3/2 + p_tildt_val/4)*gamma(11/6 + p_tildt_val/4)*gamma(1/6 + p_tildt_val/4)*gamma(5/6)
    b = (p_tildt_val + 2)*gamma(2 + p_tildt_val/4)
    c = np.power(np.power(2,3*(3*p_tildt_val - 4))/27*np.power(const.pi, 3*p_tildt_val + 8),1/(3*p_tildt_val + 2))
    return np.power(a/b,6/(3*p_tildt_val + 2))*c*nu_prime_min

def alpha_prime_nu_prime(nu_prime, p_tildt_val, nu_prime_zero_val, c_11_val, c_14_val):
    #calculs
    if nu_prime < nu_prime_zero_val:
        alpha_prime_nu_prime_zero = c_11_val
        eta =  -5/3
    else : 
        alpha_prime_nu_prime_zero = c_14_val
        eta = -(p_tildt_val + 4) / 2
    return alpha_prime_nu_prime_zero * (nu_prime / nu_prime_zero_val) ** eta

def nu_prime(nu, xi, beta, mu):
    return gamma_maj(xi) * (1 - beta * mu) * nu 

def alpha_nu(mu_val, beta_sh, beta_val, alpha_prime_nu_prime_val):
    zeta = sol(beta_sh)
    alpha_nu = gamma_maj(zeta)*(1-beta_val*mu_val)*alpha_prime_nu_prime_val
    return alpha_nu

def delta_s(beta_sh, R,  mu_val):
    zeta = sol(beta_sh)
    return R/((12*np.abs(mu_val - beta_sh)*(gamma_maj(zeta))**2))

def tan_nu(alpha_nu_val, delta_s_val):
    return alpha_nu_val * delta_s_val

nu = 20000000 #fréquence en laquelle est calculée l'intégrande

def ratio2(tau):
    if tau > 0.000001:
        return 1 - np.exp(-tau)
    else: 
        return tau



def integrand2(T,teta,phi):
    #calcul d"un premier set de variables 
    global mu_val

    mu_val = mu(teta, phi)
    radius_val = radius(T, mu_val) # Rayon 
    tval = T + mu(teta, phi)*radius_val/const.celer 
    bsh_val = shock_speed(En, T, teta, phi)
    gammash_val = np.sqrt(1/(1   -   bsh_val**2))
    zeta_val = sol(gammash_val)
    beta_val = beta(En,tval, teta,phi)
    gamma_majval = gamma_maj(zeta_val)
    nu_p_val = nu*gamma_majval*(1-beta_val*mu_val)
    p_val = p(En,tval,teta,phi)
    x_val = nu_p_val/nu_prime_m(gammash_val, p_val)


    #Calcul des premiers éléments de l'intégrande
    energ_normalized = Emission_function(x_val,p_val,tval, gammash_val)
    energ_val = energ_normalized*fact_denorm(p_val,gammash_val)
    
    #Premier rapport angulaire de l'intégrande
    angular_sh = np.abs(mu_val - bsh_val)*np.power(radius_val,2) / (1 - mu_val*bsh_val)*np.power(gamma_majval*(1-beta_val*mu_val),3)
    
    # Calcul des beta
    beta_d_val = beta_d(En, tval, teta, phi)
    beta_u_val = beta_u(En, tval, teta, phi)

    # Calcul des gamma prime/ nu prime
    gamma_prime_c_val = gamma_prime_c(gammash_val, tval)
    gamma_prime_m_val = gamma_prime_m(gammash_val, p_val)
    nu_prime_m_val = nu_prime_m(gammash_val, p_val)
    nu_prime_c_val = nu_prime_c( tval, gammash_val)

    # Calcul des autres variables
    b_prime_val = B_prime(gammash_val)
    n_prime_r_val = n_prime_r(gammash_val, p_val)
    nu_prime_val = nu_prime(nu, zeta_val, bsh_val, mu_val) 
    
    # Calcul des autres variables
    p_tildt_val = p_tildt(p_val, gamma_prime_m_val, gamma_prime_c_val, beta_u_val, beta_d_val)
    nu_prime_zero_val = nu_prime_zero(nu_prime_c_val, nu_prime_m_val, p_tildt_val)
    
    # Calcul de alpha_prime_nu_prime
    c_11_val = c_11(nu_prime_val, n_prime_r_val, b_prime_val, nu_prime_c_val, nu_prime_m_val, gamma_prime_m_val, gamma_prime_c_val, p_tildt_val)
    c_14_val = c_14(nu_prime_val, n_prime_r_val, b_prime_val, nu_prime_c_val, nu_prime_m_val, gamma_prime_m_val, gamma_prime_c_val, p_tildt_val)
    alpha_prime_nu_prime_val = alpha_prime_nu_prime(nu_prime_val, p_tildt_val, nu_prime_zero_val, c_11_val, c_14_val)
    
    #Calcul des derniers ratios du calcul de l'intégrande
    alpha_nu_val = alpha_nu(mu_val, bsh_val, beta_val, alpha_prime_nu_prime_val)
    delta_s_val = delta_s(bsh_val, radius_val,  mu_val)
    tau_nu_val = tan_nu(alpha_nu_val, delta_s_val) # Ici tau est parfois calculé avec un développement limité

    #Calcul des derniers ratio du calcul de l'intégrande 
    ratio_1 = energ_val/alpha_prime_nu_prime_val 
    ratio_2_val = ratio2(tau_nu_val)
    

    # Print de différentes variables pour debugger 
    print(f'''
    nu_prime_val: {nu_prime_val}
    p_tildt_val: {p_tildt_val}
    nu_prime_zero_val: {nu_prime_zero_val}
    nu_prime_val, n_prime_r_val, b_prime_val, nu_prime_c_val, nu_prime_m_val, gamma_prime_m_val, gamma_prime_c_val, p_tildt_val: {nu_prime_val}, {n_prime_r_val}, {b_prime_val}, {nu_prime_c_val}, {nu_prime_m_val}, {gamma_prime_m_val}, {gamma_prime_c_val}, {p_tildt_val}
    mu: {mu_val}
    radius: {radius_val}
    tval: {tval}
    bsh: {bsh_val}
    gammash: {gammash_val}
    zeta: {zeta_val}
    beta: {beta_val}
    gamma_maj: {gamma_majval}
    nu_p: {nu_p_val}
    p: {p_val}
    x: {x_val}
    energ: {energ_val}
    angular: {angular_sh}
    beta_d: {beta_d_val}
    beta_u: {beta_u_val}
    gamma_prime_c: {gamma_prime_c_val}
    gamma_prime_m: {gamma_prime_m_val}
    nu_prime_m: {nu_prime_m_val}
    nu_prime_c: {nu_prime_c_val}
    b_prime: {b_prime_val}
    n_prime_r: {n_prime_r_val}
    nu_prime: {nu_prime_val}
    p_tildt: {p_tildt_val}
    nu_prime_zero: {nu_prime_zero_val}
    c_11: {c_11_val}
    c_14: {c_14_val}
    alpha_prime_nu_prime: {alpha_prime_nu_prime_val}
    alpha_nu: {alpha_nu_val}
    delta_s: {delta_s_val}
    tau_nu: {tau_nu_val}
    ratio_1: {ratio_1}
    ratio_2: {ratio_2_val}
    sol(gammash): {sol(gammash_val)}
    ''')
    print('energ_normlized', energ_normalized)
    print('fact denorm', fact_denorm(p_val,gammash_val))
    D_mp = 40 
    D_m = D_mp * 3.086 * 10 ** 22
    return angular_sh * ratio_1 * ratio_2_val / ( 4 * const.pi * D_m ** 2)

print(f'''calcul de l'intégrande''', integrand2(200,20*3.14/180,20*3.14/180))

#Plot de différents variables en fonction de T
'''    
log_T = np.linspace(-1, 3, 30)
T_vals = 10**log_T
nu = 20000000 #fréquence en laquelle est calculée l'intégrande

teta = 20*3.14/180
phi = 20*3.14/180
Y1 = []
Y2 = []
Y3 = []
Y4 = []
Y5 = []
Y6 = []

#calcul d"un premier set de variables
for T in T_vals:
    mu_val = mu(teta, phi) 
    radius_val = radius(T,teta,phi)
    tval = T + mu(teta, phi)*radius_val/const.celer
    bsh_val = shock_speed(En,T,tval,phi)
    gammash_val = np.sqrt(1/(1   -   bsh_val**2))
    zeta_val = sol(gammash_val)
    beta_val = beta(En,tval, teta,phi)
    gamma_majval = gamma_maj(zeta_val)
    nu_p_val = nu*gamma_majval*(1-beta_val*mu_val)
    p_val = p(En,tval,teta,phi)
    x_val = nu_p_val/nu_prime_m(gammash_val, p_val)
    #Calcul des premiers éléments de l'intégrande
    energ_normalized = Emission_function(x_val,p_val,tval, gammash_val)
    energ_val = energ_normalized*fact_denorm(p_val,gammash_val)
    #Premier rapport angulaire de l'intégrande
    angular_sh = np.abs(mu_val - bsh_val)*np.power(radius_val,2) / (1 - mu_val*bsh_val)*np.power(gamma_majval*(1-beta_val*mu_val),3)

    beta_d_val = beta_d(En, tval, teta, phi)
    beta_u_val = beta_u(En, tval, teta, phi)
    gamma_prime_c_val = gamma_prime_c(gammash_val, tval)
    gamma_prime_m_val = gamma_prime_m(gammash_val, p_val)
    nu_prime_m_val = nu_prime_m(gammash_val, p_val)
    nu_prime_c_val = nu_prime_c( tval, gammash_val)
    b_prime_val = B_prime(gammash_val)
    n_prime_r_val = n_prime_r(gammash_val, p_val)
    nu_prime_val = nu_prime(nu, zeta_val, bsh_val, mu_val) 
    p_tildt_val = p_tildt(p_val, gamma_prime_m_val, gamma_prime_c_val, beta_u_val, beta_d_val)
    nu_prime_zero_val = nu_prime_zero(nu_prime_c_val, nu_prime_m_val, p_tildt_val)
    c_11_val = c_11(nu_prime_val, n_prime_r_val, b_prime_val, nu_prime_c_val, nu_prime_m_val, gamma_prime_m_val, gamma_prime_c_val, p_tildt_val)
    c_14_val = c_14(nu_prime_val, n_prime_r_val, b_prime_val, nu_prime_c_val, nu_prime_m_val, gamma_prime_m_val, gamma_prime_c_val, p_tildt_val)
    alpha_prime_nu_prime_val = alpha_prime_nu_prime(nu_prime_val, p_tildt_val, nu_prime_zero_val, c_11_val, c_14_val)
    alpha_nu_val = alpha_nu(mu_val, bsh_val, beta_val, alpha_prime_nu_prime_val)
    delta_s_val = delta_s(bsh_val, radius_val,  mu_val)
    tau_nu_val = tan_nu(alpha_nu_val, delta_s_val)
    ratio_1 = energ_val/alpha_prime_nu_prime_val 
    ratio_2_val = ratio2(tau_nu_val)

    Y1.append(nu_prime_val)
    Y2.append(p_tildt_val)
    Y3.append(nu_prime_zero_val)
    Y4.append(c_11_val)
    Y5.append(c_14_val)
    Y6.append(alpha_prime_nu_prime_val)
    #Y6.append(angular_sh*ratio_1*ratio_2_val)
    #print(Y6)
    #logY6 = np.log10(Y6)
    print(' done', T)
fig, axs = plt.subplots(3, 2, figsize=(12, 8))
axs[0, 0].plot(T_vals, Y1)
axs[0, 0].set_title('nu_prime')
axs[0, 0].set_xscale('log')
axs[0, 1].plot(T_vals, Y2, 'tab:orange')
axs[0, 1].set_title('p_tildt')
axs[0, 1].set_xscale('log')
axs[1, 0].plot(T_vals, Y3, 'tab:green')
axs[1, 0].set_title('nu_prime_zero')
axs[1, 0].set_xscale('log')
axs[1, 1].plot(T_vals, Y4, 'tab:red')
axs[1, 1].set_title('c_11')
axs[1, 1].set_xscale('log')
axs[2, 0].plot(T_vals, Y5, 'tab:red')   
axs[2, 0].set_title('c_14')
axs[2, 0].set_xscale('log')
axs[2, 1].plot(T_vals, Y6, 'tab:red')

axs[2, 1].set_title('alpha_prime_nu_prime')
axs[2, 1].set_xscale('log')

plt.show()    
'''
