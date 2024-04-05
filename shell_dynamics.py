import numpy as np

c=2.9792458e10#cm s-1
mev2erg = 1.60218e-6
mpc2 = 938.21998*mev2erg # erg
pi = np.pi

#t est un temps en seconde, suivant l'évolution spatiale du choc dans le référentiel du laboratoire
# x = E/n0 est en erg*cm3
# Donc E/(n0*mp*c5*t^3) demande que mp*c2 soit exprimé en erg, et c^3 en cm3/s^3
u = mpc2 * c**3
def U(t, x) :
    return x * t**(-3) / u

#Ensuite on précalcule les facteurs de C_BM^2 et C_ST^2
#Note qu'on calcule directement les carrés de eqs 2 et 3 à cause de l'eq.1
bm = 17/8/pi
st = (2*1.15/5)**2

#terme C_BM^2 * t^(-3)
def BM(t, x):
    return bm*U(t, x)

#terme C_ST^2 * t^(-6/5)
def ST(t, x):
    return st*U(t, x)**(2/5)

#1er terme du minimum dans l'eq.1 :
def A(t, x):
    return BM(t, x) + ST(t, x)

#Gamma_sh^2 * beta_sh^2 :
def GB2(t, x, Gshmax) :
    val = A(t, x)
    if np.isscalar(val):
        return min(val, Gshmax)
    val[val>=Gshmax**2] = Gshmax**2
    return val

#Gamma_sh^2 = Gamma_sh^2 * beta_sh^2 +1
def G2(t, x, Gshmax):
    return GB2(t, x, Gshmax)+1

#beta_sh^2 = Gamma_sh^2 * beta_sh^2 / (Gamma_sh^2 * beta_sh^2 + 1)
def B2(t, x, Gshmax):
    return GB2(t,x, Gshmax) / (GB2(t, x, Gshmax)+1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    day2sec=86400
    Gshmax=100
    E  = 10**50 #ergs
    n0 = 1e-3  #cm-3
    x = E/n0
    tarr = np.logspace(1,4,50) # 10^-2 à 10^2 jours
    #on convertit les temps en secondes quand on les passe aux fonctions
    plt.loglog(tarr, G2(tarr*day2sec, x, Gshmax), label=r'$\Gamma_{sh}^2$')
    plt.loglog(tarr, B2(tarr*day2sec, x, Gshmax), label=r'$\beta_{sh}^2$')
    plt.loglog(tarr, GB2(tarr*day2sec, x, Gshmax), label=r'$\Gamma_{sh}^2\beta_{sh}^2$')
    plt.loglog(tarr, BM(tarr*day2sec, x), label='BM')
    plt.loglog(tarr, ST(tarr*day2sec, x), label='ST')
    plt.xlabel("t[day]", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(r'$\Gamma_{sh}^2\beta_{sh}^2$', fontsize=16)
    plt.legend()
    plt.tight_layout()
