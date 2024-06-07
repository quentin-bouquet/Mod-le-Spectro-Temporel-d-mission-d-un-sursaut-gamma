import numpy as np 
import numba as nb
import scipy.constants as const

def max(a,b): 
    if a > b:
        return a
    else:
        return b

def min(a,b): 
    if a < b:
        return a
    else:
        return b

def choix_1(a, b, c, p):
    x = a / b
    if a < b:
        return np.power(x , 1/3)
    elif b <= a <= c:
        return np.power(x, -(p-1)/2)
    elif c <= a:
        return np.power(c / b, -p/2) * np.power(x, -p/2)
    
def choix_2(a, b, c, p):
    x = a / b 
    if a < b:
        return np.power(x, 1 / 3)
    elif b <= a <= c:
        return np.power(x, -1 / 2)
    elif c <= a:
        return np.power(x, -(p - 1)/2) * np.power(c / b, -p / 2) * np.power(c, 1 / 2)

