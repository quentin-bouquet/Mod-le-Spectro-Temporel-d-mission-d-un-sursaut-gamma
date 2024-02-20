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
    if a < b:
        return np.power(a / b, 1/3)
    elif b <= a <= c:
        return np.power(a / b, -(p-1)/2)
    elif c <= a:
        return np.power(c / b, -(p-1)/2) * np.power(a / c, -p/2)
    
def choix_2(a, b, c, p):
    if a < b:
        return np.power(a / b, 1/3)
    elif b <= a <= c:
        return np.power(a / b, -1/2)
    elif c <= a:
        return np.power(c / b, -1/2) * np.power(a / c, -p/2)

