# -----------------------------------------------
# pendulum equation:
#
#
#------------------------------------------------
import numpy as np

def pendulum(y,t,*args):
    k = args[0]
    g = args[1]
    L = args[2]
    m = args[3]
    theta, omega = y
    dydt = [omega/m, -(g/L)*(np.sin(theta))+k*omega/m]
    return dydt
