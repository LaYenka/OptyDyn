import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from ODE_int import *
from system_pendulum import pendulum


N = 100         # in how much sub pieces we should break a 1sec interval
T = 15          # total duration of the simulation
dt = 1 / N      # dt
g = 9.81        # acceleration of gravity
L = 1           # pendulum rope length
k = 0.8         # air resistance coefficient
m = 1           # mass of the pendulum

#initial condition
theta_0 = [0.2*np.pi / 2,0.1]      # initial angle ,theta_dot]

# time plot
t = np.linspace(0,T,N,dtype=float)
# ode_int python
theta_odeint = odeint(pendulum,theta_0,t,args = (k,g,L,m))

# setup tuple with argument list
args_sys = (k,g,L,m)
# import class to solve ODE
theta_ode = ode_int()
theta_nm, d_theta_nm = theta_ode.rk4(pendulum,theta_0,t,args_sys)
theta_nm1, d_theta_nm = theta_ode.euler(pendulum,theta_0,t,args_sys)
#For animation creating plots
# f=1
#for i in range(0,240):
#	filename = str(f)+".png"
#	f= f+1
#	plt.figure()
#	plt.plot([10,l*math.sin(theta[i,0])+10],[10,10-l*math.cos(theta[i,0])],marker=\"o\")
#	plt.xlim([0,20])
#	plt.ylim([0,20])
#	plt.savefig(filename)
fig, ax3 = plt.subplots()
ax3.plot(t,theta_odeint[:,0],'b-',label='theta ref')
ax3.plot(t,theta_nm[:,0],'bo-', label='theta - RK4')
ax3.plot(t,theta_odeint[:,1],'r--',label='theta dot - ref')
ax3.plot(t,theta_nm[:,1],'ro-',label='theta dot - RK4')
ax3.set_title('comparison reference solution')
ax3.legend()

fig, ax=plt.subplots()
ax.plot(t,theta_odeint[:,0]-theta_nm[:,0],label='Delta theta - RK4/ref')
ax.plot(t,theta_odeint[:,0]-theta_nm1[:,0],'k',label='Delta theta - Euler/ref')
ax.set_title('absolute error')
ax.legend()

# phase plane
fig1, ax1=plt.subplots()
ax1.plot(theta_nm[:-1,0], d_theta_nm[:-1,0],label='phase plot')
plt.xlabel("x")
plt.ylabel("dx/dt")
ax1.legend()

plt.show()
