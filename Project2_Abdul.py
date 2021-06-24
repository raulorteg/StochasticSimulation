import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.style.use("ggplot")

""" function to solve ODEs """
def ode_rhs(t, init, parms):
    """ ODEs' right hand side:"""
    beta, gamma = parms
    S,I,_ = init
    dsdt = -beta*S*I
    didt =  beta*S*I - gamma*I
    drdt =  gamma*I
    return [dsdt,didt,drdt]

""" Parameters """
beta=1.5; gamma=1.0  
S=10**3-10; I=10; R=0 
dt    = 0.1; T     = 20
times = np.arange(0,T, dt)

""" Solve ODEs """
ode_sol = solve_ivp(fun=lambda t, 
                    y: ode_rhs(t, y, [beta, gamma]), 
                    t_span=[min(times),max(times)], 
                    y0=list(np.array([S,I,R])/(S+I+R)), 
                    t_eval=times)
""" plot """
plt.plot(ode_sol['t'], ode_sol['y'][0], color='b', linewidth=3, label='S - susceptible')
plt.plot(ode_sol['t'], ode_sol['y'][1], color='r', linewidth=2, label='I - infected')
plt.plot(ode_sol['t'], ode_sol['y'][2], color='g', linewidth=2, label='R - recovered')
plt.xlabel('Time', fontweight='bold')
plt.ylabel('Number', fontweight='bold')
plt.title('Population')
plt.legend()

""" print """
print(f"I - infected max =  { np.round( max(ode_sol['y'][1]),3) }")
print(f"R - recovered max = { np.round( max(ode_sol['y'][2]),3) }")