import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.sparse

'''------Problem 1-------'''

L = 10   #define initial conditions
xpoints = np.linspace(-L, L, 200)
tpoints = np.arange(0, 10+0.5, 0.5)
dx = xpoints[1] - xpoints[0]

diagindex = [-199, -1, 1, 199]   #define diagonals
longdiags = np.ones(200)
data = np.array([longdiags, longdiags * (-1), longdiags, longdiags * (-1)])

A = scipy.sparse.spdiags(data, diagindex, 200, 200)   #create the dense matrix A
A1 = A.todense()

def advectionPDE(t, u, A):   #define ODE
   u_t = 0.5 * A @ u
   return u_t

f = lambda x: np.exp(-((x - 5) ** 2))   #define initial condition
u0 = f(xpoints)

sol1b = scipy.integrate.solve_ivp(lambda t, u: advectionPDE(t, u, A), [0, 10], u0, t_eval=tpoints)   #solve ODE
usol1b = sol1b.y * 2 * dx   #get the u values of the solution and multiply by 2 * dx

fig1 = plt.figure()    #plot the surface
ax1 = plt.axes(projection='3d')
T, X = np.meshgrid(sol1b.t, xpoints)
surf = ax1.plot_surface(T, X, usol1b, cmap='magma')
plt.show()


'''------Problem 2------'''
v = 0.001
L = 10
tvals = np.arange(-L, L+0.5, 0.5)
xyvals = np.linspace(-L, L, 64)

