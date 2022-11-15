import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.sparse
import time

'''------Problem 1-------'''

L = 10   #define initial conditions
xpoints = np.linspace(-L, L, 200, endpoint=False)
tpoints = np.arange(0, 10+0.5, 0.5)
dx = 0.1

diagindex = [-199, -1, 1, 199]   #define diagonals
longdiags = np.ones(200) / (2 * dx)
data = np.array([longdiags, longdiags * (-1), longdiags, longdiags * (-1)])

Amatrix = scipy.sparse.spdiags(data, diagindex, 200, 200, format='csc')   #create the dense matrix A
A1 = Amatrix.todense()

def advectionPDE(t, u, A, c):   #define ODE
   u_t = (-c) * A @ u
   return u_t

f = lambda x: np.exp(-((x - 5) ** 2))   #define initial condition
c = -0.5
u0 = f(xpoints)

sol1b = scipy.integrate.solve_ivp(lambda t, u: advectionPDE(t, u, Amatrix, c), [0, 10], u0, t_eval=tpoints)   #solve ODE
A2 = sol1b.y   #get the u values of the solution

'''fig1 = plt.figure()    #plot the surface
ax1 = plt.axes(projection='3d')
T, X = np.meshgrid(sol1b.t, xpoints)
surf = ax1.plot_surface(T, X, A2, cmap='magma')
plt.show()'''

cfunc = lambda t, x: 1 + 2 * np.sin(5 * t) - np.heaviside(x - 4, 0)   #define c(x,t)

def newadvectionPDE(t, u, A, cfunc, xpoints):   #redefine the ODE
   u_t = cfunc(t, xpoints) * A@u
   return u_t

sol1c = scipy.integrate.solve_ivp(lambda t, u: newadvectionPDE(t, u, Amatrix, cfunc, xpoints), [0, 10], u0, t_eval=tpoints)   #solve ODE
A3 = sol1c.y
#print(A3)

'''fig2 = plt.figure()    #plot the surface
ax2 = plt.axes(projection='3d')
T, X = np.meshgrid(sol1c.t, xpoints)
surf = ax2.plot_surface(T, X, A3, cmap='magma')
plt.show()'''

'''The solution for part c gives very small numbers and I am not sure why.'''

'''------Problem 2------'''
v = 0.001
L = 10
tvals = np.arange(0, 4+0.5, 0.5)
xyvals = np.linspace(-L, L, 64, endpoint=False)

f = lambda x, y: np.exp((-2 * x * x) - (y * y / 20))
X, Y = np.meshgrid(xyvals, xyvals)
omega0matrix = f(X, Y)
omega0 = np.array([np.reshape(omega0matrix, 64*64)])

'''Create the A Matrix'''
m = 64 # N value in x and y directions
n = m*m # total size of matrix

e1 = np.ones(n) # vector of ones
e2 = np.ones(n) * (-4)
e2[0] = 2
Low1 = np.tile(np.concatenate((np.ones(m-1), [0])), (m,)) # Lower diagonal 1
Low2 = np.tile(np.concatenate(([1], np.zeros(m-1))), (m,)) #Lower diagonal 2
                                    # Low2 is NOT on the second lower diagonal,
                                    # it is just the next lower diagonal we see
                                    # in the matrix.

Up1 = np.roll(Low1, 1) # Shift the array for spdiags
Up2 = np.roll(Low2, m-1) # Shift the other array

A = scipy.sparse.spdiags([e1, e1, Low2, Low1, e2, Up1, Up2, e1, e1],
                         [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)], n, n)
A4 = A.todense()

'''Create the B Matrix'''
dx = xyvals[1] - xyvals[0]

diagindex = [-(64*64 - 1 - 64), -64, 64, 64*64 - 1 - 64]
longdiags = np.ones(n) / (2 * dx)
data = np.array([longdiags, longdiags * (-1), longdiags, longdiags * (-1)])

B = scipy.sparse.spdiags(data, diagindex, 4096, 4096, format='csc')   #create the dense matrix A
A5 = B.todense()

'''Create the C Matrix'''
dy = xyvals[1] - xyvals[0]

diagindex = [-63, -1, 1, 63]
longdiags = np.ones(n) / (2 * dy)
otherdiags = np.tile(np.concatenate((np.zeros(m-1), [1])), (m,))
data = np.array([otherdiags, longdiags * (-1), longdiags, otherdiags * (-1)])

C = scipy.sparse.spdiags(data, diagindex, 4096, 4096, format='csc')   #create the dense matrix A
A6 = B.todense()

plt.spy(C)
plt.show()

'''Discretizations of both Equations'''
