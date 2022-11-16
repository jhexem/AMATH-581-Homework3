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

'''fig = plt.figure()    #plot the surface for Problem 1b
ax = plt.axes(projection='3d')
T, X = np.meshgrid(sol1b.t, xpoints)
surf = ax.plot_surface(T, X, A2, cmap='magma')
plt.show()'''

cfunc = lambda t: 1 + 2 * np.sin(5 * t) - np.heaviside(xpoints - 4, 0)   #define c(x,t)

def newadvectionPDE(t, u, A, cfunc):   #redefine the ODE
   u_t = cfunc(t) * (A@u)
   return u_t

sol1c = scipy.integrate.solve_ivp(lambda t, u: newadvectionPDE(t, u, Amatrix, cfunc), [0, 10], u0, t_eval=tpoints)   #solve ODE
A3 = sol1c.y

'''fig = plt.figure()    #plot the surface for Problem 1c
ax = plt.axes(projection='3d')
X, T = np.meshgrid(xpoints, sol1c.t)
surf = ax.plot_surface(X, T, A3.T, cmap='magma')
plt.show()'''

'''------Problem 2------'''
v = 0.001   #initial comditions
m = 64
L = 10
tvals = np.arange(0, 4+0.5, 0.5)
xyvals = np.linspace(-L, L, m, endpoint=False)
delta = xyvals[1] - xyvals[0]

f = lambda x, y: np.exp((-2 * x * x) - (y * y / 20))   #define omega0, the initial condiiton vector for omega
X, Y = np.meshgrid(xyvals, xyvals)
omega0matrix = f(X, Y)
omega0 = np.reshape(omega0matrix, m*m)

'''Create the A Matrix'''
def Afunc(m, xyvals):
   n = m*m # total size of matrix
   delta = xyvals[1] - xyvals[0]
   e1 = np.ones(n) / (delta**2) # vector of ones
   e2 = np.ones(n) * (-4) / (delta**2)
   e2[0] = 2
   Low1 = np.tile(np.concatenate((np.ones(m-1), [0])), (m,)) / (delta**2) # Lower diagonal 1
   Low2 = np.tile(np.concatenate(([1], np.zeros(m-1))), (m,)) / (delta**2) #Lower diagonal 2
                                       # Low2 is NOT on the second lower diagonal,
                                       # it is just the next lower diagonal we see
                                       # in the matrix.

   Up1 = np.roll(Low1, 1) # Shift the array for spdiags
   Up2 = np.roll(Low2, m-1) # Shift the other array

   A = scipy.sparse.spdiags([e1, e1, Low2, Low1, e2, Up1, Up2, e1, e1],
                           [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)], n, n, format='csc')
   return A

A = Afunc(m, xyvals)   #call the function that returns A
A4 = A.todense()

'''Create the B Matrix'''
def Bfunc(m, xyvals):
   n = m*m # total size of matrix
   dx = xyvals[1] - xyvals[0]

   diagindex = [-(n - m), -m, m, n - m]   #choose the indices for the diagonals to go
   longdiags = np.ones(n) / (2 * dx)   #create arrays for the diagonals and divide by 2 * dx
   data = np.array([longdiags, longdiags * (-1), longdiags, longdiags * (-1)])   #form the array that holds all values for each of the diagonals

   B = scipy.sparse.spdiags(data, diagindex, n, n, format='csc')   #create the dense matrix B
   return B

B = Bfunc(m, xyvals)   #call the function that returns B
A5 = B.todense()

'''Create the C Matrix'''
def Cfunc(m, xyvals):
   n = m*m # total size of matrix
   dy = xyvals[1] - xyvals[0]

   diagindex = [-(m-1), -1, 1, m-1]   #choose the indices for the diagonals to go
   longdiags = np.ones(n) / (2 * dy)   #create arrays for the diagonals and divide by 2 * dy
   otherdiag1 = np.tile(np.concatenate((np.zeros(m-1), [1])), (m,)) / (2 * dy)   #create the diagonals with all zeros except one 1
   otherdiag2 = np.roll(otherdiag1, 1)   #use the roll function to correctly place the 1's for the lower diagonal
   data = np.array([otherdiag2, longdiags * (-1), longdiags, otherdiag1 * (-1)])   #form the array that holds all values for each of the diagonals

   C = scipy.sparse.spdiags(data, diagindex, n, n, format='csc')   #create the dense matrix C
   return C

C = Cfunc(m, xyvals)   #call the function that returns C
A6 = B.todense()

'''Testing plots for the Laplacian Matrix to fix errors'''
testfunc = lambda x, y: np.sin(x) + np.cos(y)   #defines the function sin(x) + cos(y)
testfunc2 = lambda x, y: -np.sin(x) - np.cos(y)   #defines the actual Laplacian of testfunc

X, Y = np.meshgrid(xyvals, xyvals)

newfunc = A@np.reshape(testfunc(X, Y), m*m)   #My Laplacian matrix A applied to sin(x) + cos(y)
newfunc2 = np.reshape(testfunc2(X, Y), m*m)   #the actual Laplacian of sin(x) + cos(y)

'''fig = plt.figure()    #plot the surface for either my calculated Laplacian or the actual Laplacian
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, np.reshape(newfunc, (m, m)), cmap='magma')   #change newfunc to newfunc2 to check the difference between my output and the actual Laplacian
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
plt.show()'''

'''Discretizations of both Equations for Gaussian Elimination'''
def GEdiscretized2(t, omega, A):   #solve for psi using equation 2
   psi = scipy.sparse.linalg.spsolve(A, omega)
   return psi

def GEdiscretized1(t, omega, GEdiscretized2, A, B, C, v):   #solve for omegat using equation 1
   psi = GEdiscretized2(t, omega, A)   #get psi value
   omegat = np.multiply(C@psi, B@omega) - np.multiply(B@psi, C@omega) + v * (A@omega)   #gets the value of omegat by implimenting the ODE
   return omegat

def GEsolve(v, tvals, omega0, A, B, C):  #solve the ODE for omegat
   sol = scipy.integrate.solve_ivp(lambda t, omega: GEdiscretized1(t, omega, GEdiscretized2, A, B, C, v), [0, 4], omega0, t_eval=tvals)
   return sol

solGE = GEsolve(v, tvals, omega0, A, B, C)   #call the function that used Gaussian Elimination to solve the ODE
y_solGE = solGE.y.T   #variable for the solutions (transposed to make unstacking the solution easier)

A7 = y_solGE   #the above is commented out because the incorrect A matrix was creating an infinite loop in the solver

'''for i in range(9):   #loop through all 9 solution vectors to unstack them and make a comtour plot of each one
   unstacked = np.reshape(y_solGE[i], (m, m))   #unstack the solution vector at time i

   fig, ax = plt.subplots(1, 1)   #makes a contour plot of the matrix solution for Gaussian Elimination
   X, Y = np.meshgrid(xyvals, xyvals)
   ax.contourf(X, Y, unstacked)
   plt.show()'''

'''Discretizations of both Equations for LU Decomposition'''
def LUdiscretized2(t, omega, LUdecomp):   #solve for psi using equation 2
   psi = LUdecomp.solve(omega)
   return psi

def LUdiscretized1(t, omega, LUdiscretized2, A, B, C, v, LUdecomp):   #solve for omegat using equation 1
   psi = LUdiscretized2(t, omega, LUdecomp)   #get psi value
   omegat = np.multiply(C@psi, B@omega) - np.multiply(B@psi, C@omega) + v * (A@omega)   #gets the value of omegat by implimenting the ODE
   return omegat

def LUsolve(v, tvals, omega0, A, B, C, LUdecomp):  #solve the ODE for omegat
   sol = scipy.integrate.solve_ivp(lambda t, omega: LUdiscretized1(t, omega, LUdiscretized2, A, B, C, v, LUdecomp), [0, 4], omega0, t_eval=tvals)
   return sol

LUdecomp = scipy.sparse.linalg.splu(A)   #generates the LU decomposition
solLU = LUsolve(v, tvals, omega0, A, B, C, LUdecomp)   #call the function that solves the ODE
y_solLU = solLU.y.T   #variable for the solutions (transposed to make unstacking the solution easier)

A8 = y_solLU   #the above is commented out because the incorrect A matrix was creating an infinite loop in the solver
A9 = np.reshape(y_solLU, (9, m, m))

'''for i in range(9):   #loop through all 9 solution vectors to unstack them and make a comtour plot of each one
   fig, ax = plt.subplots(1, 1)   #makes a contour plot of the matrix solution for Gaussian Elimination
   X, Y = np.meshgrid(xyvals, xyvals)
   ax.contourf(X, Y, A9[i, :, :])
   plt.show()'''