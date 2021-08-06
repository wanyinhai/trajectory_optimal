import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# parameters
N = 50 #number of trajectory segments

# initialize NLP
opti = ca.Opti()

# decision variables
X = opti.variable(N+1)
Y = opti.variable(N+1)
Theta = opti.variable(N+1)
V = opti.variable(N+1)
Phi = opti.variable(N+1)

A = opti.variable(N+1)
Omga = opti.variable(N+1)
tf = opti.variable()
# time discretization
# t = np.linspace(0, tf, N+1)
dt = tf/N

# boundary constraints
opti.subject_to( X[0] == 0 )
opti.subject_to( X[N] ==5 )
opti.subject_to( Y[0] == 0 )
opti.subject_to( Y[N] == -5 )
opti.subject_to( Theta[0] == 0 )
opti.subject_to( Theta[N] == 0)
opti.subject_to( V[0] == 0 )
opti.subject_to( V[N] ==0 )
opti.subject_to( Phi[0] == 0 )
opti.subject_to( Phi[N] == 0 )

opti.subject_to( A[0] == 0 )
opti.subject_to( A[N] ==0 )
opti.subject_to( Omga[0] == 0 )
opti.subject_to( Omga[N] == 0 )

opti.subject_to(opti.bounded(-2, V, 2))   # track speed limit
opti.subject_to(opti.bounded(-0.72, Phi, 0.72))   # steering angle limit
opti.subject_to(opti.bounded(-0.3, A, 0.3)) 
opti.subject_to(opti.bounded(-0.54, Omga, 0.54))
opti.subject_to(tf>=0)
# variable for storing the cost function
J = 0
L = 2.8
# iterate over all k = 0,..., N-1. Note that we don't include index N
for i in range(N):
    x_left = X[i]; x_right = X[i+1]
    y_left = Y[i]; y_right = Y[i+1]
    theta_left = Theta[i]; theta_right = Theta[i+1]
    v_left = V[i]; v_right = V[i+1]
    phi_left = Phi[i]; phi_right = Phi[i+1]
    a_left = A[i]; a_right = A[i+1]
    omga_left = Omga[i]; omga_right = Omga[i+1]    
    # collocation constraints
    opti.subject_to( x_right-x_left == 0.5*tf/N*(v_right*np.cos(theta_right)+v_left*np.cos(theta_left)) )
    opti.subject_to( y_right-y_left == 0.5*tf/N*(v_right*np.sin(theta_right)+v_left*np.sin(theta_left)) )
    opti.subject_to( theta_right-theta_left == 0.5*tf/N*(v_right*np.tan(phi_right)/L+v_left*np.tan(phi_left)/L ))
    opti.subject_to( v_right-v_left == 0.5*tf/N*(a_right+a_left ))
    opti.subject_to( phi_right-phi_left == 0.5*tf/N*(omga_right+omga_left ))
    # cost function
    J += 0.5*tf/N*(omga_left**2 + omga_right**2)

J = J+10*tf
# apply cost function to opti
opti.minimize(J)

# initial guess
opti.set_initial(X, np.full(X.shape, 0))
opti.set_initial(Y, np.full(Y.shape, 0))
# opti.set_initial(X,np.linspace(0, 5, N+1))
# opti.set_initial(Y, np.linspace(0, 5, N+1))
opti.set_initial(Theta, np.full(Theta.shape, 0))
opti.set_initial(V, np.full(V.shape, 0))
opti.set_initial(Phi, np.full(Phi.shape, 0))
opti.set_initial(A, np.full(A.shape, 0))
opti.set_initial(Omga, np.full(Omga.shape, 0))
opti.set_initial(tf, 1)
# solve
opti.solver('ipopt')
sol = opti.solve()

# extract solution
sol_t = tf
sol_x = sol.value(X)
sol_y = sol.value(Y)
sol_theta = sol.value(Theta)
sol_v = sol.value(V)
sol_phi = sol.value(Phi)
sol_a = sol.value(A)
sol_omga = sol.value(Omga)

# t = np.linspace(0, sol_t, N+1)
# plot
# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax[0].plot(sol_x, sol_y, 'o')
# ax[0].set_title('Position')
# plt.show()

plt.figure(figsize=(4,3))
plt.plot(sol_x,sol_y,'o')
plt.show()
