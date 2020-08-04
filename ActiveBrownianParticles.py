# Active matter simulations using 
# "Numerical Simulations of Active Brownian Particles" 
# https://link.springer.com/chapter/10.1007/978-3-030-23370-9_7 (Paper 1)

# Notation, for the convenience of future analyses, from 
# "Minimal model of active colloids highlights the role of mechanical 
# interactions in controlling the emergent behavior of active matter"
# https://www.sciencedirect.com/science/article/abs/pii/S1359029416300024 (Paper 2)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
from scipy import constants


dt = 0.01
T = 1000
iterations = int(T//dt)

N = 1                                                         # particles number

R = 3*10**(-7)                                                  # particles radius
eta = 3*10**(-17)                                               # fluid viscosity
k_B = constants.Boltzmann                                       # Boltzmann constant
temp = 10.                                                      # absolute temperature

l = 5                                                         # arena side
A = l**2                                                        # arena area
rho = N/A                                                       # particles density

# arena particle boundaries
x_min = - l/2 + R
x_max = + l/2 - R
y_min = - l/2 + R
y_max = + l/2 - R

gamma_t = 6 * constants.pi * eta * R                            # translational friction coefficient
gamma_r = 8 * constants.pi * eta * R**3                         # rotational friction coefficient

mu = 1.                                                         # mobility
D_t = k_B*temp/gamma_t                                          # translation diffusion
# D_r = k_B*temp/gamma_r                                          # rotation diffusion
# tau_r = 1/D_r                                                   # persistence time
v_0 = 0.01                                                      # self-propulsion speed
# l_p = v_0*tau_r                                                 # persistence length
a = 1.                                                          # force parameter
k = 1.                                                          # force parameter
# Pe_r = l_p/a                                                    # rotational Péclet number (in Paper 2, D_r is determined for a given Pe_r with fixed a and v_0)
Pe_r = 150.
D_r = v_0/(Pe_r*a)


x = l*np.random.rand(iterations, N, 2) - l/2                    # 2D position (x, y)
theta = 2*constants.pi*np.random.rand(iterations, N)            # angle
F = np.zeros((iterations, N, 2))                                # soft repulsive forces
r = np.zeros((iterations, N, 2))                                # unit vector for soft repulsive forces
a_particles = a*np.ones((N, 1))                                 # force parameter vector

W_x = np.sqrt(2 * D_t) * np.random.randn(iterations, N, 2)      # Wiener noise on translation
W_theta = np.sqrt(2 * D_t) * np.random.randn(iterations, N)     # Wiener noise on rotation


length_orientation = .5

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(111, aspect='equal', autoscale_on=True, xlim=(- l/2, l/2), ylim=(- l/2, l/2))
ax1.set_xticks(np.arange(- l/2, l/2, l/10))
ax1.set_yticks(np.arange(- l/2, l/2, l/10))
ax1.grid()
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

particles, = ax1.plot([], [], 'o', lw=2, label='Particle')
particles_orientation, = ax1.plot([], [], '-', lw=2)


def init():
    """initialize animation"""
    time_text.set_text('')

    particles.set_data([], [])
    particles_orientation.set_data([], [])

    return particles, particles_orientation, time_text

def animate(i):
    for j in range(N):
        distance_ij = np.linalg.norm(x[i,j,:] - x[i,:,:], axis=1)
        # distance_ij = scipy.spatial.distance.cdist(x[i,j,:], x[i,:,:])
        direction_ij = (x[i,j,:] - x[i,:,:]) / distance_ij[:, None]
        f_ij = k * (a_particles[j,0] + a_particles[:,0] - distance_ij)
        f_ij[f_ij < 0] = 0

        F[i,j,:] = np.nansum(f_ij[:, None] * direction_ij, axis=0)




    """perform animation step"""
    dx = v_0 * np.cos(theta[i,:]) + F[i,:,0] + W_x[i,:,0]/np.sqrt(dt)
    dy = v_0 * np.sin(theta[i,:]) + F[i,:,1] + W_x[i,:,1]/np.sqrt(dt)
    dtheta = W_theta[i,:]/np.sqrt(dt)

    x[i+1,:,:] = x[i,:,:] + dt * np.array([dx, dy]).T
    theta[i+1,:] = theta[i,:] + dt * dtheta

    # particles bouncing off the walls
    theta[i+1,x[i+1,:,0]<x_min] = constants.pi - theta[i+1,x[i+1,:,0]<x_min]
    theta[i+1,x[i+1,:,0]>x_max] = constants.pi - theta[i+1,x[i+1,:,0]>x_max]

    theta[i+1,x[i+1,:,1]<y_min] = 2*constants.pi - theta[i+1,x[i+1,:,1]<y_min]
    theta[i+1,x[i+1,:,1]>y_max] = 2*constants.pi - theta[i+1,x[i+1,:,1]>y_max]


    time_text.set_text('iteration = %.2f' % i)

    particles.set_data([x[i,:,0]], [x[i,:,1]])
    # print([x[i,:,0], x[i,:,0]+length_orientation*np.cos(theta[i,:,0])])
    particles_orientation.set_data([x[i,:,0], x[i,:,0]+length_orientation*np.cos(theta[i,:])], [x[i,:,1], x[i,:,1]+length_orientation*np.sin(theta[i,:])])

    return particles, particles_orientation, time_text

# plt.plot(x[:,:,0], x[:,:,1])

ani = animation.FuncAnimation(fig, animate, frames=iterations,
                              interval=1, blit=True, init_func=init)


phi = np.pi * np.sum(a_particles**2) / A
print(phi)
print(D_r)
plt.show()


