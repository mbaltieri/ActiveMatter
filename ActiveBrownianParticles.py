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
from scipy.spatial.distance import pdist, squareform
import time
from numba import jit
# from numba import int32, float32, float64    # import the types
# from numba.experimental import jitclass

# spec = [
#     ('value', int32),               # a simple scalar field
#     ('array', float32[:]),          # an array field
#     ('array2', float64[:]),          # an array field
# ]

# @jitclass(spec)
class ABPSimulation():
    def __init__(self, dt=0.01, T=1000, l=50, translfric=6, rotatfric=8, mob=1., v_0=0.01, pecnum=11., packfrac=0.5):
        self.dt = dt
        self.T = T
        self.iterations = int(self.T//self.dt)

        self.R = 3*10**(-7)                                                  # particles radius
        self.eta = 3*10**(-17)                                               # fluid viscosity
        self.k_B = constants.Boltzmann                                       # Boltzmann constant
        self.temp = 10.                                                      # absolute temperature

        self.l = l                                                           # arena side
        self.A = self.l**2                                                   # arena area
        # self.rho = self.N/self.A                                             # particles density

        # arena particle boundaries
        self.x_min = - l/2 + self.R
        self.x_max = + l/2 - self.R
        self.y_min = - l/2 + self.R
        self.y_max = + l/2 - self.R

        self.gamma_t = translfric * constants.pi * self.eta * self.R         # translational friction coefficient
        self.gamma_r = rotatfric * constants.pi * self.eta * self.R**3       # rotational friction coefficient

        self.mu = mob                                                        # mobility
        self.D_t = self.k_B*self.temp/self.gamma_t                           # translation diffusion
        # self.D_r = self.k_B*self.temp/self.gamma_r                           # rotation diffusion
        # self.tau_r = 1/self.D_r                                              # persistence time
        self.v_0 = v_0                                                       # self-propulsion speed
        # self.l_p = self.v_0*self.tau_r                                       # persistence length
        self.a = 1.                                                          # force parameter
        self.k = 1.                                                          # force parameter
        # self.Pe_r = self.l_p/self.a                                          # rotational Péclet number (in Paper 2, D_r is determined for a given Pe_r with fixed a and v_0)
        self.Pe_r = pecnum
        self.D_r = self.v_0/(self.Pe_r*self.a)

        # self.phi = np.pi * np.sum(self.a_particles**2) / self.A              # packing fraction (in paper 2, this is used to determine the number of particles given a and A)
        self.phi = packfrac
        self.N = int(self.A * self.phi / (constants.pi * self.a**2))

        self.x = l*np.random.rand(self.iterations, self.N, 2) - self.l/2                    # 2D position (x, y)
        self.theta = 2*constants.pi*np.random.rand(self.iterations, self.N)                 # angle
        self.F = np.zeros((self.iterations, self.N, 2))                                     # soft repulsive forces
        self.r = np.zeros((self.iterations, self.N, 2))                                     # unit vector for soft repulsive forces
        self.a_particles = self.a*np.ones((self.N, 1))                                      # force parameter vector

        self.W_x = np.sqrt(2 * self.D_t) * np.random.randn(self.iterations, self.N, 2)      # Wiener noise on translation
        self.W_theta = np.sqrt(2 * self.D_t) * np.random.randn(self.iterations, self.N)     # Wiener noise on rotation

        self.length_orientation = .5

    # @jit(nopython=True)
    def step(self, i):
        # t0 = time.time()
        # self.distance_ij = squareform(pdist(self.x[i,:,:], 'euclidean'))
        # self.direction = self.x[i,:,None,:] - self.x[i,:,:]
        # self.direction_ij = np.divide(self.direction, self.distance_ij[:, :, None], out=np.zeros_like(self.direction), where=self.distance_ij[:, :, None]!=0)
        # self.f_ij = self.k * (self.a_particles[:,None,0] + self.a_particles[:,0] - self.distance_ij)
        # self.f_ij[self.f_ij < 0] = 0
        # self.F[i,:,:] = np.sum(self.f_ij[:,:, None] * self.direction_ij, axis=1)

        for j in range(self.N):
            self.distance_ij = np.linalg.norm(self.x[i,j,:] - self.x[i,:,:], axis=1)
            self.direction_ij = (self.x[i,j,:] - self.x[i,:,:]) / self.distance_ij[:, None]
            self.f_ij = self.k * (self.a_particles[j,0] + self.a_particles[:,0] - self.distance_ij)
            self.f_ij[self.f_ij < 0] = 0

            self.F[i,j,:] = np.nansum(self.f_ij[:, None] * self.direction_ij, axis=0)


        """perform animation step"""
        self.dx = self.v_0 * np.cos(self.theta[i,:]) + self.F[i,:,0] + self.W_x[i,:,0]/np.sqrt(self.dt)
        self.dy = self.v_0 * np.sin(self.theta[i,:]) + self.F[i,:,1] + self.W_x[i,:,1]/np.sqrt(self.dt)
        self.dtheta = self.W_theta[i,:]/np.sqrt(self.dt)

        self.x[i+1,:,:] = self.x[i,:,:] + self.dt * np.array([self.dx, self.dy]).T
        self.theta[i+1,:] = self.theta[i,:] + self.dt * self.dtheta

        # particles bouncing off the walls, not really working too well if thermal noise allows for "jumps"
        # theta[i+1,x[i+1,:,0]<x_min] = constants.pi - theta[i+1,x[i+1,:,0]<x_min]
        # theta[i+1,x[i+1,:,0]>x_max] = constants.pi - theta[i+1,x[i+1,:,0]>x_max]

        # theta[i+1,x[i+1,:,1]<y_min] = 2*constants.pi - theta[i+1,x[i+1,:,1]<y_min]
        # theta[i+1,x[i+1,:,1]>y_max] = 2*constants.pi - theta[i+1,x[i+1,:,1]>y_max]

        # toroidal world
        self.x[i+1,self.x[i+1,:,0]<self.x_min,0] = self.x_max
        self.x[i+1,self.x[i+1,:,0]>self.x_max,0] = self.x_min
        self.x[i+1,self.x[i+1,:,1]<self.y_min,1] = self.y_max
        self.x[i+1,self.x[i+1,:,1]>self.y_max,1] = self.y_min

        # t1 = time.time()
        # total = t1-t0
        # print('One step:', total)
    
    def init(self):
        """initialize animation"""
        time_text.set_text('')

        particles.set_data([], [])
        particles_orientation.set_data([], [])

        return particles, particles_orientation, time_text

    def animate(self, i):
        self.step(i)
        time_text.set_text('iteration = %.2f' % i)

        particles.set_data([self.x[i,:,0]], [self.x[i,:,1]])
        # print([x[i,:,0], x[i,:,0]+length_orientation*np.cos(theta[i,:,0])])
        # particles_orientation.set_data([x[i,:,0], x[i,:,0]+length_orientation*np.cos(theta[i,:])], [x[i,:,1], x[i,:,1]+length_orientation*np.sin(theta[i,:])])

        return particles, particles_orientation, time_text

dt = 0.1
T = 10
length = 3
Pn = 50
sim = ABPSimulation(l=length, pecnum=Pn, dt=dt, T=T)

# plotting 
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(111, aspect='equal', autoscale_on=True, xlim=(- sim.l/2, sim.l/2), ylim=(- sim.l/2, sim.l/2))
ax1.set_xticks(np.arange(- sim.l/2, sim.l/2, sim.l/10))
ax1.set_yticks(np.arange(- sim.l/2, sim.l/2, sim.l/10))
ax1.grid()
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
plt.title("Péclet number, $Pe_r$: {}, packing fraction, $\phi$: {:.2f}, number of particles: {}".format(sim.Pe_r, sim.phi, sim.N))

particles, = ax1.plot([], [], 'o', ms=8, lw=2, label='Particle')
particles_orientation, = ax1.plot([], [], '-', lw=2)


# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

ani = animation.FuncAnimation(fig, sim.animate, frames=int(sim.iterations-1),
                              interval=1, blit=True, init_func=sim.init)


t0 = time.time()
ani.save('lines.mp4', writer=writer)
t1 = time.time()
total = t1-t0
print(total)

# plt.show()


