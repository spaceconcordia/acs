###############################################################################
#                     I M P O R T    L I B R A R I E S
###############################################################################
import PIL # Python Imaging Library
import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sympy.utilities.lambdify import lambdify
from scipy.integrate import ode
from mpl_toolkits.mplot3d import axes3d as p3

###############################################################################
#                     S E T U P    L I B R A R I E S
###############################################################################
np.set_printoptions(precision=15)
mpl.rcParams['figure.figsize'] = (8.0, 6.0)

###############################################################################
#                       M A I N     F U N C T I O N:
###############################################################################
def main() :
  # Instantiate symbolic variables
  mu, r, x, y, z = sp.symbols('mu r x y z')

  # Define gravity potential function and take partials
  U_str = "mu / r"
  U = sp.sympify(U_str)
  U = U.subs(r, sp.sqrt(x**2 + y**2 + z**2))
  dUdx = sp.diff(U, x)
  dUdy = sp.diff(U, y)
  dUdz = sp.diff(U, z)

  # Print EOMs to the console
  print(sp.simplify(dUdx.subs(sp.sqrt(x**2 + y**2 + z**2), r)))
  print(sp.simplify(dUdy.subs(sp.sqrt(x**2 + y**2 + z**2), r)))
  print(sp.simplify(dUdz.subs(sp.sqrt(x**2 + y**2 + z**2), r)))

  # Set constants
  mu = 398600.4415
  R_earth = 6378.145

  # Substitue constants into partial expressions
  dUdx = dUdx.subs([('mu', mu)])
  dUdy = dUdy.subs([('mu', mu)])
  dUdz = dUdz.subs([('mu', mu)])

  # Transform symbolic expressions into numeric expressions 
  # (to use in the integrator)
  dUdx_func = lambdify((x, y, z), dUdx)
  dUdy_func = lambdify((x, y, z), dUdy)
  dUdz_func = lambdify((x, y, z), dUdz)

  # Set initial values and integration time parameters
  r_vec_0 = [-2436.45, -2436.45, 6891.037]  #  km
  v_vec_0 = [5.088611, -5.088611, 0.0]      #  km/s
  t_0 = 0.0
  t_f = 0.5*86400.0
  dt = 20.0

  # Equations of motion, output is velocity and acceleration
  def orbitEOM(t, rv_vecs):
      r_vec, v_vec = rv_vecs.reshape(2, 3)    
      a_vec = np.zeros(3)    
      a_vec[0] = dUdx_func(*r_vec)
      a_vec[1] = dUdy_func(*r_vec)
      a_vec[2] = dUdz_func(*r_vec)    
      deriv_vec = np.array([v_vec, a_vec])
      return deriv_vec.flatten()

  # Define initial condition state for running the integrator
  y_0 = np.array([r_vec_0, v_vec_0]).flatten()

  # Define integrator, set tolerances and the initial state
  rv = ode(orbitEOM)
  rv.set_integrator('dopri5', rtol=3e-14, atol=1e-16)
  rv.set_initial_value(y_0, t_0)

  # Define output array
  output = []
  output.append(np.insert(y_0, 0, t_0))

  # Run the integrator and populate output array with positions and velocities
  while rv.successful() and rv.t < t_f:
	  rv.integrate(rv.t + dt)
	  output.append(np.insert(rv.y, 0, rv.t))
	
  output = np.array(output)

  # Sanity check to see final values for position and velocity
  print(output[-1])
  times = output[:,0]
  positions = output[:,[1, 2, 3]]
  velocities = output[:,[4, 5, 6]]

  # Find initial position and velocity magnitudes for initial specific 
  # energy calcs
  r_0 = np.linalg.norm(r_vec_0)
  v_0 = np.linalg.norm(v_vec_0)
  E_0 = ((v_0**2) / 2) - (mu / r_0)

  # Find all position and velocity magnitudes for specific energy calcs
  r = np.linalg.norm(positions, axis=1)
  v = np.linalg.norm(velocities, axis=1)
  E_t = ((v**2) / 2) - (mu / r)

  plt.figure(1)
  plt.plot(times/3600.0, positions)
  plt.title('Satellite Position', fontsize=16)
  plt.xlabel('Time [hrs]', fontsize=16)
  plt.ylabel('x [km]', fontsize=16)
  
  ############################################################################
  # Plotting an orbiting satellite around Earth
  ############################################################################
  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
  coefs = (1, 1, 1)  

  # Radii corresponding to the coefficients:
  rx, ry, rz = [R_earth/np.sqrt(coef) for coef in coefs]


  ############################################################################
  #
  # plot_sphere()
  #
  # Plot a rotatable 3D sphere representing the earth 
  # image_path: string of the path to an equirectuagular image of the earth
  #
  ############################################################################
  def plot_sphere():
    print "Plotting Sphere..."
    
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    
    # Plot
    fig = plt.figure(2)
    ax = p3.Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='white')
    ax.plot(positions[:,0],positions[:,1],positions[:,2])
  
  #############################################################################
  #
  # plot_earth()
  #
  # Plot a rotatable 3D sphere with an image of the earth projected onto its 
  # surface
  #
  # params:
  #   image_path: string of the path to an equirectuagular image of the earth
  #
  # reference: 
  # http://stackoverflow.com/questions/30269099/creating-a-rotatable-3d-earth?lq=1
  ###############################################################################
  def plot_earth(image_path):
    print "Plotting Spherical Earth..."

    # Import image
    bm = PIL.Image.open(image_path)
  
    # scale image, divide by 256 for matplotplib RGB values 
    scaling_factor = 1
    bm = np.array(bm.resize([d/scaling_factor for d in bm.size]))/256.
    
    # Set of all spherical angles:
    # TODO - don't know if this is entirely accurate, but probably close
    u = np.linspace(-np.pi, np.pi, bm.shape[1])
    v = np.linspace(-np.pi/2, np.pi/2, bm.shape[0])[::-1]
    
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.cos(v)).T
    y = ry * np.outer(np.sin(u), np.cos(v)).T
    z = rz * np.outer(np.ones(np.size(u)), np.sin(v)).T
    
    # Plot
    fig = plt.figure(3)
    ax = p3.Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)
    ax.plot(positions[:,0],positions[:,1],positions[:,2])
    

  plot_sphere()
  plot_earth('matlab/1024px-Land_ocean_ice_2048.jpg')
  plt.show()
  print "Done\r\n"

  return

if __name__ == "__main__":
  main()
