import PIL

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

scaling_factor = 1

# load bluemarble with PIL
#bm = PIL.Image.open('bluemarble.jpg')
#bm = PIL.Image.open('earth0_globe0.jpg')
#bm = PIL.Image.open('earth.jpg')
#bm = PIL.Image.open('equirectangular_earth.jpg')
#bm = PIL.Image.open('eq_earth_huge.jpg')
bm = PIL.Image.open('matlab/1024px-Land_ocean_ice_2048.jpg')
# it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept 
bm = np.array(bm.resize([d/scaling_factor for d in bm.size]))/256.

# coordinates of the image - don't know if this is entirely accurate, but probably close
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 

# repeat code from one of the examples linked to in the question, except for specifying facecolors:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_bgcolor('black')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['top'].set_color('white')
ax.xaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='z', colors='white')

x = np.outer(np.cos(lons), np.cos(lats)).T
y = np.outer(np.sin(lons), np.cos(lats)).T
z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)

plt.show()
