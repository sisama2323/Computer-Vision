import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D


def horn_integrate(gx,gy,mask,niter):
    '''
    integrate_horn recovers the function g from its partial 
    derivatives gx and gy. 
    mask is a binary image which tells which pixels are 
    involved in integration. 
    niter is the number of iterations. 
    typically 100,000 or 200,000, 
    although the trend can be seen even after 1000 iterations.
    '''    
    g = np.ones(np.shape(gx))
    
    gx = np.multiply(gx,mask)
    gy = np.multiply(gy,mask)
    
    A = np.array([[0,1,0],[0,0,0],[0,0,0]]) #y-1
    B = np.array([[0,0,0],[1,0,0],[0,0,0]]) #x-1
    C = np.array([[0,0,0],[0,0,1],[0,0,0]]) #x+1
    D = np.array([[0,0,0],[0,0,0],[0,1,0]]) #y+1
    
    d_mask = A+B+C+D
    
    den = np.multiply(convolve(mask,d_mask,mode='same'),mask)
    den[den==0] = 1
    rden = 1.0/den
    mask2 = np.multiply(rden,mask)
    
    m_a = convolve(mask,A,mode='same')
    m_b = convolve(mask,B,mode='same')
    m_c = convolve(mask,C,mode='same')
    m_d = convolve(mask,D,mode='same')
    
    term_right = np.multiply(m_c,gx) + np.multiply(m_d,gy)
    t_a = -1.0*convolve(gx,B,mode='same')
    t_b = -1.0*convolve(gy,A,mode='same')
    term_right = term_right+t_a+t_b
    term_right = np.multiply(mask2,term_right)
    
    for k in range(niter):
        g = np.multiply(mask2,convolve(g,d_mask,mode='same'))\
            +term_right;
    
    return g

def fix_light(lights):
    new_lights = lights.copy()
    new_lights[:, 0] = lights[:,1]
    new_lights[:, 1] = lights[:,0]
    new_lights[:, 2] = -lights[:,2]
    return new_lights

def norm(b):
    return np.sqrt(np.sum(b**2))

def photometric_stereo(images,lights):
    images = np.dstack(images)
    (row, col, numpic) = np.shape(images)
    normals = np.zeros((row, col, 3))
    px = np.zeros((row, col))
    qy = np.zeros((row, col)) 
    H = np.zeros((row, col))
    albedo = np.zeros((row, col))
                                                                                                            
    for i in range(row):
        for j in range(col):
            if numpic==3:
                e = np.array([images[i,j,:]])
                b = np.dot(e, np.linalg.inv(lights.T))
                b = b.T
            else:
                e = np.array([images[i,j,:]]).T
                b = np.dot(np.dot(np.linalg.inv(np.dot(lights.T, lights)),lights.T),e)
#                 b = np.dot(np.linalg.pinv(lights),e)
            
#             print b
            albedo[i,j] = np.linalg.norm(b)
            normals[i,j,:] = b.ravel()/albedo[i,j]
            
            # x/z
            px[i,j] = normals[i,j,0]/normals[i,j,2]
            # y/z
            qy[i,j] = normals[i,j,1]/normals[i,j,2]
            
            if (px[i,j]-qy[i,j])**2 > 50:
                px[i,j] = qy[i,j] = 0
    
    for i in range(1, np.size(H, 1)):
            H[0, i] = H[0, i-1] + px[0, i] 

    for i in range(1, np.size(H, 0)):
        for j in range(0, np.size(H, 1)):
            H[i, j] = H[i-1, j] + qy[i, j]
    
    albedo_max = albedo.max()
    normalb= albedo/albedo_max
    mask = np.ones((row,col))
    for i in range(row):
        for j in range(col):
            if abs(normalb[i, j]) < 0.1:
                mask[i,j] = 0

    H_horn = horn_integrate(px, qy, mask, 2000)
    return albedo,normals,H,H_horn
        

pickle_in = open("HW2/synthetic_data.pickle")
data = pickle.load(pickle_in)

lights = np.vstack((data['l1'],data['l2'],data['l3'],data['l4']))
new_lights = fix_light(lights)

# lights = np.vstack((data['l1'],data['l2'],data['l4']))
#Hint: be careful about the light-source location and direction of light. 
#lights right now stores light-source locations
images = []
images.append(data['im1'])
images.append(data['im2'])
images.append(data['im3'])
images.append(data['im4'])

albedo, normals, depth, horn = photometric_stereo(images, new_lights)
#--------------------------------------------------------------------------
#Following code is just a working example so you don't get stuck with any 
#of the graphs required. You may want to write your own code to align the
#results in a better layout
#--------------------------------------------------------------------------

fig = plt.figure()
albedo_max = albedo.max()
albedo = albedo/albedo_max
plt.imshow(albedo, cmap='gray')
plt.show()

#showing normals as three seperate channels
figure = plt.figure()
ax1 = figure.add_subplot(131)
ax1.imshow(normals[..., 0])
ax2 = figure.add_subplot(132)
ax2.imshow(normals[..., 1])
ax3 = figure.add_subplot(133)
ax3.imshow(normals[..., 2])
plt.show()

#showing normals as quiver
X, Y, Z = np.meshgrid(np.arange(0,np.shape(normals)[0], 15), np.arange(0,np.shape(normals)[1], 15),  np.arange(0,np.shape(normals)[2], 15))
NX = normals[..., 0][::15,::15]
NY = normals[..., 1][::15,::15]
NZ = normals[..., 2][::15,::15]
fig = plt.figure(figsize=(5,5))
ax = fig.gca(projection='3d')
ax.view_init(azim=90, elev=-90)
plt.quiver(X,Y,Z,NX,NY,NZ, facecolor='r', linewidth=.5)
plt.show()

#plotting wireframe depth map
H = depth[::15,::15]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X[...,0],Y[...,0], H)
plt.show()

H = horn[::15,::15]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X[...,0],Y[...,0], H)
plt.show()