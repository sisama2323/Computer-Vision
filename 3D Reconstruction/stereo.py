import pickle
import matplotlib.pyplot as plt
from scipy.signal import convolve
from numpy import linalg
import numpy as np
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

def get_rot_mat(rot_v, unit=None):
    '''
    takes a vector and returns the rotation matrix required to align the unit vector(2nd arg) to it
    '''
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    
    rot_v = rot_v/np.linalg.norm(rot_v)
    uvw = np.cross(rot_v, unit) #axis of rotation

    rcos = np.dot(rot_v, unit) #cos by dot product
    rsin = np.linalg.norm(uvw) #sin by magnitude of cross product

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw = uvw/rsin
    u, v, w = uvw

    # Compute rotation matrix 
    R = (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )
    
    return R

def photometric_stereo(images,lights, n):
    normals = np.zeros(np.append(np.shape(images[0]),3))
    albedo = np.zeros(np.shape(images[0]))
    H = np.zeros((int(np.size(images[0],0))+1, int(np.size(images[0],1))+1))
    q = albedo
    p = albedo
    
    for i in range(0, np.size(lights, 0)):
        lights[i,:] = -lights[i, :]/np.linalg.norm(lights[i, :])
    
    for i in range(0, np.size(images,1)):
        for j in range(0, np.size(images,2)):
            if n==3:
                e = np.array([[images[0][i,j], images[1][i,j], images[2][i,j]]]).T
                b = np.dot(e, np.linalg.inv(lights))
                b = b.T
            else:
                e = np.array([[images[0][i,j], images[1][i,j], images[2][i,j], images[3][i,j]]]).T
                b = np.dot(np.dot(np.linalg.inv(np.dot(lights.T, lights)),lights.T),e)
                
                
            albedo[i,j] = np.linalg.norm(b)
            normals[i,j,:] = b.ravel()/albedo[i,j]
            
            # x/z
            p[i,j] = normals[i,j,0]/normals[i,j,2];
            # y/z
            q[i,j] = normals[i,j,1]/normals[i,j,2];
            H[i+1, j+1] = p[:i-1, j] + p[:i, j] + q[i, :j-1] + q[i, :j] 

    
    mask = np.ones(np.shape(images[0]))
    for i in range(0, np.size(images[0],0)):
        for j in range(0, np.size(images[0],1)):
            if abs(images[0][i, j]) < 0.00001:
                mask[i,j] = 0

    H_horn = horn_integrate(p,q,mask,2000)
    return albedo,normals,H,H_horn



pickle_in = open("HW2\specular_pear.pickle")
data = pickle.load(pickle_in)
#data is a dict which stores each element as a key-value pair. 
print data.keys()

#To access the value of an entity, refer it by its key.
# plt.imshow(data['im1'][:,:,0])
# plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

gray1 = rgb2gray(data['im1'])   
gray2 = rgb2gray(data['im2'])
gray3 = rgb2gray(data['im3'])    
gray4 = rgb2gray(data['im4'])         
plt.imshow(gray1, cmap = plt.get_cmap('gray'))
plt.show()

lights = np.vstack((data['l1'],data['l2'],data['l3'],data['l4']))
#lights = np.vstack((data['l1'],data['l2'],data['l4']))
#Hint: be careful about the light-source location and direction of light. 
#lights right now stores light-source locations

images = []
images.append(gray1)
images.append(gray2)
images.append(gray3)
images.append(gray4)

albedo, normals, depth, horn = photometric_stereo(images, lights, 4)

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

def RGBToSUV(I_rgb,rot_vec):
    rot_mat = get_rot_mat(rot_vec)
    
    S = np.zeros(np.shape(I_rgb[:,:,0]))
    G = np.zeros(np.shape(I_rgb[:,:,0]))
    for i in range(0, np.size(I_rgb, 0)):
        for j in range(0, np.size(I_rgb, 1)):
            I_suv = np.dot(rot_mat, I_rgb[i,j,:])
            S[i,j] = I_suv[0]
            G[i,j] = (I_suv[1]**2+I_suv[2]**2)**(0.5)
    return (S,G)

pickle_in = open("HW2/specular_sphere.pickle","rb")
data = pickle.load(pickle_in)
#sample input
S,G = RGBToSUV(data['im1'],np.hstack((data['c'][0][0],data['c'][1][0],data['c'][2][0])))