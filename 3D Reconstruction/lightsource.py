import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
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

import numpy as np
import matplotlib.pyplot as plt

def compute_radius(mask):
    area = np.size(np.argwhere(mask!=0),0)
    r = np.sqrt(area/np.pi)
    return r

def compute_center(mask):
    idx = sum(np.argwhere(mask!=0))/np.size(np.argwhere(mask!=0),0)
    x, y = idx[1], idx[0]
    return x,y

def compute_brightest(img):
    idx = np.where(img == img.max())
    x = idx[1][int(np.size(idx,1)*0.5)]
    y = idx[0][int(np.size(idx,1)*0.5)]
    return x,y

def light_direction(img, mask):
    R = np.array([[0, 0, 1]]).T
    cx,cy = compute_center(mask)
    r = compute_radius(mask)
    bx,by = compute_brightest(img)

    N = np.array([[bx-cx, by-cy, (r**2 - (bx-cx)**2 - (by-cy)**2)**0.5]]).T

    L = 2*np.dot(N.T, R)*N - R
    L = L/np.linalg.norm(L)
    return L.T



im1 = imread('HW2/sphere/sphere1.png',flatten=True)
im2 = imread('HW2/sphere/sphere2.png',flatten=True)
im3 = imread('HW2/sphere/sphere3.png',flatten=True)
im4 = imread('HW2/sphere/sphere4.png',flatten=True)
# im5 = imread('HW2/sphere/sphere5.png',flatten=True)
# im6 = imread('HW2/sphere/sphere6.png',flatten=True)
# im7 = imread('HW2/sphere/sphere7.png',flatten=True)
# im8 = imread('HW2/sphere/sphere8.png',flatten=True)
msk = imread('HW2/sphere/spheremask.png',flatten=True)
# plt.imshow(im2)
# plt.show()

l1 = np.array(light_direction(im1,msk))
l2 = np.array(light_direction(im2,msk))
l3 = np.array(light_direction(im3,msk))
l4 = np.array(light_direction(im4,msk))
# l5 = np.array(light_direction(im5,msk))
# l6 = np.array(light_direction(im6,msk))
# l7 = np.array(light_direction(im7,msk))
# l8 = np.array(light_direction(im8,msk))

lights = np.vstack((l1,l2,l3,l4))
# lights = np.vstack((l1,l2,l3,l4,l5,l6,l7,l8))
# lights[:,2] = -lights[:,2]

images = []
images.append(im1)
images.append(im2)
images.append(im3)
images.append(im4)
# images.append(im5)
# images.append(im6)
# images.append(im7)
# images.append(im8)

albedo, normals, depth, horn = photometric_stereo(images, lights)

#showing normals as quiver
X, Y, Z = np.meshgrid(np.arange(0,np.shape(normals)[0], 15), np.arange(0,np.shape(normals)[1], 15),  np.arange(0,np.shape(normals)[2], 15))

#plotting wireframe depth map
H = depth[::15,::15].T
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X[...,0],Y[...,0], H)
plt.show()

H = horn[::15,::15].T
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X[...,0],Y[...,0], H)
plt.show()