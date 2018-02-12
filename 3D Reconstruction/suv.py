import numpy as np
import pickle

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