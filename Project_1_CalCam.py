import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import os

def cam_cal(): 
    try: 
        obs_pts = np.loadtxt('observe.dat') #2d points
        model_pts = np.loadtxt('model.dat') #3d points
        obs = torch.from_numpy(obs_pts).float()
        model = torch.from_numpy(model_pts).float()
    except FileNotFoundError: 
        print("file not found")

    starttime = time.time()

    #account for noise or add noise
    if_noise = 0
    stand_dev_noise = 5

    #Use pytorch to combine 2d and 3d points into one tensor dimesion
    all_pts = torch.cat((obs, model), dim=1) #shape [num_pts, 5] - [u,v,Wx,Wy,Wz]
    total_pts, _ = all_pts.shape
    n =  27 #number of points in observe.dat 

    if if_noise:
        for i in range(0, n, 8):
            noise = torch.randn(2) * stand_dev_noise
            all_pts[i, 0:2] += noise

    rand_pts = all_pts[torch.randperm(total_pts)[:n], :] #randomly select n points
    print(rand_pts)

    #build the Q matrix for DLT. 
    #W is world points, and u, v are image points 
    Q = torch.zeros((2*n, 12))
    for i in range(n):
        wx = rand_pts[i, 2]
        wy = rand_pts[i, 3]
        wz = rand_pts[i, 4]
        u = rand_pts[i, 0]
        v = rand_pts[i, 1]

        #turn world points to homogeneous
        P = torch.tensor([wx, wy, wz, 1.0])
        #add zeros for P
        PO = torch.zeros(4)

        row1 = torch.cat((PO, -P, v*P), dim=0)
        row2 = torch.cat((P, PO, -u*P), dim=0)

        Q[2*i, :] = row1
        Q[2*i+1, :] = row2

    #solve for projection matrix M using singular value decomp
    # Qm = 0, the solution is in the last column of V from SVD
    U, S, Vh = torch.linalg.svd(Q)

    #Find the smallest value in Vh
    m = Vh[-1, :]

    #now lets reshape the projection matrix into the 3x4 that we need
    # lets also visualize it
    M = m.reshape(3, 4)
    print("projection matrix:\n", M)

    # now that we have that done lets break M down
    # to get K, R, and t 
    # these are the intrinsic and extrinsic parameters
    A = M[0:3, 0:3]
    b = M[0:3, 3]

    #a1, a2, a3 are the rotation vector coordinates
    a1 = A[0, :]
    a2 = A[1, :]
    a3 = A[2, :]

    #now lets do some math to get the intrinsic parts first
    absr = 1.0 / torch.norm(a3) #this is to normalize r
    r = absr 

    #now for u0 and v0
    u0 = r**2 * torch.dot(a1, a3)
    v0 = r**2 * torch.dot(a2, a3)

    #now cross products 
    a1xa3 = torch.linalg.cross(a1, a3)
    a2xa3 = torch.linalg.cross(a2, a3)

    #now for the skew angle
    rahhh = -r**4 * torch.dot(a1xa3, a2xa3) / (torch.norm(a1xa3) * torch.norm(a2xa3))
    cos_theta = torch.clamp(rahhh, -1.0, 1.0)
    #skew angle rahhhhhhh
    theta = torch.acos(cos_theta)
    skew = torch.sin(theta)

    #now for the focal lengths
    ep = 1e-2 #no div by 0
    kf = r**2 * torch.norm(a1xa3) * (skew + ep)
    lf = r**2 * torch.norm(a2xa3) * (skew + ep)

    #lets now do the extrinsic parameters
    r1 = a2xa3 / torch.linalg.norm(a2xa3)
    r3 = r * a3
    r2 = torch.linalg.cross(r3, r1)
    R = torch.stack((r1, r2, r3))

    cotang = cos_theta / skew

    print("skew:", skew)
    print("lf:", lf)
    print("kf:", kf)
    print("cotang:", cotang)

    #now for intrinsic matrix K
    K = torch.tensor([[kf, kf * cotang, u0],
                      [0.0, lf / skew + ep, v0],
                      [0.0, 0.0, 1.0]])

    #now for extrinsic translation vector t
    t = r * torch.linalg.inv(K) @ b 

    #Lets make sure that the calibration is right
    print("\nIntrinsic Matrix K:\n", K)
    print("\nRotation Matrix R:\n", R)
    print("\nTranslation Vector t:\n", t)

    #project the 3d grid onto the 2d image
    try: 
        image = Image.open('test_image.bmp')
        imnp = np.array(image)
        imog = imnp.copy()
        if len(imnp.shape) == 3:
            height, width, _ = imnp.shape
        else: #grayscale image
            height, width = imnp.shape
        print("image size (height, width):", height, width)
    except FileNotFoundError:
        print("image file not found")

    #lets set up the grid now
    m1 = M[0, :]
    m2 = M[1, :]
    m3 = M[2, :]

    #now points for the 3d cube
    for i in range (11): 
        for j in range(11):
            for k in range(11):
                if i == 10 or j == 10 or k == 0:
                    P = torch.tensor([i, j, k, 1.0]).float()
                    #projection matrix
                    u_proj = torch.dot(m1, P) / torch.dot(m3, P)
                    v_proj = torch.dot(m2, P) / torch.dot(m3, P)
                    pu = int(torch.round(u_proj))
                    pv = int(torch.round(v_proj))

                    #draw the 2x2 square
                    if 0 <= pv < imnp.shape[0] - 2 and 0 <= pu < imnp.shape[1] - 2:
                        imnp[pv - 3:pv + 3, pu - 3:pu + 3] = 0 # make grey

    endtime = time.time()
    totaltime = endtime - starttime
    print("\nTotal time for linear combination: ", totaltime, " seconds")

    #Now to display the images RAHHHH
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(imog)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(imnp)
    plt.title("Projected Image")
    plt.axis('off')
    plt.show()
    return M, K, R, t

if __name__ == "__main__":
    M, K, R, t = cam_cal()