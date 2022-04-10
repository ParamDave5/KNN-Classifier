import numpy as np

def compute_A(world , image):
    A = []
    for i in range(8):
        Xi = world[i]
        u, v, w = image[i]
        A_row_1 = np.array([0, 0, 0, 0, -w*Xi[0], -w*Xi[1], -w*Xi[2], -w*Xi[3], v*Xi[0], v*Xi[1], v*Xi[2], v*Xi[3]])
        A_row_2 = np.array([w*Xi[0], w*Xi[1], w*Xi[2], w*Xi[3], 0, 0, 0, 0, -u*Xi[0], -u*Xi[1], -u*Xi[2], -u*Xi[3]])
        A.append(A_row_1)
        A.append(A_row_2)
    return A


def compute_P(A):
    U, Sigma, V = np.linalg.svd(A)  

    P = np.reshape(V[-1, :], (3, 4))
    Lambda = P[-1,-1]
    P = P/Lambda
    return P

def compute_M(P):
    U , D, V = np.linalg.svd(P)
    C = V[-1, :]
    C = C/C[-1]
    C = np.reshape(C, (4, 1))
    I = np.identity(3)
    C = np.concatenate((I, -1*C[:-1]), axis=1)
    C_inv = np.linalg.pinv(C)
    M = np.matmul(P, C_inv)
    return M

def RQFactorization(M):
    c = (M[2,2]/((M[2,1])**2 + (M[2,2])**2)**(0.5))
    s = -(M[2,1]/((M[2,1])**2 + (M[2,2])**2)**(0.5))
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    M = np.matmul(M, Rx)
    c = (M[2,2]/((M[2,0])**2 + (M[2,2])**2)**(0.5))
    s = (M[2,0]/((M[2,0])**2 + (M[2,2])**2)**(0.5))
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    M = np.matmul(M, Ry)
    c = (M[1,1]/((M[1,0])**2 + (M[1,1])**2)**(0.5))
    s = -(M[1,0]/((M[1,0])**2 + (M[1,1])**2)**(0.5))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    K = np.matmul(M, Rz)
    return Rx , Ry , Rz ,K

world = [[0,0,0,1],[0,3,0,1],[0,7,0,1],[0,11,0,1],[7,1,0,1],[0,11,7,1],[7,9,0,1],[0,1,7,1]]
world = np.array(world)

image = [[757,213,1],[758,415,1],[758,686,1],[759,966,1],[1190,172,1],[329,1041,1],[1204,850,1],[340,159,1]]
image = np.array(image)


A = compute_A(world,image)
A = np.array(A)

P = compute_P(A)
M = compute_M(P)

rx , ry , rz , K = RQFactorization(M)
K_i = []
for i in K:
    for j in i:
        if abs(j) < 0.0001:
            j = 0
        K_i.append(j)
K_i = np.array(K_i)
K = K_i.reshape(3,3).astype(np.float32)
print('The camera calibration matrix is: ',K)