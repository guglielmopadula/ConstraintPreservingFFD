import numpy as np
from numba import jit,prange
from numba.extending import get_cython_function_address
import ctypes
import numpy as np
from pygem.ffd import FFD
from tqdm import trange
import time

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "binom")
functype = ctypes.CFUNCTYPE(_dble, _dble,_dble)
binom = functype(addr)

def generate_random_sdp(n):
    A=np.random.rand(n,n)
    A=A@A.T
    A=A+n*np.eye(n)
    return A


@jit(nopython=True)
def numba_meshgrid(x, y, z):
    nx, ny, nz = len(x), len(y), len(z)
    xx = np.zeros((nx, ny, nz))
    yy = np.zeros((nx, ny, nz))
    zz = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                xx[i,j,k] = x[i]
                yy[i,j,k] = y[j]
                zz[i,j,k] = z[k]

    return xx, yy, zz

@jit(nopython=True) 
def _bernstein_mesh(points,n_control_points):
    n_points=points.shape[0]
    bern=np.zeros((n_points,*n_control_points))
    for p in range(n_points):
        for i in range(n_control_points[0]):
            for j in range(n_control_points[1]):
                for k in range(n_control_points[2]):
                    bern[p,i,j,k]=binom(float(n_x-1),float(i))*points[p,0]**i*(1-points[p,0])**(n_x-1-i)*binom(float(n_y-1),float(j))*points[p,1]**j*(1-points[p,1])**(n_y-1-j)*binom(float(n_z-1),float(k))*points[p,2]**k*(1-points[p,2])**(n_z-1-k)
    return bern
#@jit(nopython=True,parallel=True) 
def _constrained_ffd(A,b,M,indices_x,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z):
    points_new=_classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    n_points=len(points)
    b_tmp=A@(points_new.reshape(-1)[indices_x])
    new_array=np.concatenate((array_mu_x.copy(),array_mu_y.copy(),array_mu_z.copy())).reshape(3,-1)
    delta_b=b-b_tmp
    num_axis=np.unique(indices_x%3)
    bmesh=_bernstein_mesh(points,n_control_points)
    A_c=np.zeros((A.shape[0],len(num_axis),len(indices_c)))
    for i in prange(A.shape[0]):
        for i_g in prange(len(indices_x)):
            for i_h in prange(len(indices_c)):
                a,b,c=_f(indices_c[i_h],n_control_points)
                d,=np.where(num_axis==(indices_x[i_g]%3))[0]
                A_c[i,d,i_h]+=A[i,i_g]*bmesh[(indices_x[i_g]//3),a,b,c]
    A_c=A_c.reshape(A.shape[0],-1)
    delta_q=np.linalg.solve(M,A_c.T)@(np.linalg.solve(A_c@np.linalg.solve(M,A_c.T),delta_b))
    print(np.linalg.norm(A_c@delta_q-delta_b))
    delta_q=delta_q.reshape(len(num_axis),len(indices_c))
    for i in range(len(num_axis)):
        for j in range(len(indices_c)):
            new_array[num_axis[i],indices_c[j]]+=delta_q[i,j]
    
    new_array_x=new_array[0].reshape(n_x,n_y,n_z)
    new_array_y=new_array[1].reshape(n_x,n_y,n_z)
    new_array_z=new_array[2].reshape(n_x,n_y,n_z)
    return _classic_ffd(points,n_control_points,new_array_x,new_array_y,new_array_z)



@jit(nopython=True) 
def _f(h,n_control_points):
    m=n_control_points[0]
    n=n_control_points[1]
    o=n_control_points[2]
    return np.array([h//(n*o),(h%(n*o))//o,h%o])


@jit(nopython=True) 
def _classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z):
    control_points=_control_points(n_control_points,array_mu_x,array_mu_y,array_mu_z).reshape(3,-1)
    n_x=n_control_points[0]
    n_y=n_control_points[1]
    n_z=n_control_points[2]
    control_x=control_points[0].reshape(n_x,n_y,n_z)
    control_y=control_points[1].reshape(n_x,n_y,n_z)
    control_z=control_points[2].reshape(n_x,n_y,n_z)
    bernstein_mesh=_bernstein_mesh(points,n_control_points)
    points_new=np.zeros_like(points)
    for p in prange(len(points)):
        for i in prange(n_x):
            for j in prange(n_y):
                for k in prange(n_z):
                    points_new[p,0]+=bernstein_mesh[p,i,j,k]*control_x[i,j,k]
                    points_new[p,1]+=bernstein_mesh[p,i,j,k]*control_y[i,j,k]
                    points_new[p,2]+=bernstein_mesh[p,i,j,k]*control_z[i,j,k]
    return points_new

@jit(nopython=True)                             
def _control_points(n_control_points,array_mu_x,array_mu_y,array_mu_z):   
    x = np.linspace(0, 1, n_control_points[0])
    y = np.linspace(0, 1, n_control_points[1])
    z = np.linspace(0, 1, n_control_points[2])
    x_coords, y_coords, z_coords = numba_meshgrid(x, y, z)
    x_coords=x_coords+array_mu_x
    y_coords=y_coords+array_mu_y
    z_coords=z_coords+array_mu_z
    return np.concatenate((x_coords.ravel(),y_coords.ravel(),z_coords.ravel()))



class CPFFD():
    def __init__(self, n_control_points=None,box_length=None,box_origin=None):
        if n_control_points is None:
            n_control_points = [2, 2, 2]

        if box_origin is None:
            box_origin=np.array([0., 0., 0.])
        
        if box_length is None:
            box_length=np.array([1., 1., 1.])

        self.box_length=box_length
        self.box_origin=box_origin
        self.n_control_points=n_control_points    
        self.array_mu_x = np.zeros(n_control_points)
        self.array_mu_y = np.zeros(n_control_points)
        self.array_mu_z = np.zeros(n_control_points)

    def f(self,h):
        return _f(h,self.n_control_points)

    def control_points(self):
        return _control_points(self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z)

    def bernstein_mesh(self,points):
        return _bernstein_mesh(points,self.n_control_points)


    def classic_ffd(self,points):
        return _classic_ffd(points,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z)

    def constrained_ffd(self,points,A,b,M,indices_x,indices_c):
        return _constrained_ffd(A,b,M,indices_x,indices_c,points,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z)



if __name__=="__main__":
    np.random.seed(0)
    seed=0
    n_x=3
    n_y=3
    n_z=3
    n_const=3
    x=np.random.rand(10000,3)
    b=np.random.rand(n_const)
    indices_c=np.random.choice(n_x*n_y*n_z,np.random.randint(n_x*n_y*n_z),replace=False)
    indices_c.sort()
    indices_x=np.random.choice(len(x.reshape(-1)),np.random.randint(len(x.reshape(-1))),replace=False)
    indices_x.sort()
    num_axis=np.unique(indices_x%3)
    A=np.random.rand(n_const,len(indices_x.reshape(-1))).reshape(n_const,-1)
    M=generate_random_sdp(len(indices_c)*len(num_axis))
    ffd=CPFFD((n_x,n_y,n_z))
    ffd.array_mu_x=np.random.rand(*ffd.array_mu_x.shape)
    ffd.array_mu_y=np.random.rand(*ffd.array_mu_y.shape)
    ffd.array_mu_z=np.random.rand(*ffd.array_mu_z.shape)
    print(b)
    x_2=ffd.constrained_ffd(x,A,b,M,indices_x,indices_c).reshape(-1)
    x=x.reshape(-1)
    print(np.linalg.norm(A@x_2[indices_x]-b)/np.linalg.norm(b))

