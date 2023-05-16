import numpy as np
from numba import jit,prange
from numba.extending import get_cython_function_address
import ctypes
import numpy as np
from tqdm import trange
import time

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "binom")
functype = ctypes.CFUNCTYPE(_dble, _dble,_dble)
binom = functype(addr)
#from scipy.special import binom

@jit(nopython=True)
def points_triangle_indexing(points,triangles):
    tmp=np.zeros((triangles.shape[0],triangles.shape[1],points.shape[1]))
    for i in range(triangles.shape[0]):
        for j in range(triangles.shape[1]):
            for k in range(points.shape[1]):
                    tmp[i,j,k]=points[triangles[i,j],k]
    return tmp

@jit(nopython=True)
def numba_concatenate(a,b):
    tmp=np.zeros((2,a.shape[0]))
    for i in range(a.shape[0]):
        tmp[0,i]=a[i]
        tmp[1,i]=b[i]
    return tmp
    


@jit(nopython=True)
def volume_x(points,triangles):
    mesh=points_triangle_indexing(points,triangles)
    n_triangles=len(triangles)
    s=0
    for i in range(n_triangles):
        x_sum=0
        for j in range(3):
            x_sum=x_sum+mesh[i,j,0]
        tmp=np.zeros((2,2),dtype=points.dtype)
        tmp[0,0]=mesh[i,1,1]-mesh[i,0,1]
        tmp[0,1]=mesh[i,1,2]-mesh[i,0,2]
        tmp[1,0]=mesh[i,2,1]-mesh[i,0,1]
        tmp[1,1]=mesh[i,2,2]-mesh[i,0,2]
        s=s+x_sum*np.linalg.det(tmp)/6
    return s


@jit(nopython=True)
def volume_y(points,triangles):
    mesh=points_triangle_indexing(points,triangles)
    n_triangles=len(triangles)
    s=0
    for i in range(n_triangles):
        y_sum=0
        for j in range(3):
            y_sum=y_sum+mesh[i,j,1]
        tmp=np.zeros((2,2))
        tmp[0,0]=mesh[i,1,2]-mesh[i,0,2]
        tmp[0,1]=mesh[i,1,0]-mesh[i,0,0]
        tmp[1,0]=mesh[i,2,2]-mesh[i,0,2]
        tmp[1,1]=mesh[i,2,0]-mesh[i,0,0]
        s=s+y_sum*np.linalg.det(tmp)/6
    return s


@jit(nopython=True)
def volume_z(points,triangles):
    mesh=points_triangle_indexing(points,triangles)
    n_triangles=len(triangles)
    s=0
    for i in range(n_triangles):
        z_sum=0
        for j in range(3):
            z_sum=z_sum+mesh[i,j,2]
        tmp=np.zeros((2,2),dtype=points.dtype)
        tmp[0,0]=mesh[i,1,0]-mesh[i,0,0]
        tmp[0,1]=mesh[i,1,1]-mesh[i,0,1]
        tmp[1,0]=mesh[i,2,0]-mesh[i,0,0]
        tmp[1,1]=mesh[i,2,1]-mesh[i,0,1]
        s=s+z_sum*np.linalg.det(tmp)/6
    return s


@jit(nopython=True)
def volume_tetra(points,triangles):
    mesh=points_triangle_indexing(points,triangles)
    M=np.zeros((triangles.shape[0],triangles.shape[1],4),dtype=points.dtype)
    n_triangles=len(triangles)
    s=0
    for i in range(n_triangles):
        for j in range(4):
            for k in range(3):
                M[i,k,j]=mesh[i,j,k]
            M[i,3,j]=1    
        s=s+np.linalg.det(M[i])/6
    return s

@jit(nopython=True)
def get_coeff_x(points,triangles):
    p=np.zeros(points.shape[0],dtype=points.dtype)
    mesh=points_triangle_indexing(points,triangles)
    n_triangles=len(triangles)
    for i in range(n_triangles):
        tmp=np.zeros((2,2))
        tmp[0,0]=mesh[i,1,1]-mesh[i,0,1]
        tmp[0,1]=mesh[i,1,2]-mesh[i,0,2]
        tmp[1,0]=mesh[i,2,1]-mesh[i,0,1]
        tmp[1,1]=mesh[i,2,2]-mesh[i,0,2]
        k=triangles[i]
        p[k[0]]=p[k[0]]+np.linalg.det(tmp)/6
        p[k[1]]=p[k[1]]+np.linalg.det(tmp)/6
        p[k[2]]=p[k[2]]+np.linalg.det(tmp)/6
    return p

@jit(nopython=True)
def get_coeff_y(points,triangles):
    p=np.zeros(points.shape[0],dtype=points.dtype)
    mesh=points_triangle_indexing(points,triangles)
    n_triangles=len(triangles)
    for i in range(n_triangles):
        tmp=np.zeros((2,2))
        tmp[0,0]=mesh[i,1,2]-mesh[i,0,2]
        tmp[0,1]=mesh[i,1,0]-mesh[i,0,0]
        tmp[1,0]=mesh[i,2,2]-mesh[i,0,2]
        tmp[1,1]=mesh[i,2,0]-mesh[i,0,0]
        k=triangles[i]
        p[k[0]]=p[k[0]]+np.linalg.det(tmp)/6
        p[k[1]]=p[k[1]]+np.linalg.det(tmp)/6
        p[k[2]]=p[k[2]]+np.linalg.det(tmp)/6
    return p

@jit(nopython=True)
def get_coeff_z(points,triangles):
    p=np.zeros(points.shape[0],dtype=points.dtype)
    mesh=points_triangle_indexing(points,triangles)
    n_triangles=len(triangles)
    for i in range(n_triangles):
        tmp=np.zeros((2,2))
        tmp[0,0]=mesh[i,1,0]-mesh[i,0,0]
        tmp[0,1]=mesh[i,1,1]-mesh[i,0,1]
        tmp[1,0]=mesh[i,2,0]-mesh[i,0,0]
        tmp[1,1]=mesh[i,2,1]-mesh[i,0,1]
        k=triangles[i]
        p[k[0]]=p[k[0]]+np.linalg.det(tmp)/6
        p[k[1]]=p[k[1]]+np.linalg.det(tmp)/6
        p[k[2]]=p[k[2]]+np.linalg.det(tmp)/6
    return p

@jit(nopython=True)
def get_coeff_x_tetra(points,triangles):
    p=np.zeros(points.shape[0],dtype=points.dtype)
    mesh=points_triangle_indexing(points,triangles)
    M=np.zeros((triangles.shape[0],triangles.shape[1],4),dtype=points.dtype)
    n_triangles=len(triangles)
    for i in range(n_triangles):
        for j in range(4):
            for k in range(3):
                M[i,k,j]=mesh[i,j,k]
            M[i,3,j]=1  
    for i in range(n_triangles):
        k=triangles[i]
        p[k[0]]=p[k[0]]+(-M[i,1,2]*M[i,2,1]+M[i,1,3]*M[i,2,1]+M[i,1,1]*M[i,2,2]-M[i,1,3]*M[i,2,2]-M[i,1,1]*M[i,2,3]+M[i,1,2]*M[i,2,3])*(1)/6
        p[k[1]]=p[k[1]]+(-M[i,1,2]*M[i,2,0]+M[i,1,3]*M[i,2,0]+M[i,1,0]*M[i,2,2]-M[i,1,3]*M[i,2,2]-M[i,1,0]*M[i,2,3]+M[i,1,2]*M[i,2,3])*(-1)/6
        p[k[2]]=p[k[2]]+(-M[i,1,1]*M[i,2,0]+M[i,1,3]*M[i,2,0]+M[i,1,0]*M[i,2,1]-M[i,1,3]*M[i,2,1]-M[i,1,0]*M[i,2,3]+M[i,1,1]*M[i,2,3])*(1)/6
        p[k[3]]=p[k[3]]+(-M[i,1,1]*M[i,2,0]+M[i,1,2]*M[i,2,0]+M[i,1,0]*M[i,2,1]-M[i,1,2]*M[i,2,1]-M[i,1,0]*M[i,2,2]+M[i,1,1]*M[i,2,2])*(-1)/6
    return p

@jit(nopython=True)
def get_coeff_y_tetra(points,triangles):
    p=np.zeros(points.shape[0],dtype=points.dtype)
    mesh=points_triangle_indexing(points,triangles)
    M=np.zeros((triangles.shape[0],triangles.shape[1],4),dtype=points.dtype)
    n_triangles=len(triangles)
    for i in range(n_triangles):
        for j in range(4):
            for k in range(3):
                M[i,k,j]=mesh[i,j,k]
            M[i,3,j]=1  
    for i in range(n_triangles):
        k=triangles[i]
        p[k[0]]=p[k[0]]+(-M[i,0,2]*M[i,2,1]+M[i,0,3]*M[i,2,1]+M[i,0,1]*M[i,2,2]-M[i,0,3]*M[i,2,2]-M[i,0,1]*M[i,2,3]+M[i,0,2]*M[i,2,3])*(-1)/6
        p[k[1]]=p[k[1]]+(-M[i,0,2]*M[i,2,0]+M[i,0,3]*M[i,2,0]+M[i,0,0]*M[i,2,2]-M[i,0,3]*M[i,2,2]-M[i,0,0]*M[i,2,3]+M[i,0,2]*M[i,2,3])*(1)/6
        p[k[2]]=p[k[2]]+(-M[i,0,1]*M[i,2,0]+M[i,0,3]*M[i,2,0]+M[i,0,0]*M[i,2,1]-M[i,0,3]*M[i,2,1]-M[i,0,0]*M[i,2,3]+M[i,0,1]*M[i,2,3])*(-1)/6
        p[k[3]]=p[k[3]]+(-M[i,0,1]*M[i,2,0]+M[i,0,2]*M[i,2,0]+M[i,0,0]*M[i,2,1]-M[i,0,2]*M[i,2,1]-M[i,0,0]*M[i,2,2]+M[i,0,1]*M[i,2,2])*(1)/6
    return p

@jit(nopython=True)
def get_coeff_z_tetra(points,triangles):
    p=np.zeros(points.shape[0])
    mesh=points_triangle_indexing(points,triangles)
    M=np.zeros((triangles.shape[0],triangles.shape[1],4))
    n_triangles=len(triangles)
    for i in range(n_triangles):
        for j in range(4):
            for k in range(3):
                M[i,k,j]=mesh[i,j,k]
            M[i,3,j]=1  
    for i in range(n_triangles):
        k=triangles[i]
        p[k[0]]=p[k[0]]+(-M[i,0,2]*M[i,1,1]+M[i,0,3]*M[i,1,1]+M[i,0,1]*M[i,1,2]-M[i,0,3]*M[i,1,2]-M[i,0,1]*M[i,1,3]+M[i,0,2]*M[i,1,3])*(1)/6
        p[k[1]]=p[k[1]]+(-M[i,0,2]*M[i,1,0]+M[i,0,3]*M[i,1,0]+M[i,0,0]*M[i,1,2]-M[i,0,3]*M[i,1,2]-M[i,0,0]*M[i,1,3]+M[i,0,2]*M[i,1,3])*(-1)/6
        p[k[2]]=p[k[2]]+(-M[i,0,1]*M[i,1,0]+M[i,0,3]*M[i,1,0]+M[i,0,0]*M[i,1,1]-M[i,0,3]*M[i,1,1]-M[i,0,0]*M[i,1,3]+M[i,0,1]*M[i,1,3])*(1)/6
        p[k[3]]=p[k[3]]+(-M[i,0,1]*M[i,1,0]+M[i,0,2]*M[i,1,0]+M[i,0,0]*M[i,1,1]-M[i,0,2]*M[i,1,1]-M[i,0,0]*M[i,1,2]+M[i,0,1]*M[i,1,2])*(-1)/6
    return p

@jit(nopython=True)
def generate_random_sdp(n):
    A=np.random.rand(n,n)
    A=A@A.T
    A=A+n*np.eye(n)
    return A

@jit(nopython=True)
def numba_mean_0(A):
    m=A.shape[0]
    n=A.shape[1]
    mean=np.zeros(n,dtype=A.dtype)
    for i in range(m):
        mean=mean+A[i]
    mean=mean/m
    return mean

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
    n_x=n_control_points[0]
    n_y=n_control_points[1]
    n_z=n_control_points[1]
    bern=np.zeros((n_points,n_x,n_y,n_z))
    for p in range(n_points):
        for i in range(n_control_points[0]):
            for j in range(n_control_points[1]):
                for k in range(n_control_points[2]):
                    bern[p,i,j,k]=binom(float(n_x-1),float(i))*points[p,0]**i*(1-points[p,0])**(n_x-1-i)*binom(float(n_y-1),float(j))*points[p,1]**j*(1-points[p,1])**(n_y-1-j)*binom(float(n_z-1),float(k))*points[p,2]**k*(1-points[p,2])**(n_z-1-k)
    return bern
@jit(nopython=True,parallel=True) 
def _constrained_ffd(A,b,M,indices_x,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z):
    points_new=_classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    n_x=n_control_points[0]
    n_y=n_control_points[1]
    n_z=n_control_points[1]
    b_tmp=A@(points_new.reshape(-1)[indices_x])
    new_array=np.concatenate((array_mu_x.copy(),array_mu_y.copy(),array_mu_z.copy())).reshape(3,-1)
    delta_b=b-b_tmp
    num_axis=np.unique(indices_x%3)
    bmesh=_bernstein_mesh(points,n_control_points)
    A_c=np.zeros((A.shape[0],len(num_axis),len(indices_c)),dtype=points.dtype)
    for i in prange(A.shape[0]):
        for i_g in prange(len(indices_x)):
            for i_h in prange(len(indices_c)):
                a,b,c=_f(indices_c[i_h],n_control_points)
                d,=np.where(num_axis==(indices_x[i_g]%3))[0]
                A_c[i,d,i_h]+=A[i,i_g]*bmesh[(indices_x[i_g]//3),a,b,c]
    A_c=A_c.reshape(A.shape[0],-1)
    delta_q=np.linalg.solve(M,A_c.T)@(np.linalg.solve(A_c@np.linalg.solve(M,A_c.T),delta_b))
    delta_q=delta_q.reshape(len(num_axis),len(indices_c))
    for i in range(len(num_axis)):
        for j in range(len(indices_c)):
            new_array[num_axis[i],indices_c[j]]+=delta_q[i,j]
    
    new_array_x=new_array[0].reshape(n_x,n_y,n_z)
    new_array_y=new_array[1].reshape(n_x,n_y,n_z)
    new_array_z=new_array[2].reshape(n_x,n_y,n_z)
    tmp=_classic_ffd(points,n_control_points,new_array_x,new_array_y,new_array_z)
    return tmp,new_array_x,new_array_y,new_array_z


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
    points_new=np.zeros_like(points,dtype=points.dtype)
    for p in range(len(points)):
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
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

@jit(nopython=True)                             
def _barycenter_preserving_ffd(points,M,n_control_points,array_mu_x,array_mu_y,array_mu_z):
    indices_c=np.arange(np.prod(n_control_points))
    return  _barycenter_preserving_ffd_adv(points,M,n_control_points,array_mu_x,array_mu_y,array_mu_z,indices_c)

@jit(nopython=True)                             
def _barycenter_preserving_ffd_adv(points,M,n_control_points,array_mu_x,array_mu_y,array_mu_z,indices_c):
    n_points=len(points.reshape(-1,3))
    indices_x=np.arange(n_points*3)
    A=np.zeros((3,n_points*3),dtype=points.dtype)
    for i in range(3):
        for j in range(i,n_points*3,3):
            A[i,j]=1/n_points     
    b=numba_mean_0(points)
    tmp=_constrained_ffd(A,b,M,indices_x,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    return tmp




@jit(nopython=True)                             
def _volume_preserving_ffd_adv(points,M,n_control_points,array_mu_x,array_mu_y,array_mu_z,triangles,indices_c):
    array_x_bak=array_mu_x.copy()
    array_y_bak=array_mu_y.copy()
    array_z_bak=array_mu_z.copy()
    n_points=len(points.reshape(-1,3))
    points_deformed=_classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    Vol_true=volume_x(points,triangles)
    Vol_def=volume_x(points_deformed,triangles)
    ax=1/3*(Vol_true-Vol_def)
    ay=1/3*(Vol_true-Vol_def)
    az=1/3*(Vol_true-Vol_def)
    indices_x=np.unique(3*np.arange(n_points))
    Ax=get_coeff_x(points_deformed,triangles).reshape(1,-1)
    bx=Vol_def+ax
    tmp,new_array_x,new_array_y,new_array_z=_constrained_ffd(Ax,bx,M,indices_x,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    array_mu_x=new_array_x
    array_mu_y=new_array_y
    array_mu_z=new_array_z
    points_deformed=_classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    indices_y=3*np.arange(n_points)+1
    Ay=get_coeff_y(points_deformed,triangles).reshape(1,-1)
    by=bx+ay
    tmp,new_array_x,new_array_y,new_array_z=_constrained_ffd(Ay,by,M,indices_y,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    array_mu_x=new_array_x
    array_mu_y=new_array_y
    array_mu_z=new_array_z
    points_deformed=_classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    indices_z=3*np.arange(n_points)+2
    Az=get_coeff_z(points_deformed,triangles).reshape(1,-1)
    bz=by+az
    tmp,new_array_x,new_array_y,new_array_z=_constrained_ffd(Az,bz,M,indices_z,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    array_mu_x=new_array_x
    array_mu_y=new_array_y
    array_mu_z=new_array_z
    return tmp,array_mu_x-array_x_bak,array_mu_y-array_y_bak,array_mu_z-array_z_bak

@jit(nopython=True)                             
def _volume_preserving_ffd(points,M,n_control_points,array_mu_x,array_mu_y,array_mu_z,triangles):
    indices_c=np.arange(n_control_points[0]*n_control_points[1]*n_control_points[2])
    return _volume_preserving_ffd_adv(points,M,n_control_points,array_mu_x,array_mu_y,array_mu_z,triangles,indices_c)

@jit(nopython=True)                             
def _double_preserving_ffd_adv(points,M,n_control_points,array_mu_x,array_mu_y,array_mu_z,triangles,indices_c):
    array_x_bak=array_mu_x.copy()
    array_y_bak=array_mu_y.copy()
    array_z_bak=array_mu_z.copy()
    n_points=len(points.reshape(-1,3))
    bb=numba_mean_0(points)
    points_deformed=_classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    Vol_true=volume_tetra(points,triangles)
    Vol_def=volume_tetra(points_deformed,triangles)
    ax=1/3*(Vol_true-Vol_def)
    ay=1/3*(Vol_true-Vol_def)
    az=1/3*(Vol_true-Vol_def)
    indices_x=3*np.arange(n_points)
    Avx=get_coeff_x_tetra(points_deformed,triangles).reshape(-1)
    Abx=1/n_points*np.ones(n_points,dtype=points.dtype).reshape(-1)
    Ax=numba_concatenate(Avx,Abx)
    bbx=bb[0]
    bvx=Vol_def+ax
    bx=np.array([bvx,bbx])
    tmp,new_array_x,new_array_y,new_array_z=_constrained_ffd(Ax,bx,M,indices_x,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    array_mu_x=new_array_x
    array_mu_y=new_array_y
    array_mu_z=new_array_z
    points_deformed=_classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    indices_y=3*np.arange(n_points)+1
    Avy=get_coeff_y_tetra(points_deformed,triangles).reshape(-1)
    Aby=1/n_points*np.ones(n_points,dtype=points.dtype).reshape(-1)
    Ay=numba_concatenate(Avy,Aby)
    bby=bb[1]
    bvy=bvx+ay
    by=np.array([bvy,bby])
    tmp,new_array_x,new_array_y,new_array_z=_constrained_ffd(Ay,by,M,indices_y,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    array_mu_x=new_array_x
    array_mu_y=new_array_y
    array_mu_z=new_array_z
    points_deformed=_classic_ffd(points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    indices_z=3*np.arange(n_points)+2
    Avz=get_coeff_z_tetra(points_deformed,triangles).reshape(-1)
    Abz=1/n_points*np.ones(n_points,dtype=points.dtype).reshape(-1)
    Az=numba_concatenate(Avz,Abz)
    bbz=bb[2]
    bvz=bvy+az
    bz=np.array([bvz,bbz])
    tmp,new_array_x,new_array_y,new_array_z=_constrained_ffd(Az,bz,M,indices_z,indices_c,points,n_control_points,array_mu_x,array_mu_y,array_mu_z)
    array_mu_x=new_array_x
    array_mu_y=new_array_y
    array_mu_z=new_array_z
    return tmp,array_mu_x-array_x_bak,array_mu_y-array_y_bak,array_mu_z-array_z_bak




@jit(nopython=True)
def _g(i,j,k,n_control_points):
        return i*n_control_points[1]*n_control_points[2]+j*n_control_points[2]+k


class CPFFD():
    def __init__(self, n_control_points=None,box_length=None,box_origin=None):
        if n_control_points is None:
            n_control_points = np.array([2, 2, 2])

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

    def mesh_to_local_space(self, mesh):
        return (mesh-self.box_origin)/self.box_length
        
    def mesh_to_global_space(self,mesh):
        return mesh*self.box_length+self.box_origin

    def f(self,h):
        return _f(h,self.n_control_points)

    def control_points(self):
        return _control_points(self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z)

    def bernstein_mesh(self,points):
        return _bernstein_mesh(points,self.n_control_points)
    
    def g(self,i,j,k):
        return _g(i,j,k,*self.n_control_points)
    
    def classic_ffd(self,points):
        return _classic_ffd(points,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z)

    def constrained_ffd(self,points,A,b,M,indices_x,indices_c):
        points,new_array_x,new_array_y,new_array_z=_constrained_ffd(A,b,M,indices_x,indices_c,points,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z)
        self.array_mu_x=self.array_mu_x+new_array_x
        self.array_mu_y=self.array_mu_y+new_array_y
        self.array_mu_z=self.array_mu_z+new_array_z
        return points

class BPFFD(CPFFD):
    def __init__(self,n_control_points=None,box_length=None,box_origin=None):
        super().__init__(n_control_points,box_length,box_origin)
    
    def barycenter_ffd(self,points,M):
        points,new_array_x,new_array_y,new_array_z=_barycenter_preserving_ffd(points,M,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z)
        self.array_mu_x=self.array_mu_x+new_array_x
        self.array_mu_y=self.array_mu_y+new_array_y
        self.array_mu_z=self.array_mu_z+new_array_z
        return points

    def barycenter_ffd_adv(self,points,M,points_c):
        points,new_array_x,new_array_y,new_array_z=_barycenter_preserving_ffd_adv(points,M,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z,points_c)
        self.array_mu_x=self.array_mu_x+new_array_x
        self.array_mu_y=self.array_mu_y+new_array_y
        self.array_mu_z=self.array_mu_z+new_array_z
        return points



class VPFFD(CPFFD):
    def __init__(self,n_control_points=None,box_length=None,box_origin=None):
        super().__init__(n_control_points,box_length,box_origin)
    
    def volume_ffd(self,points,M,triangles):
        points,new_array_x,new_array_y,new_array_z=_volume_preserving_ffd(points,M,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z,triangles)
        self.array_mu_x=self.array_mu_x+new_array_x
        self.array_mu_y=self.array_mu_y+new_array_y
        self.array_mu_z=self.array_mu_z+new_array_z
        return points
    
    def volume_ffd_adv(self,points,M,triangles,indices_c):
        points,new_array_x,new_array_y,new_array_z=_volume_preserving_ffd_adv(points,M,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z,triangles,indices_c)
        self.array_mu_x=self.array_mu_x+new_array_x
        self.array_mu_y=self.array_mu_y+new_array_y
        self.array_mu_z=self.array_mu_z+new_array_z
        return points

class DPFFD(CPFFD):
    def __init__(self,n_control_points=None,box_length=None,box_origin=None):
        super().__init__(n_control_points,box_length,box_origin)

    
    def double_ffd_adv(self,points,M,triangles,indices_c):
        points,new_array_x,new_array_y,new_array_z=_double_preserving_ffd_adv(points,M,self.n_control_points,self.array_mu_x,self.array_mu_y,self.array_mu_z,triangles,indices_c)
        self.array_mu_x=self.array_mu_x+new_array_x
        self.array_mu_y=self.array_mu_y+new_array_y
        self.array_mu_z=self.array_mu_z+new_array_z
        return points




if __name__=="__main__":
    np.random.seed(0)
    seed=0
    n_x=3
    n_y=3
    n_z=3
    n_points=500000
    n_const=1
    x=np.random.rand(n_points,3)
    b=np.random.rand(n_const)
    indices_c=np.random.choice(n_x*n_y*n_z,np.random.randint(low=1,high=n_x*n_y*n_z),replace=False)
    indices_c.sort()
    indices_x=np.random.choice(len(x.reshape(-1)),np.random.randint(low=1,high=len(x.reshape(-1))),replace=False)
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

