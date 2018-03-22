from sympy import *
from sympy.matrices import *
import operator, functools
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from furlib.alpha import *

def euler_angles(H):
    """
    Calcula los ángulos de Euler ZXZ a partir de la matriz de transformación 
    homogénea H.
    """
    R = H[:3,:3]
    r33,r13,r23,r31,r32,r11,r12 = R[2,2],R[0,2],R[1,2],R[2,0],R[2,1],R[0,0],R[0,1]
    if r23!=0 and r13!=0:
        theta = atan2(sqrt(1-r33**2),r33)
        phi = atan2(r23,r13)
        psi = atan2(r32,-r31)
    elif r33==1:
        theta = 0
        phi = 0
        psi = atan2(-r12,r11)
    elif r33==-1:
        theta = pi
        psi = 0
        phi = atan2(-r12,-r11)
    else:
        theta = atan2(sqrt(1-r33**2),r33)
        phi = atan2(r23,r13)
        psi = atan2(r32,-r31)
    return phi.evalf(),theta.evalf(),psi.evalf()

def dhs(a,alpha,d,theta):
    """
    Calcula la matriz de Denavit-Hartenberg de manera simbólica
    """
    M = Matrix([[cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha),a*cos(theta)],
                  [sin(theta),cos(theta)*cos(alpha),-cos(theta)*sin(alpha),a*sin(theta)],
                  [0,sin(alpha),cos(alpha),d],
                  [0,0,0,1]])
    return M


class Manipulator(object):
    """
    Define un manipulador dados los parámetros de DH.
    """
    def __init__(self,*args):
        self.Ts = [] # Transformation matrices i to i-1
        self.type = [] # Joint type -> "r" revolute, "p" prismatic
        for k in args:
            self.Ts.append(dhs(k[0],k[1],k[2],k[3])) # Compute Ti->i-1
            if len(k)>4:
                self.type.append(k[4])
            else:
                self.type.append('r')
        self.dof = len(args) # Degree of freedom
    
    def z(self,i):
        """
        z-dir of every i-Frame
        """
        if i == 0: return Matrix([[0],[0],[1]])
        MTH = eye(4)
        for k in range(i):
            MTH = MTH*self.Ts[k]
        return MTH[:3,2]
    
    def p(self,i):
        """
        Position for every i-Frame
        """
        if i == 0: return Matrix([[0],[0],[0]])
        MTH = eye(4)
        for k in range(i):
            MTH = MTH*self.Ts[k]
        return MTH[:3,3]
    
    @property
    def J(self):
        """
        Jacobian matrix
        """
        n = self.dof
        M_ = zeros(6,n)
        for i in range(self.dof):
            if self.type[i]=='r':
                jp = self.z(i).cross(self.p(n) - self.p(i))
                jo = self.z(i)
            else:
                jp = self.z(i)
                jo = zeros(3,1)
            jp = jp.col_join(jo)
            M_[:,i] = jp
        return simplify(M_)
    
    @property
    def T(self):
        """ 
        T_n^0 
        Homogeneous transformation matrix of N-Frame respect to Base-Frame
        """
        return simplify(functools.reduce(operator.mul, self.Ts))
        
    def Ti_0(self,i):
        return simplify(functools.reduce(operator.mul, self.Ts[:i+1]))
        
    def plot_diagram(self,vals):
        #return None
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        Ts = self.Ts
        points = []
        Ti_0 = []
        points.append(zeros(1,3))
        for i in range(self.dof):
            Ti_0.append(self.Ti_0(i).subs(vals))
            points.append((self.Ti_0(i)[:3,3]).subs(vals))
        
        X = [k[0] for k in points]
        Y = [k[1] for k in points]
        Z = [k[2] for k in points]
        ax.plot(X,Y,Z, "o-", color="#778877", lw=3)
        ax.plot([0],[0],[0], "mo", markersize=6)
        ax.set_axis_off()
        ax.view_init(90,0)
        
        px,py,pz = float(X[-1]),float(Y[-1]),float(Z[-1])
        dim = max([px,py,pz])
        
        self.draw_uvw(eye(4),ax, dim)
        for T in Ti_0:
            self.draw_uvw(T, ax, dim)
            
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)
        ax.set_zlim(-dim, dim)
    
    def draw_uvw(self,H,ax,sz=1):
        u = H[:3,0]
        v = H[:3,1]
        w = H[:3,2]
        o = H[:3,3]
        L = sz/5
        ax.quiver(o[0],o[1],o[2],u[0],u[1],u[2],color="r", length=L)
        ax.quiver(o[0],o[1],o[2],v[0],v[1],v[2],color="g", length=L)
        ax.quiver(o[0],o[1],o[2],w[0],w[1],w[2],color="b", length=L)
        

def ea2htm(phi,theta,psi):
    """
    Calculate HTM from ZXZ Euler Angles.
    """
    return htmr(phi)*htmr(theta,"X")*htmr(psi)

def htmr(t,axis="z"):
    """
    Calculate the homogeneous transformation matrix of a rotation 
    respect to x,y or z axis.
    """
    from sympy import sin,cos,tan
    if axis in ("z","Z",3):
        M = Matrix([[cos(t),-sin(t),0,0],
                  [sin(t),cos(t),0,0],
                  [0,0,1,0],
                  [0,0,0,1]])
    elif axis in ("y","Y",2):
        M = Matrix([[cos(t),0,sin(t),0],
                  [0,1,0,0],
                  [-sin(t),0,cos(t),0],
                  [0,0,0,1]])
    elif axis in ("x","X",1):
        M = Matrix([[1,0,0,0],
                  [0,cos(t),-sin(t),0,],
                  [0,sin(t),cos(t),0],
                  [0,0,0,1]])
    else:
        return eye(4)
    return M


def htmt(dx,dy,dz):
    """
    Calculate the homogeneous transformation matrix of a traslation  
    """
    M = Matrix([[1,0,0,dx],
                [0,1,0,dy],
                [0,0,1,dz],
                [0,0,0,1]])
    return M
    
    
if __name__=="__main__":
    rr = Manipulator((0,pi/2,0,t1),(0,0,d2,0))
    rr.plot_diagram({t1:pi/2,d2:100})
    plt.show()
