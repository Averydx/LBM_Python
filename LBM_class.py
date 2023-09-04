import numpy as np
from numpy.typing import NDArray
from sklearn import preprocessing as pre
from typing import Tuple


class LBM: 
    cxs: NDArray[np.int_]
    cys: NDArray[np.int_]
    weights: NDArray[np.float_]
    idxs : NDArray[np.int_]
    F: NDArray
    cylinder: NDArray
    cylinder_xy: Tuple[int,int]
    mouse_pos: Tuple[int,int]

    Nx : int
    Ny : int
    tau: float


    def __init__(self,Nx,Ny,tau,cylinder_xy) -> None:
        rho0        = 400    # average density
        self.tau         = tau    # collision timescale
        # Lattice speeds / weights
        NL = 9
        self.Nx = Nx
        self.Ny = Ny
        self.idxs = np.arange(NL)
        self.cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
        self.cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
        self.weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
        X, Y = np.meshgrid(range(self.Nx), range(self.Ny))
        self.cylinder_xy = cylinder_xy

        # Initial Conditions - flow to the right with some perturbations
        self.F = np.ones((self.Ny,self.Nx,NL)) + 0.01*np.random.randn(self.Ny,self.Nx,NL)
        self.F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/self.Nx*4))
        rho = np.sum(self.F,2)
        for i in self.idxs:
            self.F[:,:,i] *= rho0 / rho

        self.mouse_pos = (0,0)
        self.cylinder_location()
        
    def cylinder_location(self):
        self.cylinder = np.full((self.Ny,self.Nx),False)
        for y in range(self.Ny): 
            for x in range(self.Nx): 
                if self.distance(self.cylinder_xy[0], self.cylinder_xy[1], x , y) < 50: 
                    self.cylinder[y][x] = True
                if self.distance(self.mouse_pos[0],self.mouse_pos[1],x,y) < 5: 
                    self.cylinder[y][x] = True

    def distance(self,x1,y1,x2,y2): 
            return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def solve(self): 
            # Drift
        self.cylinder_location()
        for i, cx, cy in zip(self.idxs, self.cxs, self.cys):
            self.F[:,:,i] = np.roll(self.F[:,:,i], cx, axis=1)
            self.F[:,:,i] = np.roll(self.F[:,:,i], cy, axis=0)
        
        # Set reflective boundaries
        bndryF = self.F[self.cylinder,:]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
        
        # Calculate fluid variables
        rho = np.sum(self.F,2)
        ux  = np.sum(self.F*self.cxs,2) / rho
        uy  = np.sum(self.F*self.cys,2) / rho

        # Apply Collision
        Feq = np.zeros(self.F.shape)
        for i, cx, cy, w in zip(self.idxs, self.cxs, self.cys, self.weights):
            Feq[:,:,i] = rho*w* (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2)
        
        self.F += -(1.0/self.tau) * (self.F - Feq)
        
        # Apply boundary 
        self.F[self.cylinder,:] = bndryF

        ux[self.cylinder] = 0
        uy[self.cylinder] = 0
        speeds = np.sqrt(ux**2 + uy **2)
        return speeds