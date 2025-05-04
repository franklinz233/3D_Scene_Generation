import numpy as np
from .base import CameraPathBase


class Orbit(CameraPathBase):
    
    def camera_target_up(self):
        t = np.linspace(0, 1, self.nframe)
        # rotation angles at each frame
        theta = 2 * np.pi * t * self.nframe 
        # try not to change y (up-down for floor and sky)
        pos = np.zeros((self.nframe,3))
        x = np.sin(theta)
        y = np.zeros_like(x)
        z = np.cos(theta)
        targets = np.vstack([x,y,z]).T
        camera_ups = np.array([[0,-1.,0]]).repeat(self.nframe,axis=0)
        # camera_triples
        cameras = []
        for i in range(self.nframe):
            cameras.append((pos[i],targets[i],camera_ups[i]))
        return cameras
    
