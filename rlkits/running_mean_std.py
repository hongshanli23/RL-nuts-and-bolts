try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import numpy as np

class RunningMeanStd:
    def __init__(self, epsilon=1e-2, shape=(), default_clip_range=np.inf):
        self._sum = np.zeros(shape=shape, dtype=np.float64)
        self._sumsq = np.full(shape=shape, fill_value=epsilon, 
                              dtype=np.float64)
        self._count = np.array(epsilon, dtype=np.float64)
        
        self.shape = shape
        self.epsilon = epsilon
        self.default_clip_range = default_clip_range
    
    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        
        addvec = np.concatenate(
            [
             x.sum(axis=0).ravel(), 
             np.square(x).sum(axis=0).ravel(),
             np.array([len(x)], dtype='float64')
            ])
        
        totalvec = np.zeros(n*2 + 1, 'float64')
        if MPI is not None:
            MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        else:
            totalvec = addvec
        
        self._sum += totalvec[0:n].reshape(self.shape)
        self._sumsq += totalvec[n:2*n].reshape(self.shape)
        self._count += totalvec[2*n]
        return
    
    @property
    def mean(self):
        return self._sum / self._count
    
    @property
    def std(self):
        return np.sqrt(
            np.maximum(self._sumsq / self._count - np.square(self.mean),
                       self.epsilon))
    
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v- self.mean)/self.std, -clip_range, clip_range)
    
    def denormalize(self, v):
        return self.mean + v*self.std
    
    