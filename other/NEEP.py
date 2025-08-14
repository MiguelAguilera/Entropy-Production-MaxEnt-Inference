import torch, sys
sys.path.insert(0, '..')
import observables 
from utils import numpy_to_torch, torch_to_numpy

class DatasetNEEP(observables.Dataset):    
    def get_objective(self, theta): # Return objective value for parameters theta
        if self.nsamples == 0:
            return float('nan')

        theta = numpy_to_torch(theta)
        th_g_max, norm_const, _ = self._get_tilted_values(theta)
        Z                   = norm_const * torch.exp( th_g_max )
        return float( theta @ self.g_mean - Z + 1 )

    def get_gradient(self, theta):
        theta = numpy_to_torch(theta)
        th_g_max, norm_const, _ = self._get_tilted_values(theta)
        Z                   = norm_const * torch.exp( th_g_max ) 
        return self.g_mean - Z*self.get_tilted_mean(theta)  # Gradient of the objective function
