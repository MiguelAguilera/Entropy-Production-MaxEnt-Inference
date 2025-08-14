import torch, sys
sys.path.insert(0, '..')
import observables 
from utils import numpy_to_torch, torch_to_numpy

class DatasetNEEP(observables.Dataset):

    def get_Z(self, theta):
        theta = numpy_to_torch(theta)
        th_g_max, norm_const, _ = self._get_tilted_values(theta)
        Z = norm_const * torch.exp( th_g_max )
        # if torch.isinf(Z):
        #     Z = 10e20
        return Z


    def get_objective(self, theta): # Return objective value for parameters theta
        if self.nsamples == 0:
            return float('nan')

        theta = numpy_to_torch(theta)
        return float( theta @ self.g_mean - self.get_Z(theta) + 1 )


    def get_gradient(self, theta):
        grad = self.g_mean - self.get_Z(theta)*self.get_tilted_mean(theta)  # Gradient of the objective function
        return grad
