import torch
import torch.nn as nn
import torch.nn.functional as F

class Planar(nn.Module):

    def __init__(self):

        super(Planar, self).__init__()
        self.h = nn.Tanh()

    def forward(self, z, u, w, b):
        """
        Computes the following transformation:
        z' = z + u h( w^T z + b)
        Input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        # Equation (10)
        z = z.unsqueeze(2)
        prod = torch.bmm(w, z) + b
        f_z = z + u * self.h(prod) # this is a 3d vector
        f_z = f_z.squeeze(2) # this is a 2d vector

        # compute logdetJ
        # Equation (11)
        psi = w * (1 - self.h(prod) ** 2)  # w * h'(prod)
        # Equation (12)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u))+ 1e-5)
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return f_z, log_det_jacobian


    
