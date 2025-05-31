import torch
import torch.nn as nn
import models.flows as flows

class PFL(nn.Module):
    """
    The base VAE class.
    Can be used as a base class for VAE with normalizing flows.
    """

    def __init__(self, encoder, decoder, args):
        super(PFL, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        # extract model settings from args
        self.z_size = args.embed_dim
        self.input_size = args.embed_dim
        self.input_dim = args.embed_dim
        self.encoder_dim = args.embed_dim
        self.decoder_dim = args.embed_dim
        self.is_cuda = True
        self.L = args.sample_num
        

        self.p_mu = nn.Sequential(
                    nn.Linear(self.decoder_dim, self.input_dim),
                    #nn.Sigmoid() 
                )
        self.log_det_j = 0.


    def reparameterize(self, mu, var, mode = None):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        if mode == "train":
            std = var.sqrt()
            eps = torch.randn_like(std)
            z = eps * std + mu
            return z
        else:
            sample_bias_list = []
            for i in range(self.L):
                std = var.sqrt()
                eps = torch.randn_like(std)
                z = eps * std + mu
                sample_bias_list.append(z)
            z = torch.stack(sample_bias_list, dim = 0)
            return z



    def encode(self, x):
        mu, var = self.encoder(x)
        return mu, var

    def decode(self, z):
        h = self.decoder(z)
        x_mean = self.p_mu(h)

        return x_mean.view(-1, self.input_size)

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k = z.
        """

        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, self.log_det_j, z, z  # the last three outputs are useless; only to match outputs of flowVAE


class PlanarPFL(PFL):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, encoder, decoder, args):
        super(PlanarPFL, self).__init__(encoder, decoder, args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.encoder_dim, self.num_flows * self.z_size)
        self.amor_w = nn.Linear(self.encoder_dim, self.num_flows * self.z_size)
        self.amor_b = nn.Linear(self.encoder_dim, self.num_flows)


        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)
    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = x.size(0)

        mu, var = self.encoder(x)
        u = self.amor_u(x).view(batch_size, self.num_flows, self.z_size, 1)
        w = self.amor_w(x).view(batch_size, self.num_flows, 1, self.z_size)
        b = self.amor_b(x).view(batch_size, self.num_flows, 1, 1)

        return mu, var, u, w, b

    def forward(self, x, mode = None):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        if self.is_cuda:
            self.log_det_j = torch.zeros([x.shape[0]]).cuda()
        else:
            self.log_det_j = torch.zeros([x.shape[0]])
        
        z_mu, z_var, u, w, b = self.encode(x)
        z_0 = self.reparameterize(z_mu, z_var, mode = mode)
        if mode == "train":
            # Normalizing flows
            log_det_j = self.log_det_j
            z_list = []
            z_list.append(z_0.clone())
            for k in range(self.num_flows):
                flow_k = getattr(self, 'flow_' + str(k))
                z_k, log_det_jacobian = flow_k(z_list[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
                z_list.append(z_k)
                log_det_j = log_det_j + log_det_jacobian
            x_mean = self.decode(z_list[-1])
            return x_mean, z_mu, z_var, log_det_j, z_list[0], z_list[-1]
        else:
            z0_list = []
            zk_list = []
            log_det_j_list = []
            for i in range(z_0.shape[0]):
                log_det_j = self.log_det_j
                z_list = []
                z_list.append(z_0[i,:,:].clone())
                for k in range(self.num_flows):
                    flow_k = getattr(self, 'flow_' + str(k))
                    z_k, log_det_jacobian = flow_k(z_list[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
                    z_list.append(z_k)
                    log_det_j = log_det_j + log_det_jacobian
                z0_list.append(z_list[0])
                zk_list.append(z_list[-1])
                x_mean = self.decode(z_list[-1])
                log_det_j_list.append(log_det_j)
            z_k_final = torch.cat(zk_list, dim = 0)
            return x_mean, z_k_final, z_k_final, z_k_final, z_k_final, z_k_final

class PlanarPFL_state(PFL):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, encoder, decoder, args):
        super(PlanarPFL_state, self).__init__(encoder, decoder, args)

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        flow = flows.Planar
        self.num_flows = args.num_flows

        self.amor_u = nn.Parameter(torch.randn(1, self.num_flows, self.z_size, 1))
        self.amor_w = nn.Parameter(torch.randn(1, self.num_flows, 1, self.z_size))
        self.amor_b = nn.Parameter(torch.randn(1, self.num_flows, 1, 1))

        self.state = nn.Parameter(torch.randn(1, self.encoder_dim))

        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

        

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        mu, var = self.encoder(x)
        u = self.amor_u
        w = self.amor_w
        b = self.amor_b

        return mu, var, u, w, b
    

    def forward(self, x, mode = None):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        x = self.state

        if self.is_cuda:
            self.log_det_j = torch.zeros([x.shape[0]]).cuda()
        else:
            self.log_det_j = torch.zeros([x.shape[0]])
        z_mu, z_var, u, w, b = self.encode(x)
        
        z_0 = self.reparameterize(z_mu, z_var, mode = mode)
    
        if mode == "train":
            # Normalizing flows
            log_det_j = self.log_det_j
            z_list = []
            z_list.append(z_0.clone())
            for k in range(self.num_flows):
                flow_k = getattr(self, 'flow_' + str(k))
                z_k, log_det_jacobian = flow_k(z_list[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
                z_list.append(z_k)
                log_det_j = log_det_j + log_det_jacobian

            x_mean = self.decode(z_list[-1])
            return x_mean, z_mu, z_var, log_det_j, z_list[0], z_list[-1]
        else:
            z0_list = []
            zk_list = []
            log_det_j_list = []
            for i in range(z_0.shape[0]):
                log_det_j = self.log_det_j
                z_list = []
                z_list.append(z_0[i,:,:].clone())
                for k in range(self.num_flows):
                    flow_k = getattr(self, 'flow_' + str(k))
                    z_k, log_det_jacobian = flow_k(z_list[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
                    z_list.append(z_k)
                    log_det_j = log_det_j + log_det_jacobian
                z0_list.append(z_list[0])
                zk_list.append(z_list[-1])
                x_mean = self.decode(z_list[-1])
                log_det_j_list.append(log_det_j)
            z_k_final = torch.cat(zk_list, dim = 0)
            return x_mean, z_k_final, z_k_final, z_k_final, z_k_final, z_k_final
        
