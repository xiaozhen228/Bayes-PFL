import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import log_normal_dist


class Orthogonal_Loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Orthogonal_Loss, self).__init__()
        self.epsilon = epsilon
    
    def compute_orthogonal_loss(self, embeddings):
        B, L, C = embeddings.shape
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        cosine_sim_matrix = torch.einsum('blc,bkc->blk', embeddings_norm, embeddings_norm)
        cosine_sim_squared = cosine_sim_matrix ** 2
        eye_mask = torch.eye(L, device=embeddings.device).unsqueeze(0)
        cosine_sim_squared = cosine_sim_squared * (1 - eye_mask)
        cosine_loss = cosine_sim_squared.mean()
        return cosine_loss
    def forward(self, embeddings, args):
        Loss_noraml_text = self.compute_orthogonal_loss(embeddings[:, 0:args.prompt_num,:])
        Loss_abnormal_text = self.compute_orthogonal_loss(embeddings[:,args.prompt_num:,:])
        orthogonal_loss = Loss_noraml_text + Loss_abnormal_text
        return orthogonal_loss

class FocalLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(FocalLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, pred, gt, gamma=2.0, alpha=1, mask_ratio=1.0):
        gt_one_hot = F.one_hot(gt.long(), num_classes=2).permute(0, 3, 1, 2).float()  
        pt = (pred * gt_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** gamma  # (1 - p_t)^γ
        focal_loss = -alpha * focal_weight * torch.log(pt + self.epsilon)
        fg_mask = gt > 0
        bg_mask = ~fg_mask
        fg_loss = focal_loss * fg_mask.float()
        bg_loss = focal_loss * bg_mask.float()
        fg_pixels = fg_mask.sum().float()
        bg_pixels = bg_mask.sum().float()
        fg_loss_final = fg_loss.sum() / (fg_pixels + self.epsilon)
        bg_loss_final = bg_loss.sum() / (bg_pixels + self.epsilon)
        loss = fg_loss_final  + bg_loss_final
        return loss


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def compute_loss(self, pred, target):
        target_sum = torch.sum(target)
        intersection = torch.sum(pred * target) 
        union = torch.sum(pred) + target_sum 
        dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
        loss = 1 - dice
        return loss 

    def forward(self, pred, target):
 
        target = target.float()  
        loss_f = self.compute_loss(pred[:, 1, :, :],  target.clone())
        loss_b = self.compute_loss(pred[:, 0, :, :],  1 - target)
        #loss = loss_f + loss_b
        loss = loss_f
        return loss

def binary_loss_function(x_recon, x, z_mu, z_var, z_0, z_k, log_det_jacobians, z_size, cuda, beta=1, summ = True, log_vamp_zk = None, if_rec = True):
    """
    z_mu: mean of z_0
    z_var: variance of z_0
    z_0: first stochastic latent variable
    z_k: last stochastic latent variable
    log_det_jacobians: log det jacobian
    beta: beta for annealing according to Equation 20
    log_vamp_zk: default None but log_p(zk) if VampPrior used
    the function returns: Free Energy Bound (ELBO), reconstruction loss, kl
    """
    batch_size = x.size(0) 
    
    logvar=torch.zeros(batch_size, z_size)    
    if cuda == True:                      
        logvar=logvar.cuda()
        
    # calculate log_p(zk) under standard Gaussian unless log_p(zk) under VampPrior given
    if log_vamp_zk is None:
        log_p_zk = log_normal_dist(z_k, mean=0, logvar=logvar, dim=1) # ln p(z_k) = N(0,I)
    else:
        log_p_zk = log_vamp_zk
    
                 
    log_q_z0 = log_normal_dist(z_0, mean=z_mu, logvar=z_var.log(), dim=1)

    log_p_zk = log_p_zk + 1e-8
    log_q_z0 = log_q_z0 + 1e-8

    
    if (summ == True):  ## Computes the binary loss function with summing over batch dimension 
        
        #Reconstruction loss: Binary cross entropy
        reconstruction_loss = nn.MSELoss(reduction='sum')

        if if_rec:
            log_p_xz = reconstruction_loss(x_recon, x)  #-log_p(x|z_k)
        else:
            log_p_xz = 0
        log_p_xz = log_p_xz
        kl = torch.sum(log_q_z0 - log_p_zk) - torch.sum(log_det_jacobians) #sum over batches
        #elbo = elbo / batch_size
        log_p_xz = log_p_xz / batch_size
        kl = kl / batch_size

        elbo = 0
        
        return elbo, log_p_xz, kl
    
    else:              ## Computes the binary loss function without summing over batch dimension (used during testing) 
        if len(log_det_jacobians.size()) > 1:
            log_det_jacobians = log_det_jacobians.view(log_det_jacobians.size(0), -1).sum(-1)

        reconstruction_loss = nn.BCELoss(reduction='none')
        log_p_xz = reconstruction_loss(x_recon.view(batch_size, -1), x.view(batch_size, -1))  #-log_p(x|z_k)
        log_p_xz = torch.sum(log_p_xz, dim=1)
        
        #Equation (20)
        elbo = log_q_z0 - log_p_zk - log_det_jacobians + log_p_xz 

        return elbo, log_p_xz, (log_q_z0 - log_p_zk - log_det_jacobians)
