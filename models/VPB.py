import torch 
from torch import nn
import numpy as np
from torch.nn import functional as F 
import models.PFL as PFL
from loss import binary_loss_function

class_mapping = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut"
}
class InferenceBlock(nn.Module):
    def __init__(self, input_units, d_theta, output_units):
        """
        :param d_theta: dimensionality of the intermediate hidden layers.
        :param output_units: dimensionality of the output.
        :return: batch of outputs.
        """
        super(InferenceBlock, self).__init__()
        self.module = nn.Sequential(
            #nn.Linear(input_units, output_units, bias=True),
            nn.Linear(input_units, d_theta, bias=True),
            nn.Softplus(),
            nn.Linear(d_theta, d_theta, bias=True),
            nn.Softplus(),
            nn.Linear(d_theta, output_units, bias=True),
        )

    def forward(self, inps):
        out = self.module(inps)
        return out




class Encoder(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super(Encoder, self).__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        weight_mean = self.weight_mean(inps)
        weight_log_variance = self.weight_log_variance(inps)
        return weight_mean, torch.exp(0.5 * weight_log_variance)
    
class Decoder(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super(Decoder, self).__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        weight_mean = self.weight_mean(inps)
        return weight_mean



class Fuse_Block(nn.Module):
    def __init__(self, dim_i,  dim_hid, dim_out):
        super(Fuse_Block, self).__init__()

        self.pre_process = nn.Sequential(nn.Linear(dim_i, dim_hid), nn.ReLU(), nn.Linear(dim_hid, dim_hid))
        self.post_process = nn.Linear(dim_hid, dim_out)
    
    def forward(self, x):
        x = self.pre_process(x)
        x = torch.mean(x, dim = 1)
        x = self.post_process(x)
        return x
    


class Zero_Parameter(nn.Module):

    def __init__(self, dim_v,dim_t, dim_out, num_heads = 4, k = 4):
        super().__init__()
        self.num_heads = num_heads 
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out
        self.scale = dim_out ** -0.5        
        self.q_proj_pre = nn.Conv1d(dim_t, dim_out, kernel_size=1)
        self.linear_proj = nn.ModuleList([nn.Linear(dim_v, dim_t, bias = False) for i in range(k)])

    def forward(self, F_t, F_s, layer):
        B1, N1, C1 = F_t.shape
        B2, N2, C2 = F_s.shape
        assert B1 == B2

        F_s = self.linear_proj[layer](F_s)
        F_s = F_s / F_s.norm(dim=-1, keepdim = True)

        q_t = self.q_proj_pre(F_t.permute(0, 2, 1)).permute(0, 2, 1).reshape(B1, N1, self.num_heads, self.head_dim)
        k_s = F_s.reshape(B2, N2, self.num_heads, self.head_dim)
        v_s = F_s.reshape(B2, N2, self.num_heads, self.head_dim)
        attn_t = torch.einsum('bnkc,bmkc->bknm', q_t, k_s) * self.scale
        attn_t = attn_t.softmax(dim = -1)
        F_t_a = torch.einsum('bknm,bmkc->bnkc', attn_t, v_s).reshape(B1, N1, self.dim_out)
        F_t_a = F_t_a + F_t
        F_t_a = F_t_a / F_t_a.norm(dim=-1, keepdim = True)
        return F_t_a, F_s



class Global_Feature(nn.Module):
    def __init__(self, dim_i, dim_hid, dim_out, k):

        super(Global_Feature, self).__init__()
        
        self.fuse_modules = nn.Linear(dim_i * k, dim_hid)
        self.compress =  nn.Linear(dim_hid, 1)
        self.post_process = nn.Linear(dim_hid, dim_out)


    def forward(self, inps):
        x = torch.cat(inps, dim = 2)
        x = self.fuse_modules(x)
        x_temp = self.compress(x)
        attention_weights = nn.Softmax(dim=1)(x_temp) 
        x = torch.sum(attention_weights * x, dim=1)
        x = self.post_process(x)
        return x




class TextEncoder(nn.Module):

    def __init__(self, clip_model, args):
        super().__init__()
        self.clip_model = clip_model
        self.num_tokens = args.prompt_context_len
        self.context_length = clip_model.context_length # 77

    @property
    def dtype(self):
        return self.clip_model.visual.conv1.weight.dtype
    
    def forward(self, text, visual_feature, prompt_bias_list, mode):
        prompt_bias = prompt_bias_list[0]   
        prompt_bias_state = prompt_bias_list[1]
        pos_y = [1] * text.shape[0] # "X "+ name_new.lower(),  the position of [SOS] token is 0 and "X" is 1
        x = self.clip_model.token_embedding(text).type(self.dtype)
        if mode == "train":
            x_new_array = torch.zeros((x.shape[0],visual_feature.shape[0], x.shape[2]), dtype = x.dtype).to(x.device)
            for i in range(text.shape[0]):
                visual_feature_new = visual_feature.clone()
                visual_feature_new[:,:self.num_tokens,:] = visual_feature_new[:,:self.num_tokens,:] + prompt_bias[i].reshape(1,1,-1)
                visual_feature_new[:,self.num_tokens:,:] = visual_feature_new[:,self.num_tokens:,:] + prompt_bias_state.reshape(1,1,-1)
                x_temp = x[i,:,:] + torch.zeros((visual_feature.shape[0], x.shape[1], x.shape[2]), dtype= x.dtype, device=x.device)
                x_new = torch.cat([x_temp[:, 0:pos_y[i], :], visual_feature_new, x_temp[:, (pos_y[i]+1):(self.context_length - visual_feature.shape[1] + 1),:]], dim = 1)
                x_new = x_new + self.clip_model.positional_embedding.type(self.dtype)
                x_new = x_new.permute(1, 0, 2)  # NLD -> LND
                x_new, attn, tokens = self.clip_model.transformer(x_new)
                x_new = x_new.permute(1, 0, 2)  # LND -> NLD
                x_new = self.clip_model.ln_final(x_new).type(self.dtype) 
                x_new = x_new[:, torch.where(text[i] == 49407)[0] + visual_feature.shape[1] - 1,:] @ self.clip_model.text_projection  # 49407 is the position of [cls] token in the CLIP vocabulary
                x_new = x_new.reshape(-1,x_new.shape[-1])
                x_new_array[i, :, :] = x_new
        else:
            x_new_array = torch.zeros((x.shape[0],visual_feature.shape[0] * prompt_bias.shape[0], x.shape[2]), dtype = x.dtype).to(x.device)
            for i in range(text.shape[0]):
                text_feature_list = []
                for j in range(prompt_bias.shape[0]):
                    visual_feature_new = visual_feature.clone()
                    visual_feature_new[:,:self.num_tokens,:] = visual_feature_new[:,:self.num_tokens,:] + prompt_bias[j].reshape(1,1,-1)
                    visual_feature_new[:,self.num_tokens:,:] = visual_feature_new[:,self.num_tokens:,:] + prompt_bias_state[j].reshape(1,1,-1)
                    x_temp = x[i,:,:] + torch.zeros((visual_feature.shape[0], x.shape[1], x.shape[2]), dtype= x.dtype, device=x.device)
                    x_new = torch.cat([x_temp[:, 0:pos_y[i], :], visual_feature_new, x_temp[:, (pos_y[i]+1):(self.context_length - visual_feature.shape[1] + 1),:]], dim = 1)
                    x_new = x_new + self.clip_model.positional_embedding.type(self.dtype)
                    x_new = x_new.permute(1, 0, 2)  # NLD -> LND
                    x_new, attn, tokens = self.clip_model.transformer(x_new)
                    x_new = x_new.permute(1, 0, 2)  # LND -> NLD
                    x_new = self.clip_model.ln_final(x_new).type(self.dtype) 
                    x_new = x_new[:, torch.where(text[i] == 49407)[0] + visual_feature.shape[1] - 1,:] @ self.clip_model.text_projection  # 49407 is the position of [cls] token in the CLIP vocabulary
                    x_new = x_new.reshape(-1,x_new.shape[-1])
                    text_feature_list.append(x_new)
                x_new = torch.stack(text_feature_list, dim = 0).reshape(-1, x_new.shape[-1])
                x_new_array[i, :, :] = x_new
        return x_new_array
    
class Context_Prompting(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vision_width = args.vision_width
        self.text_width = args.text_width
        self.embed_dim = args.embed_dim

        self.prompt_context = nn.Parameter(torch.randn(args.prompt_num, args.prompt_context_len, self.text_width))  # Constructing the learnable context vectors in the original prompt bank.
        self.prompt_state_normal = nn.Parameter(torch.randn(args.prompt_num, args.prompt_state_len, self.text_width)) # Constructing the learnable noraml state vectors in the original prompt bank.
        self.prompt_state_abnormal = nn.Parameter(torch.randn(args.prompt_num, args.prompt_state_len, self.text_width)) # Constructing the learnable abnoraml state vectors in the original prompt bank.
        
        self.temperature_pixel = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temperature_image = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.context_encoder = Encoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )  

        self.context_decoder = Decoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )  # An initial attempt was made to investigate whether adding a reconstruction loss could improve performance; however, this component has since been discarded.

        self.state_encoder = Encoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )

        self.state_decoder = Decoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )  #  # An initial attempt was made to investigate whether adding a reconstruction loss could improve performance; however, this component has since been discarded.


        self.PFL_context = PFL.PlanarPFL(self.context_encoder, self.context_decoder, args) # For image-specific distribution
        # self.PFL = PFL.PlanarPFL_state(self.PFL_encoder, self.PFL_decoder, args) 
        self.PFL_normal = PFL.PlanarPFL_state(self.state_encoder, self.state_decoder, args) # For image-agnostic distribution
        self.PFL_abnormal = PFL.PlanarPFL_state(self.state_encoder, self.state_decoder, args)  # For image-agnostic distribution


        self.fuse = Global_Feature(self.vision_width, self.vision_width // 2, self.text_width, k = len(self.args.features_list))
        self.RCA = Zero_Parameter(dim_v = self.vision_width, dim_t = self.text_width, dim_out= self.text_width, k = len(args.features_list))
        self.class_mapping = nn.Linear(self.text_width, self.text_width)
        self.image_mapping = nn.Linear(self.text_width, self.text_width)
        self._initialize_weights()

        nn.init.trunc_normal_(self.prompt_context, mean=0, std=0.02)
        nn.init.trunc_normal_(self.prompt_state_normal, mean=0.5, std=0.02)
        nn.init.trunc_normal_(self.prompt_state_abnormal, mean=-0.5, std=0.02)

        self.zk_n = None
        self.zk_a = None

    def _initialize_weights(self):
        """Initialize the weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std= 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def prompt_names(self, names):
        sentence = []
        for name in names:
            if name in class_mapping.keys():
                name_new = class_mapping[name]
            else:
                name_new = name
            #sentence.append("a photo of a <|class|>" + name)
            #sentence.append("<|class|> damaged " + name)
            sentence.append("X "+ name_new.lower())
        return sentence
    



    def forward_ensemble(self, model_text_encoder, global_img_feature, patch_img_feature ,names, device, tokenizer, mode = "train"):
        beta = 0
        # global_img = global_img_feature + self.fuse(patch_img_feature)
        global_img = global_img_feature   # Directly utilizing the original global image features from CLIP

        x_mean, z_mu, z_var, log_det_jacobians, z0, zk = self.PFL_context(global_img.clone(), mode = mode)
        x_mean_n, z_mu_n, z_var_n, log_det_jacobians_n, z0_n, zk_n = self.PFL_normal(global_img.clone(), mode = mode)
        x_mean_a, z_mu_a, z_var_a, log_det_jacobians_a, z0_a, zk_a = self.PFL_abnormal(global_img.clone(), mode = mode)


        if self.zk_n is None and self.zk_a is None and mode != "train":  # 
            self.zk_n = zk_n
            self.zk_a = zk_a
        
        if mode != "train":
            zk_n = self.zk_n
            zk_a = self.zk_a


        #Compute the loss 
        if mode == "train":
            loss, rec, kl = binary_loss_function(x_mean, global_img, z_mu, z_var, z0, zk, log_det_jacobians,
                                                self.args.embed_dim, True, beta = beta, log_vamp_zk = None, if_rec = True)

            loss, rec_n, kl_n = binary_loss_function(x_mean_n, global_img, z_mu_n, z_var_n, z0_n, zk_n, log_det_jacobians_n,
                                                self.args.embed_dim, True, beta = beta, log_vamp_zk = None, if_rec = False)

            loss, rec_a, kl_a = binary_loss_function(x_mean_a, global_img, z_mu_a, z_var_a, z0_a, zk_a, log_det_jacobians_a,
                                                self.args.embed_dim, True, beta = beta, log_vamp_zk = None, if_rec = False)
            
        else:
            loss = 0
            rec = 0
            kl = 0
            kl_n  = 0
            kl_a = 0
        
        visual_normal = torch.cat([self.prompt_context , self.prompt_state_normal], dim = 1)
        visual_abnormal = torch.cat([self.prompt_context , self.prompt_state_abnormal], dim = 1)
        prompt_names = self.prompt_names(names)
        prompt_names = tokenizer(prompt_names).to(device)
        normal_embeddings = model_text_encoder(prompt_names, visual_normal, [zk, zk_n], mode = mode)
        abnormal_embeddings = model_text_encoder(prompt_names, visual_abnormal, [zk, zk_a], mode = mode) 
        text_embeddings = torch.cat([normal_embeddings, abnormal_embeddings], dim =1)
        text_embeddings = text_embeddings / text_embeddings.norm(dim = -1,keepdim = True)
        return text_embeddings, [rec, 0.01 * kl ,0.01 * kl_n ,0.01 * kl_a]
    
    def forward(self, text_features, image_features, patch_tokens, stage, mode):
        if stage == 1:
            text_embeddings_mapping = self.class_mapping(text_features)
            text_embeddings_mapping = text_embeddings_mapping / text_embeddings_mapping.norm(dim = -1, keepdim = True)
            image_embeddings_mapping = image_features
            image_embeddings_mapping = image_embeddings_mapping / image_embeddings_mapping.norm(dim=-1, keepdim = True)
            pro_img = self.temperature_image.exp() * text_embeddings_mapping @ image_embeddings_mapping.unsqueeze(2)
            anomaly_maps_list = []
            for layer in range(len(patch_tokens)):
                text_embeddings_update, dense_feature = self.RCA(text_features, patch_tokens[layer].clone(), layer)
                anomaly_map = (self.temperature_pixel.exp() * dense_feature @ text_embeddings_update.permute(0,2,1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                if mode == "train":
                    # NOTE: this bilinear interpolation is unreproducible and may occasionally lead to unstable ZSAD performance.
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B,self.args.prompt_num * 2, H, H),
                                                    size = self.args.image_size, mode = 'bilinear', align_corners= True)
                else:
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B,self.args.prompt_num * 2 * self.args.sample_num, H, H),
                                                    size = self.args.image_size, mode = 'bilinear', align_corners= True)
                anomaly_maps_list.append(anomaly_map)
            return pro_img, anomaly_maps_list
        else:
            text_embeddings_mapping = self.class_mapping(text_features)
            text_embeddings_mapping = text_embeddings_mapping / text_embeddings_mapping.norm(dim = -1, keepdim = True)
            image_embeddings_mapping = self.image_mapping(image_features + self.fuse(patch_tokens))
            image_embeddings_mapping = image_embeddings_mapping / image_embeddings_mapping.norm(dim=-1, keepdim = True)
            pro_img = self.temperature_image.exp() * text_embeddings_mapping @ image_embeddings_mapping.unsqueeze(2)

            anomaly_maps_list = []
            for layer in range(len(patch_tokens)):
                text_embeddings_update, dense_feature = self.RCA(text_features, patch_tokens[layer].clone(), layer)
                anomaly_map = (self.temperature_pixel.exp() * dense_feature @ text_embeddings_update.permute(0,2,1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                if mode == "train":
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B,self.args.prompt_num * 2, H, H),
                                                    size = self.args.image_size, mode = 'bilinear', align_corners= True)
                else:
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B,self.args.prompt_num * 2 * self.args.sample_num, H, H),
                                                    size = self.args.image_size, mode = 'bilinear', align_corners= True)
                anomaly_maps_list.append(anomaly_map)
            return pro_img, anomaly_maps_list

    



