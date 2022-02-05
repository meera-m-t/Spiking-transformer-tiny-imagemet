import os
import torch
from SpykeTorch import functional as sf
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


use_cuda = True
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,                 
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        k_conv1 = 5
        k_conv2 = 2
        self.conv1 = snn.Convolution(self.in_channels, self.out_channels, k_conv1, 0.8, 0.05)
        self.conv1_t = 10
        self.k1 = 1
        self.r1 = 2

        self.conv2 = snn.Convolution(self.out_channels, self.out_channels, k_conv2, 0.8, 0.05)
        self.conv2_t = 1
        self.k2 = 1
        self.r2 = 1

        self.stdp1 = snn.STDP(self.conv1, (0.04, -0.03))
        self.stdp2 = snn.STDP(self.conv2, (0.04, -0.03))
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0

    
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input):     
        input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:           
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)            
            self.spk_cnt1 += 1
            if self.spk_cnt1 >= 5000:
                self.spk_cnt1 = 0
                ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                ap = torch.min(ap, self.max_ap)
                an = ap * -0.75
                self.stdp1.update_all_learning_rate(ap.item(), an.item())
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
            self.save_data(input, pot, spk, winners)

            if pot.shape[2] == 112:                  
                x = 79
                y = 2
                z = 1
            elif pot.shape[2] == 56:
                x = 37
                y = 2
                z = 1

            elif pot.shape[2] == 28:
                x = 16
                y = 2
                z = 1

            else:
                print(pot.shape[2],"hello........................................")
              
            spk_in = sf.pad(sf.pooling(spk, y ,y, z), (x,x,x,x))
            spk_in = sf.pointwise_inhibition(spk_in)     
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)          
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
            self.save_data(spk_in, pot, spk, winners)
            spk_out = sf.pad(sf.pooling(spk, 2 ,2, 1), (2,2,2,2))
            return spk_out
        else:        
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if pot.shape[2] == 112:                  
                x = 79
                y = 2
                z = 1
            elif pot.shape[2] == 56:
                x = 37
                y = 2
                z = 1
            elif pot.shape[2] == 28:
                x = 16
                y = 2
                z = 1


            else:
                print(pot.shape[2],"hello........................................")             
            spk_in = sf.pad(sf.pooling(spk, y ,y, z), (x,x,x,x))      
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            spk_out = sf.pad(sf.pooling(spk, 2 ,2, 1), (2,2,2,2))      
            return spk_out


    def stdp(self):
        self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm([dim*56, 224])
        self.norm1 = nn.LayerNorm([dim*14, 224])  
        self.norm2 = nn.LayerNorm([784, 224])  
        self.norm3 = nn.LayerNorm([257, 224])     
        self.fn = fn
    def forward(self, x, **kwargs):
        if x.shape[1] == 12544:
            return self.fn(self.norm(x), **kwargs)
        elif x.shape[1] == 3136:
            return self.fn(self.norm1(x), **kwargs)
        elif x.shape[1] == 784:
            return self.fn(self.norm2(x), **kwargs)
        else:
            print( x.shape[1],"........................................................iiiiiiiiiiiiiii")

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads, dim_head ,  dropout = 0.5):

        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.heads = 1
        self.dim_head = dim_head     
        self.inner_dim = self.dim_head *  self.heads
        project_out = not (heads == 1 and dim_head == self.dim)
        self.scale = self.dim_head ** -0.5
     
        #
        self.to_q = SepConv2d(self.dim, self.inner_dim)
        self.to_k = SepConv2d(self.dim, self.inner_dim)
        self.to_v = SepConv2d(self.dim, self.inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads         
 
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size) 

        q = self.to_q(x)    
        self.to_q.stdp()     
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
              
        v = self.to_v(x)
        self.to_v.stdp()        
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        
        k = self.to_k(x)
        self.to_k.stdp()        
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim,dropout):                        
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.dim = dim
        self.img_size = img_size  
        self.dim_head = dim_head          
        self.dropout = dropout       
        self.dim_head = dim_head  
        self.mlp_dim = mlp_dim     
        for _ in range(depth):      
            self.layers.append(nn.ModuleList([
                PreNorm(self.dim, ConvAttention(self.dim, self.img_size, self.heads, self.dim_head,self.dropout)), 
                PreNorm(self.dim, FeedForward(self.dim, self.mlp_dim, self.dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:        
            x = attn(x) + x  	
            x = ff(x) + x   
        return x



class CvT(nn.Module):
    def __init__(self, input_channels, features_per_class, number_of_classes,s2_kernel_size,
               threshold, stdp_lr, anti_stdp_lr,dropout=0.5, image_size=224, dim=224, kernels=[224, 224, 224], strides=[113, 57, 29],
                 heads=[1, 14,6] , depth = [1, 2, 10],pool='cls', emb_dropout=0.5, scale_dim=4):
        super(CvT, self).__init__()
        self.features_per_class = features_per_class
        self.number_of_classes = number_of_classes
        self.number_of_features = features_per_class * number_of_classes
        self.kernel_size = s2_kernel_size
        self.threshold = threshold
        self.stdp_lr = stdp_lr
        self.anti_stdp_lr = anti_stdp_lr
        self.dropout = torch.ones(self.number_of_features) * dropout
        self.to_be_dropped = torch.bernoulli(self.dropout).nonzero()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim
        ##### Stage 1 ################################################################
        self.conv1 = snn.Convolution(input_channels, kernels[0], strides[0], 0.8, 0.05)
        self.stdp1 = snn.STDP(self.conv1, (0.0004, -0.0003))                
        self.conv1_t = 15
        self.k1 = 1
        self.r1 = 3
        self.R1 = Rearrange('b c h w -> b (h w) c', h = image_size//2, w = image_size//2)
        self.norm1 = nn.LayerNorm([self.dim*56,224])        
        self.stage1_transformer = Transformer(dim=self.dim, img_size=image_size//2,depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=self.dim * scale_dim, dropout=dropout)
        self.R11 = Rearrange('b (h w) c -> b c h w', h = image_size//2, w = image_size//2)


        ##### Stage 2 ###################################################################

        scale = heads[1]//heads[0]
        self.dim2 = scale*self.dim
        self.conv2 = snn.Convolution(224, kernels[1], strides[1], 0.8, 0.05)
        self.conv2_t = 10
        self.k2 = 1
        self.r2 = 1      
        self.stdp2 = snn.STDP(self.conv1, (0.04, -0.03))  
        self.R2 = Rearrange('b c h w -> b (h w) c', h = image_size//4, w = image_size//4)
        self.norm2 = nn.LayerNorm([self.dim2,224])
        
        self.stage2_transformer =  Transformer(dim=self.dim ,     img_size=image_size//4, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=self.dim  * scale_dim, dropout=dropout)                              
                                           
        self.R22 = Rearrange('b (h w) c -> b c h w', h = image_size//4, w = image_size//4)
        
        ##### Stage 3 ##################################################################
        input_channels = self.dim2
        scale = heads[2] // heads[1]
        self.dim3 = scale*self.dim2
        self.conv3 = snn.Convolution(224, kernels[2], strides[2], 0.8, 0.05)
        self.stdp3 = snn.STDP(self.conv3, (0.04, -0.03), False, 0.2, 0.8)
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.04, 0.005), False, 0.2, 0.8)
        self.R3 = Rearrange('b c h w -> b (h w) c', h = image_size//8, w = image_size//8)
        self.norm3 = nn.LayerNorm([784,224])        
        self.stage3_transformer = Transformer(dim=self.dim, img_size=image_size//8, depth=depth[2], heads=heads[2], dim_head=self.dim,
                                              mlp_dim=self.dim* scale_dim, dropout=dropout)
        self.R33 = Rearrange('b (h w) c -> b c h w', h = image_size//8, w = image_size//8)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout_large = nn.Dropout(emb_dropout)


        self.norm33 = nn.LayerNorm(self.dim)
        self.mlp_head = nn.Linear(self.dim, self.number_of_classes)
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.decision_map = []
        for i in range(200):
            self.decision_map.extend([i]*20)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def forward(self, input, max_layer):
        input = input.float()    
        if self.training:
            pot = self.conv1(input)
            pot = self.R1(pot)        
            pot = self.norm1(pot)                      
            pot = self.stage1_transformer(pot)              
            pot = self.R11(pot)                       
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 5000:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.save_data(input, pot, spk, winners)
                return spk, pot
 
            spk_in =  spk
            pot = self.conv2(spk_in)
            pot = self.R2(pot)
            pot = self.norm2(pot) 
            pot = self.stage2_transformer(pot)   
            pot = self.R22(pot) 
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 5000:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)           
                self.save_data(input, pot, spk, winners)
                return spk, pot

            pot = self.conv3(spk)
            pot = self.R3(pot)
            pot = self.norm3(pot)
            pot = self.stage3_transformer(pot)
            pot = self.R33(pot)
            spk = sf.fire(pot)      
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        else:
            pot = self.conv1(input)
            pot = self.R1(pot)
            pot = self.norm1(pot)
            pot= self.stage1_transformer(pot)            
            pot = self.R11(pot)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                return spk, pot
            pot = self.conv2(spk)
            pot = self.R2(pot)
            pot = self.norm2(pot)
            pot = self.stage2_transformer(pot) 
            pot = self.R22(pot)                           
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                return spk, pot
            pot = self.conv3(spk)             
            pot = self.R3(pot)
            pot = self.norm3(pot)
            pot = self.stage3_transformer(pot)
            print(pot.shape)
            pot = self.R33(pot)                     
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output


    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

