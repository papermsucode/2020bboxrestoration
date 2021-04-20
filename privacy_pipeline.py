import numpy as np
import random
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from archface import LResNet100

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



class Pipeline(nn.Module):
    """
    Classs for reconstructing a face from its embedding. Da 1 iteration returning reconstructed face and a corresponding cosine similarity.
    How it works:
        Do __call__ method until get satisfactory cosine e.x. (pipeline = Pipeline(); cos=0; while cos < 0.98: face, cos = pipeline())
    
    Arguments:
        emb_target: (torch.FloatTensor) face embedding to be attacked
        net: (nn.Module) model for producing a face embedding
        device: (torch.device)
        batch_size: (int) batch size
        dim: (int) dimention of cropped image (images should be square so dim_x=dim_y)
        edge: (int) number of edge pixels (left and right) that shouldn't be used for reconstruction (0 is also OK)
        up: (int) number of top pixels  that shouldn't be used for reconstruction (0 is also OK)
        sym_part: (float 0.0-1.0) symmetric constraint. If equels to 1.0 then the reconstructed face is fully symmetric
        gauss_amplityde: (float) max amplitude of sampled gausses
        color: (bool) cororful reconstruction or not
        multistart: (bool) start algorithm multiple times to choose the best trajectory
    """
    
    def __init__(self, emb_target, net, device, batch_size=80, dim=112, edge=20, up=5, sym_part=1, gauss_amplitude=0.01, color=False, multistart=True):
        super(Pipeline,self).__init__()
        
        self.resnet = net
        self.device = device
        self.resnet.eval().to(self.device)
        self.dim = dim
        self.edge = edge
        self.up = up
        
        self.batch_size = batch_size
        self.sym_part = sym_part
        self.gauss_amplitude = gauss_amplitude
        self.color = color
        
        self.multistart = multistart
        self.best_multi_face = None
        self.best_multi_cos = -1
        self.iters = 0
        self.iters_before_restart = 100
        self.N_restarts = 10
        
        self.emb_target = emb_target.expand(batch_size,-1).to(self.device)
        
        self.im_init = torch.zeros(batch_size,3,dim,dim).to(self.device)
        
        self.face = torch.zeros((batch_size,3,dim,dim,)).to(self.device)
        self.best_face = None
        
        x, y = np.meshgrid(np.linspace(0,dim,dim), np.linspace(0,dim,dim))
        self.x_grid = torch.FloatTensor(x).expand(batch_size,dim,dim).to(self.device)
        self.y_grid = torch.FloatTensor(y).expand(batch_size,dim,dim).to(self.device)
        
        self.best_ind = None
        self.best_value = 0
        self.global_best_value = -1
        self.best_cos = -1
        self.norm = None
        self.objective = 0
        
        
    def gauss_2d(self,dim_x,dim_y,x_coord,y_coord,sigma1,sigma2,A):
        batch_size = self.batch_size
        
        x = self.x_grid.T
        y = self.y_grid.T
        
        sigma1 = sigma1.expand(dim_x,batch_size).to(self.device)
        sigma2 = sigma2.expand(dim_x,batch_size).to(self.device)
        x_coord = x_coord.expand(dim_x,batch_size).to(self.device)
        y_coord = y_coord.expand(dim_x,batch_size).to(self.device)
        
        g_x = torch.exp( - (x[:,0] - x_coord) ** 2 / (2 * sigma1 ** 2)).expand(dim_x,dim_y,batch_size)
        g_y = torch.exp( - (y[0] - y_coord) ** 2 / (2 * sigma2 ** 2)).expand(dim_y,dim_x,batch_size)
        
        g_y = torch.transpose(g_y,0,1)
        
        A = A.expand(dim_x,batch_size)
        A = A.expand(dim_x,batch_size)
        A = A.expand(dim_y,dim_x,batch_size)
        g = A.to(self.device) * g_x * g_y
        g = g.permute(2,0,1)
        
        return g
    
    def gen_gauss(self):
        dim = self.dim
        ampl = self.gauss_amplitude
        batch_size = self.batch_size
        
        w = torch.rand(batch_size,dtype=torch.float) * 150
        w = 0.5 * (10 + w.to(self.device) + nn.ReLU()((300 - 550 * nn.ReLU()(self.best_cos.to(self.device)) ** 0.3))) ** 0.5 
        w1 = w2 = w
        
        A1 = ampl * (torch.rand(batch_size, dtype=torch.float) - 0.5)
        
        x = torch.FloatTensor(batch_size).uniform_(self.edge,dim-self.edge)
        y = torch.FloatTensor(batch_size).uniform_(self.up,self.dim)
        
        gauss1 = self.gauss_2d(dim,dim,x,y,w1,w2,A1)
        
        if self.color:
            i1,i2,i3 = random.choice([(0,0,1),(0,1,0),(1,0,0)])
        else:
            i1,i2,i3 = 1,1,1
            
        gauss1 = gauss1 + self.sym_part * torch.flip(gauss1,[2])
        out = torch.transpose(torch.stack([i1 * gauss1, i2 * gauss1, i3 * gauss1]),1,0)
        return out
    
    def forward(self):
        batch_size = self.batch_size
        iters_before_restart = self.iters_before_restart
        if self.iters > iters_before_restart * self.N_restarts:
            self.im_init = 0.994 * self.im_init
        
        face = self.face + self.im_init
        
        emb = self.resnet(face)
        
        cos_target = torch.cosine_similarity(emb,self.emb_target)
        self.norm = torch.norm(emb[self.best_ind]).item()
        
        objective = cos_target - ((torch.norm(self.emb_target,dim=1) - torch.norm(emb,dim=1)) ** 2) / 400
        
        self.objective = objective
        best_ind = torch.argmax(objective)
        best_value = torch.max(objective)
        best_cos = torch.max(cos_target)

        self.best_ind = best_ind
        self.best_cos = best_cos
        self.best_value = best_value
        
        if best_value > self.global_best_value:
            self.global_best_cos = self.best_value
            self.bad_iters = 0
            self.best_face = self.face[self.best_ind].clone()
            
        gauss = self.gen_gauss()
        
        N_restarts = self.N_restarts
        restarts = [iters_before_restart * (n + 1) for n in range(N_restarts)]
        if self.multistart:
            self.iters += 1
            if self.iters in restarts:
                if self.best_cos > self.best_multi_cos:
                    self.best_multi_cos = self.best_cos
                    self.best_multi_face = self.face[self.best_ind]
                    self.face = torch.zeros((self.batch_size,3,self.dim,self.dim)).to(self.device)
                else:
                    self.face = torch.zeros((self.batch_size,3,self.dim,self.dim)).to(self.device)
                    
            if self.iters == iters_before_restart * N_restarts:
                self.face[:] = self.best_multi_face
                
        self.face = self.face[self.best_ind].clone().expand((batch_size,3,self.dim,self.dim))
        self.face = self.face + gauss
        self.face = torch.clamp(self.face,0,1)
        
        return face[self.best_ind], best_cos.item()
            

        