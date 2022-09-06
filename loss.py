from __future__ import print_function
import torch
import torch.nn.functional as F




def h2_pq(mu_1, sigma_1, mu_2, sigma_2):
    return 1 - torch.sqrt( torch.div( 2 * sigma_1 * sigma_2, (sigma_1 **2 + sigma_2 ** 2) ) ) * torch.exp( - 0.25 * torch.div( (mu_1 - mu_2) ** 2, sigma_1 ** 2 + simga_2 ** 2  ))  # check this equation  with zzy 

def cosine_pq(mu_1, sigma_1, mu_2, sigma_2):
    return torch.exp( -(mu_1 - mu_2) ** 2 / ( 2 * (sigma_1 ** 2 + sigma_2 ** 2 ))) / torch.sqrt( ( (sigma_1 ** 2 + sigma_2 ** 2 ) / ( 2 * sigma_1 * sigma_2) ) + 1e-9) # (bsz, bsz)

def akd_ma_loss(teacher_ma, student_ma):
    # relation kd angle 
    t_mu, t_sigma = teacher_ma  # (bsz, ), (bsz, )
    s_mu, s_sigma = student_ma  # (bsz, ), (bsz, )
    with torch.no_grad():
        t_cosine = cosine_pq(t_mu.unsqueeze(1), t_sigma.unsqueeze(1), 
        t_mu.unsqueeze(0), t_sigma.unsqueeze(0))
    s_cosine = cosine_pq(s_mu.unsqueeze(1), s_sigma.unsqueeze(1), 
        s_mu.unsqueeze(0), s_sigma.unsqueeze(0))
    loss = F.smooth_l1_loss(s_cosine, t_cosine)
    return loss 



def jf_pq(mu_1, sigma_1, mu_2, sigma_2): 
    return 0.5 * ( torch.div(1, sigma_1 ** 2) + torch.div(1, sigma_2 ** 2)) * ((sigma_1 - sigma_2) ** 2 + (mu_1 - mu_2) ** 2 )
    

def kl_pq(mu_1, sigma_1, mu_2, sigma_2): #
    return torch.log(torch.div(sigma_2, sigma_1)) +  torch.div( sigma_1 ** 2 + (mu_1 - mu_2) ** 2, 
                                                                    2 * sigma_2 ** 2) - 0.5 


def kd_ma_loss(teacher_ma, student_ma, dkd=False):
    t_mu, t_sigma = teacher_ma  # (bsz, ), (bsz, )
    s_mu, s_sigma = student_ma  # (bsz, ), (bsz, )
    
    # q from N( mu_1, sigma_1 )   p from N( mu_2, sigma_2)
    # KL( p || q ) =  log ( sigma_2 / sigma_1) + \frac{ sigma_1 ** 2  + (mu_1 - mu_2) ** 2 }{2 * sigma_2 ** 2} - 0.5 
    # 
    # in KD:   p is teacher distribution , q is the student distribution
    # 
    kl_loss = kl_pq(t_mu, t_sigma, s_mu, s_sigma) 
                #torch.log(torch.div(s_sigma , t_sigma )) +  torch.div( t_sigma ** 2 + (t_mu - s_mu) ** 2, 
            #                                                      2 * s_sigma ** 2) - 0.5 
    # print(kl_loss)
    if dkd:
        precision = 1.0 / s_sigma ** 2  # (bsz, )
        # print(precision)
        pmax, pmin = precision.max(), precision.min()
        weight = 1 - ((precision - pmin )/ (pmax - pmin)  + 1e-3)
        # print(weight)
        kl_loss *= weight   # element-wise 

    loss = kl_loss.mean()
    return loss 

def rkd_ma_loss_jf(teacher_ma, student_ma): 
     # relation kd 
    t_mu, t_sigma = teacher_ma  # (bsz, ), (bsz, )
    s_mu, s_sigma = student_ma  # (bsz, ), (bsz, )    
    with torch.no_grad():
        t_d = jf_pq(t_mu.unsqueeze(1), t_sigma.unsqueeze(1),
                              t_mu.unsqueeze(0), t_sigma.unsqueeze(0))
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td 
    s_d = jf_pq(s_mu.unsqueeze(1), s_sigma.unsqueeze(1),
                s_mu.unsqueeze(0), s_sigma.unsqueeze(0))
    mean_sd = s_d[s_d > 0].mean()
    s_d = s_d / mean_sd 
    loss = F.smooth_l1_loss(s_d, t_d)
    return loss 
    


def rkd_ma_loss_js(teacher_ma, student_ma): 
    # relation kd 
    t_mu, t_sigma = teacher_ma  # (bsz, ), (bsz, )
    s_mu, s_sigma = student_ma  # (bsz, ), (bsz, )
    
    with torch.no_grad(): # compute teacher distance matrix 
        t_pq_distance = kl_pq(t_mu.unsqueeze(1), t_sigma.unsqueeze(1),
                            t_mu.unsqueeze(0), t_sigma.unsqueeze(0))

        t_qp_distance = kl_pq(t_mu.unsqueeze(0), t_sigma.unsqueeze(0),
                            t_mu.unsqueeze(1), t_sigma.unsqueeze(1))  
        t_d= ( t_pq_distance + t_qp_distance) /2 
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td 
        # print(t_d)  
   # print(distance_matrix)
    s_pq_distance = kl_pq(s_mu.unsqueeze(1), s_sigma.unsqueeze(1),
                         s_mu.unsqueeze(0), s_sigma.unsqueeze(0))

    s_qp_distance = kl_pq(s_mu.unsqueeze(0), s_sigma.unsqueeze(0),
                         s_mu.unsqueeze(1), s_sigma.unsqueeze(1))  
    s_d= ( s_pq_distance + s_qp_distance) /2 
    mean_sd = s_d[s_d > 0].mean()
    s_d = s_d / mean_sd 
    # huber loss here 
    loss = F.smooth_l1_loss(s_d, t_d) 
    return loss 

def Linf_step(x, g, eps, lr):
    dx = lr * (g).sign()
    x = (x+dx).clamp(-eps, eps)
    return x
    
def L2_step(x, g, eps, lr):
    dx = L2_norm(g, lr)
    x = L2_clip(x+dx, eps)
    return x
    
def L2_clip(x, eps):
    norm = ((x**2).sum(-1).sum(-1)+1e-10).sqrt()
    norm = norm.unsqueeze(-1).unsqueeze(-1)
    return torch.clamp(norm, 0, eps) * x / norm
        
def L2_norm(x, eps):
    norm = ((x**2).sum(-1).sum(-1)+1e-10).sqrt()
    norm = norm.unsqueeze(-1).unsqueeze(-1)
    return eps * x / norm
