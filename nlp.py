# in the name of God the most compasisonate the most merciful 
# good video's lectures about nlp and llms :
# 
# see huggingface_tutorials.py section as well
# Aligning LLMs with Direct Preference Optimization (great presentation watch it)
# sidenote: sft = supervised finetuning, dpo=direct preference optimization (for alignment)
# the notebooks and slides are downloaded here as well check them out
# https://www.youtube.com/watch?v=QXVCqtAZAn4
# 
# Stanford CS25: V3 I Retrieval Augmented Language Models 
# https://www.youtube.com/watch?v=mE7IDf2SmJg
# 
#  3 Vector-based Methods for Similarity Search (TF-IDF, BM25, SBERT) 
# https://www.youtube.com/watch?v=ziiF1eFM3_4

# **SPLADE** is a class of neural retrieval models that learn query/document expansion via the BERT MLM head and sparse regularization³. It merges sparse and dense vector embeddings to improve the performance of vector search². SPLADE uses pretrained language models to learn term expansions and enhance the relevance of sparse vectors². It's used in applications like information retrieval².
# **Query Expansion (QE)**, on the other hand, is a process in Information Retrieval which consists of selecting and adding terms to the user’s query with the goal of minimizing query-document mismatch and thereby improving retrieval performance⁵. It involves techniques such as⁴:
# - Finding synonyms of words, and searching for the synonyms as well.
# - Finding semantically related words (e.g. antonyms, meronyms, hyponyms, hypernyms).
# - Finding all the various morphological forms of words by stemming each word in the search query.
# - Fixing spelling errors and automatically searching for the corrected form or suggesting it in the results.
# - Re-weighting the terms in the original query.
# The goal of query expansion is to enrich the user’s query by finding additional search terms, either automatically, or semi-automatically that represent the user’s information need more accurately and completely⁵. This helps to avoid problems like term mismatch and increases the chances of matching the user’s query to the representations of relevant ideas in documents⁵.
# Source: Conversation with Bing, 1/30/2024
# (1) naver/splade: SPLADE: sparse neural search (SIGIR21, SIGIR22) - GitHub. https://github.com/naver/splade.
# (2) SPLADE for Sparse Vector Search Explained | Pinecone. https://www.pinecone.io/learn/splade/.
# (3) Query Expansion for Information Retrieval | SpringerLink. https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_947.
# (4) Query expansion - Wikipedia. https://en.wikipedia.org/wiki/Query_expansion.
# (5) Laravel Splade - Single Page Applications with Laravel Blade | Laravel .... https://splade.dev/.
# (6) Query expansion - Stanford University. https://nlp.stanford.edu/IR-book/html/htmledition/query-expansion-1.html.
# (7) What is WITH QUERY EXPANSION MODE in MySQL Fulltext Search. https://dba.stackexchange.com/questions/15659/what-is-with-query-expansion-mode-in-mysql-fulltext-search.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributed import get_rank

class GaussianDiffusion(nn.Module):
    def __init__(self, dtype:torch.dtype, model, betas:np.ndarray, w:float, v:float, device:torch.device):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device)
        self.model.dtype = self.dtype
        self.betas = torch.tensor(betas,dtype=self.dtype)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.device = device
        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas)
        
        self.log_alphas_cum = torch.cumsum(self.log_alphas, dim = 0)
        self.alphas_cum = torch.exp(self.log_alphas_cum)
        # self.alphas_bar = torch.cumprod(self.alphas, dim = 0)
        
        self.log_alphas_cum_prev = F.pad(self.log_alphas_cum[:-1],[1,0],'constant', 0)
        self.alphas_cum_prev = torch.exp(self.log_alphas_cum_prev)
        self.log_one_minus_alphas_cum_prev = torch.log(1.0 - self.alphas_cum_prev)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas)
        # self.sqrt_alphas = torch.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_cum = 0.5 * self.log_alphas_cum
        self.sqrt_alphas_cum = torch.exp(self.log_sqrt_alphas_cum)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_cum = torch.log(1.0 - self.alphas_cum)
        self.sqrt_one_minus_alphas_cum = torch.exp(0.5 * self.log_one_minus_alphas_cum)
        
        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(self.log_one_minus_alphas_cum_prev - self.log_one_minus_alphas_cum)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0))
        self.mu_coef_x0 = self.betas * torch.exp(0.5 * self.log_alphas_cum_prev - self.log_one_minus_alphas_cum)
        self.mu_coef_xt = torch.exp(0.5 * self.log_alphas + self.log_one_minus_alphas_cum_prev - self.log_one_minus_alphas_cum)
        self.vars = torch.cat((self.tilde_betas[1:2],self.betas[1:]), 0)
        
        self.recip_sqrt_alphas = torch.exp(-self.log_sqrt_alphas)
        self.coef2 = self.recip_sqrt_alphas * self.betas / self.sqrt_one_minus_alphas_cum
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_cum = torch.exp(-self.log_sqrt_alphas_cum)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_cum = torch.exp(self.log_one_minus_alphas_cum - self.log_sqrt_alphas_cum)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)
    @staticmethod
    def _extract(coef:torch.Tensor, t:torch.Tensor, x_shape:tuple) -> torch.Tensor:
        assert t.shape[0] == x_shape[0]
        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        chosen = coef[t]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_mean_variance(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of q(x_t|x_0)
        """
        # I guess this is akin to our 
        # sqrt_one_minus_alphas_cumprod_t = self._get_value_for_timestep_t(self.sqrt_one_minus_alphas_cumprod, timesteps, is_batch)
        # sqrt_recip_alphas_t = self._get_value_for_timestep_t(self.sqrt_recip_alphas, timesteps, is_batch)
        # model_mean =  sqrt_recip_alphas_t * (input_images - betas_t*predicted_noise/sqrt_one_minus_alphas_cumprod_t)
        # 
        sqrt_alphas_cum_t = self._extract(self.sqrt_alphas_cum, t, x_0.shape)
        var = self._extract(1.0 - self.sqrt_alphas_cum, t, x_0.shape)
        mean = sqrt_alphas_cum_t * x_0
        return mean, var
    
    def q_sample(self, x_0:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sample from q(x_t|x_0)
        """
        noise_eps = torch.randn_like(x_0, requires_grad=False)
        sqrt_alphas_cum_t = self._extract(self.sqrt_alphas_cum, t, x_0.shape)
        sqrt_one_minus_alphas_cum_t = self._extract(self.sqrt_one_minus_alphas_cum, t, x_0.shape)
        return  sqrt_alphas_cum_t * x_0 +  sqrt_one_minus_alphas_cum_t * noise_eps, noise_eps
    
    def q_posterior_mean_variance(self, x_0:torch.Tensor, x_t:torch.Tensor, t:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of q(x_{t-1}|x_t,x_0)
        """
        mu_coef_x0_t = self._extract(self.mu_coef_x0, t, x_0.shape)
        mu_coef_xt_t = self._extract(self.mu_coef_xt, t, x_t.shape)
         
        posterior_mean = mu_coef_x0_t * x_0  + mu_coef_xt_t * x_t
        
        posterior_var_max_t = self._extract(self.tilde_betas, t, x_t.shape)
        log_posterior_var_min_t = self._extract(self.log_tilde_betas_clipped, t, x_t.shape)
        # betas_t
        log_posterior_var_max_t = self._extract(torch.log(self.betas), t, x_t.shape)
        log_posterior_var = self.v * log_posterior_var_max_t + (1 - self.v) * log_posterior_var_min_t
        neo_posterior_var = torch.exp(log_posterior_var)
        
        return posterior_mean, posterior_var_max_t, neo_posterior_var
    
    def p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        # if model_kwargs == None:
        #     model_kwargs = {}
        # B, C = x_t.shape[:2]
        # assert t.shape == (B,)
        # cemb_shape = model_kwargs['cemb'].shape
        #pred_eps_cond
        pred_noise_cond = self.model(x_t, t, **model_kwargs)
        # model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        # pred_eps_uncond
        pred_noise_uncond = self.model(x_t, t, **model_kwargs)
        # pred_eps - weighting the conditonal and undconditional output to get the final output
        pred_noise = (1 + self.w) * pred_noise_cond - self.w * pred_noise_uncond
        # assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        # assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        # assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"
        # p_mean = self._predict_xt_prev_mean_from_eps(x_t, t.type(dtype=torch.long), pred_noise)
        recip_sqrt_alphas_t = self._extract(coef = self.recip_sqrt_alphas, t = t, x_shape = x_t.shape)
        coef2_t = self._extract(coef = self.coef2, t = t, x_shape = x_t.shape)
        # looks like our model_mean =  sqrt_recip_alphas_t * (input_images - betas_t*predicted_noise/sqrt_one_minus_alphas_cumprod_t)
        p_mean =  recip_sqrt_alphas_t * x_t -  coef2_t * pred_noise
        p_var_t = self._extract(self.vars, t.type(dtype=torch.long), x_t.shape)
        return p_mean, p_var_t

    def _predict_x0_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        sqrt_recip_alphas_cum_t = self._extract(coef = self.sqrt_recip_alphas_cum, t = t, x_shape = x_t.shape)
        sqrt_one_minus_alphas_cum_t = self._extract(coef = self.sqrt_one_minus_alphas_cum, t = t, x_shape = x_t.shape)
        return  sqrt_recip_alphas_cum_t * x_t -  sqrt_one_minus_alphas_cum_t * noise

    # def _predict_xt_prev_mean_from_eps(self, x_t:torch.Tensor, t:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
    #     return self._extract(coef = self.recip_sqrt_alphas, t = t, x_shape = x_t.shape) * x_t - \
    #         self._extract(coef = self.coef2, t = t, x_shape = x_t.shape) * eps

    def p_sample(self, x_t:torch.Tensor, t:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        sample x_{t-1} from p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        # assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.p_mean_variance(x_t , t, **model_kwargs)
        # assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        # assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    
    def sample(self, shape:tuple, **model_kwargs) -> torch.Tensor:
        """
        sample images from p_{theta}
        """
        local_rank = get_rank()
        if local_rank == 0:
            print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.ones([x_t.shape[0]], device = self.device) * self.T
        for _ in tqdm(range(self.T),dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            tlist -= 1
            with torch.no_grad():
                x_t = self.p_sample(x_t, tlist, **model_kwargs)
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process...')
        return x_t
    
    def ddim_p_mean_variance(self, x_t:torch.Tensor, t:torch.Tensor, prevt:torch.Tensor, eta:float, **model_kwargs) -> torch.Tensor:
        """
        calculate the parameters of p_{theta}(x_{t-1}|x_t)
        """
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        cemb_shape = model_kwargs['cemb'].shape
        pred_eps_cond = self.model(x_t, t, **model_kwargs)
        model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        
        assert torch.isnan(x_t).int().sum() == 0, f"nan in tensor x_t when t = {t[0]}"
        assert torch.isnan(t).int().sum() == 0, f"nan in tensor t when t = {t[0]}"
        assert torch.isnan(pred_eps).int().sum() == 0, f"nan in tensor pred_eps when t = {t[0]}"

        alphas_bar_t = self._extract(coef = self.alphas_cum, t = t, x_shape = x_t.shape)
        alphas_bar_prev = self._extract(coef = self.alphas_cum_prev, t = prevt + 1, x_shape = x_t.shape)
        sigma = eta * torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_prev))
        p_var = sigma ** 2
        coef_eps = 1 - alphas_bar_prev - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = torch.sqrt(coef_eps)
        p_mean = torch.sqrt(alphas_bar_prev) * (x_t - torch.sqrt(1 - alphas_bar_t) * pred_eps) / torch.sqrt(alphas_bar_t) + \
            coef_eps * pred_eps
        return p_mean, p_var
    
    def ddim_p_sample(self, x_t:torch.Tensor, t:torch.Tensor, prevt:torch.Tensor, eta:float, **model_kwargs) -> torch.Tensor: 
        if model_kwargs == None:
            model_kwargs = {}
        B, C = x_t.shape[:2]
        assert t.shape == (B,), f"size of t is not batch size {B}"
        mean, var = self.ddim_p_mean_variance(x_t , t.type(dtype=torch.long), prevt.type(dtype=torch.long), eta, **model_kwargs)
        assert torch.isnan(mean).int().sum() == 0, f"nan in tensor mean when t = {t[0]}"
        assert torch.isnan(var).int().sum() == 0, f"nan in tensor var when t = {t[0]}"
        noise = torch.randn_like(x_t)
        noise[t <= 0] = 0 
        return mean + torch.sqrt(var) * noise
    
    def ddim_sample(self, shape:tuple, num_steps:int, eta:float, select:str, **model_kwargs) -> torch.Tensor:
        local_rank = get_rank()
        if local_rank == 0:
            print('Start generating(ddim)...')
        if model_kwargs == None:
            model_kwargs = {}
        # a subsequence of range(0,1000)
        if select == 'linear':
            tseq = list(np.linspace(0, self.T-1, num_steps).astype(int))
        elif select == 'quadratic':
            tseq = list((np.linspace(0, np.sqrt(self.T), num_steps-1)**2).astype(int))
            tseq.insert(0, 0)
            tseq[-1] = self.T - 1
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{select}"')
        
        x_t = torch.randn(shape, device = self.device)
        tlist = torch.zeros([x_t.shape[0]], device = self.device)
        for i in tqdm(range(num_steps),dynamic_ncols=True, disable=(local_rank % torch.cuda.device_count() != 0)):
            with torch.no_grad():
                tlist = tlist * 0 + tseq[-1-i]
                if i != num_steps - 1:
                    prevt = torch.ones_like(tlist, device = self.device) * tseq[-2-i]
                else:
                    prevt = - torch.ones_like(tlist, device = self.device) 
                x_t = self.ddim_p_sample(x_t, tlist, prevt, eta, **model_kwargs)
                torch.cuda.empty_cache()
        x_t = torch.clamp(x_t, -1, 1)
        if local_rank == 0:
            print('ending sampling process(ddim)...')
        return x_t
    
    def trainloss(self, x_0:torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        calculate the loss of denoising diffusion probabilistic model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t = torch.randint(self.T, size = (x_0.shape[0],), device=self.device)
        x_t, eps = self.q_sample(x_0, t)
        pred_eps = self.model(x_t, t, **model_kwargs)
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        return loss
    
