import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from glob import glob
import os
from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def exists(x):
    return x is not None

# 갯수만큼 그룹을 나눠서 return
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Dataset(Dataset):
    def __init__(self, folder, image_size, transforms=None): 
        super().__init__()
        self.paths = glob(os.path.join(folder, '*/*.png'))
        self.transforms = transforms
    
    def __len__(self): 
        return len(self.paths)  

    def __getitem__(self, idx): 
        path = self.paths[idx]
        
        img = Image.open(path) 
        if self.transforms is not None: 
            img = self.transforms(img) 
            
        return img
    
#### model
class ResnetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
    


class Unet(nn.Module):
    def __init__(self,
                 dim, 
                 init_dim, 
                 out_dim, 
                 dim_mults=(1, 2, 4, 8),
                 channels=3, 
                 self_condition=False, ## ??
                 resnet_block_group=8, 
                 learnd_variance = False,
                 learnd_fourier_features = False, 
                 learned_sinusoidal_sim = 16, 
                 full_attn = (False, False, False, True), 
                 flash_attn = False
                 ):
        super().__init__()
        
        self.channels = channels 
        self.self_condition = self_condition ## ?? 
        input_channels = channels * (2 if self_condition else 1)
        
        # default 앞. 혹은 뒤가 유용한 함수 혹은 변수 일경우 해당 데이터 return
        # 만약 둘다 유효하면 앞을 return
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        
        dims = [init_dim, *map(lambda m : dim * m, dim_mults)] # innit_Dim 부터 리스트를 만듬
        in_out = list(zip(dims[:-1], dims[1:]))  #zip 으로 짝을 맞춰서, (1, 2), (2, 3) ..이렇게
        
        block_klass = partial(ResnetBlock, groups=resnet_block_group)
        
    def forward(self, x):
        return x 
    
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x 
    

#### train
class Traniner(object): 
    def __init__(self, 
                 diffusion_model, 
                 folder, # train foler, 
                #  *,
                 train_batch_size = 32, 
                 gradient_accmulate_every = 1, # gradient 를 얼마 누적
                 augment_horizontal_flip = True, 
                 train_lr = 1e-4,
                 train_num_steps = 100000, 
                 ema_update_every = 10, 
                 ema_decay = 0.995, 
                 adam_betas = (0.9, 0.99),
                 save_and_sample_every=1000, 
                 num_samples = 25, 
                 results_folder = '/Users/sehwanyoo/Dropbox/Experiment/diffusion_study/outputs/basic',
                 amp = False, # automatic mixed precision ?? 이해 필요
                 mixed_precision = False, # mixed precision ?? 이해 필요
                 split_batches = True, # ??? 이해 필요 
                 convert_image_to = None, 
                 calculate_fid = True, # ??? 이해 필요
                 inception_block_idx = 2048, # ??? 이해 필요
                 max_grad_norm = 1., # ??? 이해 필요
                 num_fid_samples = 50000, # ??? 이해 필요
                 save_best_and_latest_only = False 
                 ):
        super().__init__()
        
        self.accelerator = Accelerator(
            split_batches = split_batches, # 배치를 여러개 나눠서 처리, 큰 배치를 처리 할수 없을 때 사용
            mixed_precision = mixed_precision if amp else 'no' # 혼합 정밀도 훈련 설정, float16 / 32 이 동시에 사용하여 연산 속도 증가
        )
        
        self.model = diffusion_model
        self.channels = diffusion_model.channels 
        
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        
        if save_best_and_latest_only: 
            self.best_fid = 1e10
            
        self.save_best_and_latest_only = save_best_and_latest_only
        
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accmulate_every
        
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size 
        
        self.max_grad_norm = max_grad_norm
        
        transforms = T.Comppose([
            T.Resize(self.image_size),
            T.ToTensor(),   
        ])
        
        ### dataset setting
        self.ds = Dataset(folder, self.image_size, transforms)
        dataloader = DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dataloader = self.accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)
        
        
        ### Optimier setting
        self.optimzer = Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)
        
        if self.accelerator.is_main_process: 
            self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)
            
        ### output setting 
        self.results_folder = results_folder 
        
        self.step = 0 
        
        
        #### preprepare model, optimizer, 데이터는 이미 위에서 완료
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
        ### FID 계산
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process
        
        if self.calculate_fid:
            if not self.model.is_ddim_sampling:
                self.accelerator.print(
                    'WARNING!'
                )
                
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl = self.dataloader, 
                sampler = self.ema.ema_model, # 이미지 생성 모델 지정, Ema 를 적용한 모델을 사용
                channels=self.channels, 
                accelerator=self.accelerator, 
                stats_dir = self.results_folder, 
                device=self.device, 
                num_fid_samples = num_fid_samples,  # FID 계산을 위해 생성할 샘플 이미지의 수 
                inception_block_idx = inception_block_idx # 
            )
            
            
    @property   
    def device(self): 
        return self.accelerator.device
    
    def save(self, milestone): 
        # if not self.accelerator.is_main_process: 
        if not self.accelerator.is_local_main_process: 
            return
        
        data = {
            'step' : self.step, 
            'model' : self.accelerator.get_state_dict(self.model), 
            'opt' : self.optimizer.state_dict(), 
            'ema' : self.ema.state_dict(), 
            'scalar' : self.accelerator.scaler.state_dict() if exists(self.accelerator) else None, 
        }
        
        torch.save(data, os.path.join(self.results_folder, f'model-{milestone}.pt'))
        
    def load(self, milestone):
        accelerator = self.accelerator
        device = self.device
        
        data = torch.load(os.path.join(self.results_folder, f'model-{milestone}.pt'), map_location=device)
        
        model = self.accelerator.unwrap_model(data) # Multi-gpu model 에서 사용하면, load 할 때 어렵기 때문에 그걸 고려하여 call 하는 함수
        
        ## 모두 dict 로 저장해서(save) 불러올 때도 dict 로
        model.load_state_dict(data['model'])
        
        self.step = data['step']
        self.optimizer.load_state_dict(data['optimizer'])
        
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])
            
        if exists(data['scalar']) and exists(self.accelerator.scaler):
            self.accelerator.scaler.load_state_dict(data['scalar'])
        
        
    def train(self): 
        accelerator = self.accelerator
        device = self.device
        
        with tqdm(initial=self.step, total=self.train_num_steps, disable= not accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps: 
                total_loss = 0 
                
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dataloder).to(device) 
                    
                    with self.accelerator.autocast(): 
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # gradient 가 너무 커져서 학습적 과정에 수치적 불안정성을 일으키는 것을 방지, 임계값을 초과하지 않도록 조절 
                pbar.set_description(f'loss : {total_loss:.4f}')
                
                accelerator.wait_for_everyone() # 분산 환경에서 waiting
                
                ##### !!!! 확인 필요
                self.optimizer.step() 
                self.optimizer.zero_grad()
                
                accelerator.wait_for_everyone()
                
                
                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update() 
                    
                    if self.step != 0 and self.step % self.save_and_sample_every == 0: 
                        self.ema.ema_model.eval()
                        
                        # torch_no_grad() 와 달리 추가적인 view 연산에 대한 메모리 할당 최적화
                        with torch.inference_model():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                            
                        all_images = torch.cat(all_images_list, dim=0)
                        
                        utils.save_image(all_images, os.path.join(self.results_folder, f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))
                        
                        if self.calculate_fid: 
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid score: {fid_score}')
                            
                        if self.save_best_and_latest_only: 
                            if self.best_fid > fid_score: 
                                self.best_fid = fid_score
                                self.save('best')
                            self.save('latest')
                            
                        else:
                            self.save(milestone)
                            
                    pbar.update(1)
                    
        accelerator.print('training complete')
                                
                            
                            
                
                
                
                