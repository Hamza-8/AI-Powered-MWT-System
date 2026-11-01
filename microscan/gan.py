"""Mask-conditioned complex 3D CycleGAN with independent training components.

Adapts residual-attention mechanisms of the original branch to frequency/Rx/Tx;
frequency is never downsampled, and no tanh clips signed complex measurements.
"""
from __future__ import annotations
import random
import torch
from torch import nn
from torch.nn import functional as F
from .models import ResidualBlock3D, groups, match_shape, measurement_features, sanitize_measurements


class AttentionGate3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden=max(1,channels//2)
        self.skip=nn.Conv3d(channels,hidden,1)
        self.gate=nn.Conv3d(channels,hidden,1)
        self.score=nn.Conv3d(hidden,1,1)
    def forward(self,skip,gate):
        return skip*torch.sigmoid(self.score(F.silu(self.skip(skip)+self.gate(gate))))


class AttentionGenerator3D(nn.Module):
    def __init__(self,widths=(32,64,128,256),res_blocks=4):
        super().__init__()
        self.encoders,self.downs=nn.ModuleList(),nn.ModuleList()
        previous=6
        for i,width in enumerate(widths):
            self.encoders.append(ResidualBlock3D(previous,width))
            if i<len(widths)-1:
                self.downs.append(nn.Conv3d(width,widths[i+1],3,stride=(1,2,2),padding=1))
                previous=widths[i+1]
        self.bottleneck=nn.Sequential(*(ResidualBlock3D(widths[-1],widths[-1]) for _ in range(res_blocks)))
        self.ups,self.attention,self.decoders=nn.ModuleList(),nn.ModuleList(),nn.ModuleList()
        for a,b in zip(reversed(widths[1:]),reversed(widths[:-1])):
            self.ups.append(nn.ConvTranspose3d(a,b,(1,2,2),stride=(1,2,2)))
            self.attention.append(AttentionGate3D(b))
            self.decoders.append(ResidualBlock3D(2*b,b))
        self.output=nn.Conv3d(widths[0],2,1)
        nn.init.normal_(self.output.weight,std=1e-4)
        nn.init.zeros_(self.output.bias)
    def forward(self,values,mask,frequencies_ghz):
        clean=sanitize_measurements(values,mask)
        x=measurement_features(values,mask,frequencies_ghz)
        skips=[]
        for i,block in enumerate(self.encoders):
            x=block(x)
            if i<len(self.downs):
                skips.append(x)
                x=self.downs[i](x)
        x=self.bottleneck(x)
        for up,attention,decoder,skip in zip(self.ups,self.attention,self.decoders,reversed(skips)):
            x=match_shape(up(x),skip.shape[2:])
            x=decoder(torch.cat((x,attention(skip,x)),1))
        return torch.where(mask.bool(),clean+self.output(x),torch.zeros_like(clean))


class PatchGAN3D(nn.Module):
    def __init__(self,widths=(32,64,128,256),spectral_norm=True):
        super().__init__()
        layers=[]
        previous=6
        for width in widths:
            conv=nn.Conv3d(previous,width,3,stride=(1,2,2),padding=1)
            if spectral_norm: conv=nn.utils.spectral_norm(conv)
            layers.extend((conv,nn.LeakyReLU(.2)))
            previous=width
        layers.append(nn.Conv3d(previous,1,3,padding=1))
        self.layers=nn.Sequential(*layers)
    def forward(self,values,mask,frequencies_ghz):
        return self.layers(measurement_features(values,mask,frequencies_ghz))


class CycleGAN3D(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        c=config or {}
        widths=c.get("channels",(32,64,128,256))
        self.G=AttentionGenerator3D(widths,c.get("res_blocks",4))
        self.F=AttentionGenerator3D(widths,c.get("res_blocks",4))
        self.D_X=PatchGAN3D(c.get("discriminator_channels",widths),c.get("spectral_norm",True))
        self.D_Y=PatchGAN3D(c.get("discriminator_channels",widths),c.get("spectral_norm",True))
    def forward(self,x,mask,frequencies_ghz):
        return self.G(x,mask,frequencies_ghz)


class ReplayBuffer:
    def __init__(self,capacity=50):
        self.capacity=capacity
        self.items=[]
    def query(self,batch):
        output=[]
        for item in batch.detach():
            item=item.unsqueeze(0)
            if not self.capacity:
                output.append(item)
            elif len(self.items)<self.capacity:
                self.items.append(item.cpu().clone())
                output.append(item)
            elif random.random()>.5:
                index=random.randrange(self.capacity)
                previous=self.items[index].to(item)
                self.items[index]=item.cpu().clone()
                output.append(previous)
            else: output.append(item)
        return torch.cat(output)
    def state_dict(self):
        return {"capacity":self.capacity,"items":self.items}
    def load_state_dict(self,state):
        self.capacity=state["capacity"]
        self.items=state["items"]


def masked_l1(prediction,target,mask):
    support=mask.bool().expand_as(prediction)
    diff=torch.where(support,prediction-target,torch.zeros_like(prediction))
    return diff.abs().sum()/support.sum().clamp_min(1)


def paired_complex_metrics(prediction,target,mask,amplitude_floor=1e-8):
    p=torch.complex(prediction[:,0],prediction[:,1])
    t=torch.complex(target[:,0],target[:,1])
    support=mask[:,0].bool()
    amplitude=(p.abs()-t.abs()).abs()
    phase=torch.angle(p*t.conj()).abs()
    phase_support=support & (t.abs()>amplitude_floor) & (p.abs()>amplitude_floor)
    return dict(amplitude_mae=amplitude[support].mean().item() if support.any() else None,
                phase_mae_rad=phase[phase_support].mean().item() if phase_support.any() else None,
                complex_mae=masked_l1(prediction,target,mask).item(),paired=True)


class GANTrainer:
    def __init__(self,gan,lr=2e-4,lambda_cycle=10.,lambda_identity=5.,lambda_paired=1.,lambda_differential=1.,buffer_size=50):
        self.gan=gan
        self.lambda_cycle,self.lambda_identity=lambda_cycle,lambda_identity
        self.lambda_paired,self.lambda_differential=lambda_paired,lambda_differential
        self.optimizer_G=torch.optim.Adam(list(gan.G.parameters())+list(gan.F.parameters()),lr=lr,betas=(.5,.999))
        self.optimizer_D=torch.optim.Adam(list(gan.D_X.parameters())+list(gan.D_Y.parameters()),lr=lr,betas=(.5,.999))
        self.buffer_X,self.buffer_Y=ReplayBuffer(buffer_size),ReplayBuffer(buffer_size)
        self.steps=0
    def generator_step(self,x,y,mask,frequencies_ghz,paired=False,pair_x=None,pair_y=None):
        gan=self.gan
        for discriminator in (gan.D_X,gan.D_Y):
            discriminator.eval()
            discriminator.requires_grad_(False)
        self.optimizer_G.zero_grad(set_to_none=True)
        fake_y=gan.G(x,mask,frequencies_ghz)
        fake_x=gan.F(y,mask,frequencies_ghz)
        cycle_x=gan.F(fake_y,mask,frequencies_ghz)
        cycle_y=gan.G(fake_x,mask,frequencies_ghz)
        adv=((gan.D_Y(fake_y,mask,frequencies_ghz)-1).square().mean()+(gan.D_X(fake_x,mask,frequencies_ghz)-1).square().mean())
        cycle=masked_l1(cycle_x,x,mask)+masked_l1(cycle_y,y,mask)
        identity=masked_l1(gan.G(y,mask,frequencies_ghz),y,mask)+masked_l1(gan.F(x,mask,frequencies_ghz),x,mask)
        paired_loss=masked_l1(fake_y,y,mask) if paired else fake_y.sum()*0
        differential=fake_y.sum()*0
        if pair_x is not None or pair_y is not None:
            if pair_x is None or pair_y is None or not paired:
                raise ValueError("Differential supervision requires verified aligned current/baseline pairs")
            predicted_difference=fake_y-gan.G(pair_x,mask,frequencies_ghz)
            differential=masked_l1(predicted_difference,y-pair_y,mask)
        loss=adv+self.lambda_cycle*cycle+self.lambda_identity*identity+self.lambda_paired*paired_loss+self.lambda_differential*differential
        loss.backward()
        self.optimizer_G.step()
        return fake_x.detach(),fake_y.detach(),dict(generator=float(loss.detach()),adversarial=float(adv.detach()),cycle=float(cycle.detach()),identity=float(identity.detach()),paired_loss=float(paired_loss.detach()),differential=float(differential.detach()))
    def discriminator_step(self,x,y,fake_x,fake_y,mask,frequencies_ghz):
        gan=self.gan
        for discriminator in (gan.D_X,gan.D_Y):
            discriminator.train()
            discriminator.requires_grad_(True)
        self.optimizer_D.zero_grad(set_to_none=True)
        # Replay stores image+mask together, preserving matching mask distributions.
        pack_x=self.buffer_X.query(torch.cat((fake_x,mask.to(fake_x)),1))
        pack_y=self.buffer_Y.query(torch.cat((fake_y,mask.to(fake_y)),1))
        replay_mask_x,replay_mask_y=pack_x[:,2:3],pack_y[:,2:3]
        # Reuse replay masks on real examples too: no domain-specific mask shortcut.
        real_x=sanitize_measurements(x,replay_mask_x.bool() & mask.bool())
        real_y=sanitize_measurements(y,replay_mask_y.bool() & mask.bool())
        common_x=replay_mask_x.bool() & mask.bool()
        common_y=replay_mask_y.bool() & mask.bool()
        loss=.5*((gan.D_X(real_x,common_x,frequencies_ghz)-1).square().mean()+gan.D_X(pack_x[:,:2],common_x,frequencies_ghz).square().mean()+
                  (gan.D_Y(real_y,common_y,frequencies_ghz)-1).square().mean()+gan.D_Y(pack_y[:,:2],common_y,frequencies_ghz).square().mean())
        loss.backward()
        self.optimizer_D.step()
        return float(loss.detach())
    def step(self,x,y,mask_x,mask_y=None,frequencies_ghz=None,paired=False,pair_x=None,pair_y=None):
        if frequencies_ghz is None: raise ValueError("Explicit frequency coordinates required")
        if mask_y is None: mask_y=mask_x
        mask=mask_x.bool() & mask_y.bool()
        if not mask.any(): raise ValueError("GAN batch has no common observed support")
        x,y=sanitize_measurements(x,mask),sanitize_measurements(y,mask)
        fake_x,fake_y,metrics=self.generator_step(x,y,mask,frequencies_ghz,paired,pair_x,pair_y)
        metrics["discriminator"]=self.discriminator_step(x,y,fake_x,fake_y,mask,frequencies_ghz)
        if paired: metrics.update(paired_complex_metrics(fake_y,y,mask))
        else: metrics["paired"]=False
        self.steps+=1
        return metrics
    def state_dict(self):
        return dict(optimizer_G=self.optimizer_G.state_dict(),optimizer_D=self.optimizer_D.state_dict(),buffer_X=self.buffer_X.state_dict(),buffer_Y=self.buffer_Y.state_dict(),steps=self.steps)
    def load_state_dict(self,state):
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        self.buffer_X.load_state_dict(state["buffer_X"])
        self.buffer_Y.load_state_dict(state["buffer_Y"])
        self.steps=state["steps"]
