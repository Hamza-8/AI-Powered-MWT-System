import copy
import torch
from microscan.gan import CycleGAN3D,GANTrainer,paired_complex_metrics

torch.set_num_threads(2)

def unchanged(before,module):
    return all(torch.equal(value,dict(module.named_parameters())[key]) for key,value in before.items())

def parameters(module):
    return {key:value.detach().clone() for key,value in module.named_parameters()}

def test_gan_updates_are_separate_and_resume():
    gan=CycleGAN3D({'channels':[4,8,8],'res_blocks':1,'spectral_norm':False})
    trainer=GANTrainer(gan,buffer_size=2)
    x,y=torch.randn(1,2,3,6,6),torch.randn(1,2,3,6,6)
    mask=torch.ones(1,1,3,6,6,dtype=torch.bool); frequency=torch.tensor([.6,.8,1.])
    before_D=parameters(gan.D_X)
    fake_x,fake_y,metrics=trainer.generator_step(x,y,mask,frequency,paired=True)
    assert unchanged(before_D,gan.D_X)
    before_G=parameters(gan.G)
    trainer.discriminator_step(x,y,fake_x,fake_y,mask,frequency)
    assert unchanged(before_G,gan.G)
    assert not unchanged(before_D,gan.D_X)
    state=copy.deepcopy(trainer.state_dict())
    other=GANTrainer(gan,buffer_size=2); other.load_state_dict(state)
    assert len(other.buffer_X.items)==1
    metrics=trainer.step(x,y,mask,mask,frequency,paired=False)
    assert metrics['paired'] is False
    assert 'complex_mae' not in metrics

def test_frequency_preserved_identity_and_hidden_isolation():
    gan=CycleGAN3D({'channels':[4,8,8],'res_blocks':1,'spectral_norm':False})
    for k in [3,10]:
        x=torch.randn(1,2,k,18,18); mask=torch.ones(1,1,k,18,18,dtype=torch.bool)
        mask[:,:,:,-2:]=False
        hidden=x.clone(); hidden[:,:,:,-2:]=float('nan')
        freq=torch.linspace(.6,1.05,k)
        a=gan.G(x,mask,freq); b=gan.G(hidden,mask,freq)
        assert a.shape==x.shape
        assert torch.equal(a,b)
        assert torch.isfinite(gan.D_X(a,mask,freq)).all()
        assert (a-torch.where(mask,x,torch.zeros_like(x))).abs().max()<.02

def test_phase_metrics_ignore_unobserved_and_zero_amplitudes():
    p=torch.zeros(1,2,3,2,2); t=p.clone(); mask=torch.ones(1,1,3,2,2,dtype=torch.bool)
    metrics=paired_complex_metrics(p,t,mask)
    assert metrics['phase_mae_rad'] is None
    assert metrics['complex_mae']==0
