"""Bounded native-grid structural checks. Does not train or open sealed samples."""
import json
from pathlib import Path
import platform
import time
import torch
from microscan.models import MSRNN, SegmentationUNet
from microscan.gan import CycleGAN3D
from microscan.training import source_fingerprint


def main():
    torch.manual_seed(17); torch.set_num_threads(1)
    report = {"scope": "fresh structural CPU eval/no_grad only; untrained weights; not scientific accuracy or CUDA validation",
              "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "source_sha256": source_fingerprint(), "forwards": {}}
    with torch.no_grad():
        model = MSRNN().eval()
        shapes={}
        encoder_hook=model.coarse.encoder.register_forward_hook(lambda module,args,result: shapes.update(encoder_output_shape=list(result.shape)))
        adapter_hook=model.coarse.adapter.register_forward_hook(lambda module,args,result: shapes.update(adapter_shape=list(result.shape)))
        y=torch.zeros(1,2,10,18,18); mask=torch.ones(1,1,10,18,18,dtype=torch.bool)
        start=time.perf_counter(); out=model(y,mask)
        record={"input_shape":list(y.shape), "coarse_shape":list(out['coarse'].shape),
                "refined_shape":list(out['refined'].shape), "parameters":sum(p.numel() for p in model.parameters()),
                "seconds":time.perf_counter()-start,"finite":bool(torch.isfinite(out['refined']).all()),
                **shapes,"latent_shape":[1,model.coarse.latent_channels,*([model.coarse.latent_size]*3)]}
        encoder_hook.remove();adapter_hook.remove()
        assert record['encoder_output_shape']==[1,256,51] and record['adapter_shape']==[1,1024,32]
        assert record['coarse_shape']==record['refined_shape']==[1,2,128,128,128] and record['finite']
        report['forwards']['MSRNN']=record
        seg=SegmentationUNet().eval(); start=time.perf_counter(); logits=seg(out['refined'])
        record={"input_shape":list(out['refined'].shape),"output_shape":list(logits.shape),
                "parameters":sum(p.numel() for p in seg.parameters()),"seconds":time.perf_counter()-start,
                "finite":bool(torch.isfinite(logits).all())}
        assert record['output_shape']==[1,5,128,128,128] and record['finite']
        report['forwards']['segmentation']=record
        del seg,model,out,logits
        gan=CycleGAN3D().eval()
        for k in (3,10):
            y=torch.zeros(1,2,k,18,18);mask=torch.ones(1,1,k,18,18,dtype=torch.bool);f=torch.linspace(.6,1.05,k)
            start=time.perf_counter(); generated=gan.G(y,mask,f);disc=gan.D_X(generated,mask,f)
            record={"input_shape":list(y.shape),"generator_output_shape":list(generated.shape),
                    "discriminator_output_shape":list(disc.shape),"seconds":time.perf_counter()-start,
                    "finite":bool(torch.isfinite(generated).all() and torch.isfinite(disc).all())}
            record.update({name+'_parameters':sum(p.numel() for p in getattr(gan,name).parameters()) for name in ('G','F','D_X','D_Y')})
            assert generated.shape==y.shape and disc.shape[2]==k and record['finite']
            report['forwards']['GAN_'+str(k)]=record
    if platform.system()=='Linux':
        import resource
        report['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    path=Path('docs/verification/production_forwards.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
