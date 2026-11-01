"""Protocol mechanism tests; no full frequency experiment is launched."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from microscan import frequency as f

CSV = Path(__file__).parents[1] / "data/pilot/frequency_subsets.csv"
MAT = Path(__file__).parents[1] / "data/pilot/frequency_pilot.mat"


class FrequencyProtocolTests(unittest.TestCase):
    def config(self):
        return {"seeds": [17, 29], "epochs": 1, "batch_size": 1, "physics_mode": "A"}

    def results(self, ten=.10, others=.104):
        results=[]
        for k in range(3,11):
            for seed in self.config()["seeds"]:
                metric={"balanced_contrast_rmse": ten if k==10 else others,
                        "localization_error_mm": 2., "relative_volume_error": .1}
                results.append({"status":"completed", "test_accessed":False,"k":k,"seed":seed,
                    "physics_requested":"A","physics_active":"none","parameter_count":123,
                    "validation":{"refined":metric,"coarse":metric},
                    "baselines":{"zero_contrast":{"balanced_contrast_rmse":.7},
                                 "training_mean_volume":{"balanced_contrast_rmse":.5}}})
        for row in results:
            row["identity"]={"config":self.config(),"k":row["k"],"seed":row["seed"],
                "indices_zero_based":list(range(row["k"])),"data_sha256":"same_data",
                "source":{"same.py":"same_code"},"preprocessing":{"algorithm":"same_policy","scale":row["k"]}}
            row["identity_sha256"]=f.canonical_hash(row["identity"])
        return results

    def test_csv_matlab_index_conversion(self):
        subsets=f.read_subsets(CSV)
        self.assertEqual(subsets[3], [0,5,9])
        self.assertEqual(subsets[10], list(range(10)))
        self.assertEqual(len(subsets),8)

    def test_config_disallows_physics_gan_segmentation(self):
        for key,value in [("physics_mode","B"),("use_cyclegan",True),("use_segmentation",True),("fixed_frequency_slots",3)]:
            cfg=self.config();cfg[key]=value
            with self.assertRaises(ValueError): f.validate_config(cfg)

    def test_hidden_nan_and_zero_are_not_missing(self):
        signal=torch.tensor([[[[0.,2.,float('nan')]]],[[[0.,4.,float('nan')]]]])
        mask=torch.tensor([[[[True,True,False]]]])
        scale=f.fit_input_scale([{"signals":signal,"mask":mask}])
        self.assertEqual(scale["observed_real_values"],4)
        self.assertAlmostEqual(scale["scale"],5**.5)
        self.assertTrue(torch.isfinite(f.transform_signal(signal,mask,scale)).all())
        self.assertEqual(f.transform_signal(signal,mask,scale)[0,0,0,2],0)

    def test_balanced_metric_not_empty_background_accuracy(self):
        target=torch.zeros(2,8,8,8);target[0]=1
        target[0,2:4,2:4,2:4]=45;target[1,2:4,2:4,2:4]=.8
        zero=torch.zeros_like(target);zero[0]=1
        metric=f.reconstruction_metrics(zero,target)
        self.assertAlmostEqual(metric["balanced_contrast_rmse"],2**-.5)
        self.assertEqual(metric["relative_volume_error"],1)
        self.assertEqual(f.reconstruction_metrics(target,target)["balanced_contrast_rmse"],0)

    def test_select_smallest_eligible_and_inconclusive(self):
        selected=f.select_results(self.results(),self.config())
        self.assertEqual(selected["selected_k"],3)
        bad=f.select_results(self.results(ten=.6),self.config())
        self.assertEqual(bad["status"],"inconclusive")
        self.assertIsNone(bad["selected_k"])

    def test_localization_and_seed_budget_gate(self):
        runs=self.results()
        for row in runs:
            if row["k"]==3:row["validation"]["refined"]["localization_error_mm"]=11
        self.assertEqual(f.select_results(runs,self.config())["selected_k"],4)
        with self.assertRaises(ValueError):f.select_results(runs[:-1],self.config())
        runs=self.results();runs[0]["parameter_count"]=124
        with self.assertRaises(ValueError):f.select_results(runs,self.config())
        runs=self.results();runs[0]["test_accessed"]=True
        with self.assertRaises(ValueError):f.select_results(runs,self.config())

    def test_cross_candidate_code_and_budget_mismatch_rejected(self):
        runs=self.results()
        runs[0]["identity"]["source"]["same.py"]="different_code"
        runs[0]["identity_sha256"]=f.canonical_hash(runs[0]["identity"])
        with self.assertRaises(ValueError):f.select_results(runs,self.config())
        runs=self.results()
        runs[0]["identity"]["config"]["epochs"]=7
        runs[0]["identity_sha256"]=f.canonical_hash(runs[0]["identity"])
        with self.assertRaises(ValueError):f.select_results(runs,self.config())

    def test_fixed_frequency_grid_cannot_be_overridden(self):
        cfg=self.config();cfg["model"]={"frequency_grid_ghz":[.6,.85,1.05]}
        with self.assertRaises(ValueError):f.validate_config(cfg)
        cfg=self.config();cfg["model"]={"correction_depth":0}
        with self.assertRaises(ValueError):f.validate_config(cfg)

    def test_numerical_execution_is_part_of_identity(self):
        cpu=f.execution_identity("cpu",True)
        self.assertFalse(cpu["amp_active"])
        self.assertEqual(cpu["device_type"],"cpu")
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/"data.mat";data.write_bytes(b"A")
            a=f.run_identity(self.config(),data,CSV,3,17,{"scale":1.},cpu)
            changed=dict(cpu,torch_version="different")
            b=f.run_identity(self.config(),data,CSV,3,17,{"scale":1.},changed)
            self.assertNotEqual(f.canonical_hash(a),f.canonical_hash(b))

    def test_data_and_preprocessing_in_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data.mat";p.write_bytes(b"A")
            a=f.run_identity(self.config(),p,CSV,3,17,{"scale":1.})
            p.write_bytes(b"B")
            b=f.run_identity(self.config(),p,CSV,3,17,{"scale":1.})
            self.assertNotEqual(f.canonical_hash(a),f.canonical_hash(b))
            b=f.run_identity(self.config(),p,CSV,3,17,{"scale":2.})
            self.assertNotEqual(f.canonical_hash(a),f.canonical_hash(b))

    def test_prepare_never_requests_sealed_test(self):
        sample={"signals":torch.zeros(2,10,2,2),"mask":torch.ones(1,10,2,2,dtype=torch.bool),
                "target":torch.zeros(2,4,4,4)}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(f,"_dataset",return_value=[sample]) as loader:
            data=Path(tmp)/"x.mat";data.write_bytes(b"fixture")
            f.prepare(self.config(),data,CSV,tmp)
            self.assertEqual([call.args[1] for call in loader.call_args_list],["validation","train"])
            self.assertFalse(json.loads((Path(tmp)/"protocol.json").read_text())["test_accessed"])

    def test_selection_locked_after_test_is_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            f.atomic_json(Path(tmp)/"sealed_test_results.json",{"test_accessed":True})
            with self.assertRaises(PermissionError):f.select_and_freeze(self.config(),MAT,CSV,tmp)

    def test_sealed_test_requires_selection_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            f.atomic_json(Path(tmp)/"selection.json",{"status":"inconclusive"})
            f.atomic_json(Path(tmp)/"frozen_frequencies.json",{"status":"pending_selection"})
            with self.assertRaises(ValueError):f.evaluate_sealed_test(self.config(),MAT,CSV,tmp)

    def test_equal_frequency_training_budgets_required(self):
        cfg=self.config();cfg.update(epochs=3,patience=1)
        with self.assertRaises(ValueError):f.validate_config(cfg)

    def test_failed_test_read_still_locks_selection(self):
        # Mechanism fixture: validates artifact plumbing without scientific training
        # or accessing the actual sealed geometry.
        sample={"signals":torch.ones(2,10,2,2),"mask":torch.ones(1,10,2,2,dtype=torch.bool)}
        cfg=self.config();pre=f.fit_input_scale([sample])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(f,"source_identity",return_value=f.source_identity()):
            root=Path(tmp);all_results=[]
            for k in range(3,11):
                for seed in cfg["seeds"]:
                    identity=f.run_identity(cfg,MAT,CSV,k,seed,pre)
                    runroot=root/f"K{k:02d}"/f"seed_{seed}"
                    saved={"model":{},"preprocessing":pre,"identity":identity,"identity_sha256":f.canonical_hash(identity)}
                    f.atomic_checkpoint(runroot/"best.pt",saved)
                    result={"k":k,"seed":seed,"identity_sha256":saved["identity_sha256"],
                            "best_checkpoint_sha256":f.file_sha256(runroot/"best.pt")}
                    f.atomic_json(runroot/"result.json",result);all_results.append(result)
            report={"status":"selected","selected_k":3,"results_sha256":f.canonical_hash(all_results)}
            f.atomic_json(root/"selection.json",report)
            indices=f.read_subsets(CSV)[3]
            frozen={"status":"frozen","selected_k":3,"selection_sha256":f.canonical_hash(report),
                "data_sha256":f.file_sha256(MAT),"csv_sha256":f.file_sha256(CSV),
                "indices_zero_based":indices,"grid_ghz":f.GRID_GHZ,"frequencies_ghz":[f.GRID_GHZ[i] for i in indices]}
            f.atomic_json(root/"frozen_frequencies.json",frozen)
            def loader(mat,split,*args,**kwargs):
                if split=="test":raise RuntimeError("interrupted while opening test")
                return [sample]
            with mock.patch.object(f,"_dataset",side_effect=loader):
                with self.assertRaisesRegex(RuntimeError,"interrupted"):
                    f.evaluate_sealed_test(cfg,MAT,CSV,tmp)
            self.assertTrue((root/"sealed_test_access.json").exists())
            with self.assertRaises(PermissionError):f.select_and_freeze(cfg,MAT,CSV,tmp)

    def test_bounded_real_candidate_resume_preserves_rng_and_rejects_identity_change(self):
        from microscan.models import tiny_config
        cfg=self.config();cfg.update(model=tiny_config(),grid_size=16,seeds=[17],epochs=2,amp=False)
        # Other agents may edit unrelated package files during this bounded test;
        # freeze the source snapshot within this single in-process mechanism run.
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(f,"source_identity",return_value=f.source_identity()):
            first=f.run_candidate(cfg,MAT,CSV,tmp,3,17,max_updates=1)
            self.assertEqual(first["status"],"running")
            resumed=f.run_candidate(cfg,MAT,CSV,tmp,3,17,max_updates=1,resume=True)
            self.assertEqual(resumed["updates"],2)
            checkpoint=torch.load(Path(tmp)/"K03/seed_17/resume.pt",weights_only=False)
            self.assertIn("scaler",checkpoint)
            self.assertIn("numpy",checkpoint["rng"])
            self.assertEqual(checkpoint["inference_requirements"]["phi"],False)
            with tempfile.TemporaryDirectory() as direct:
                f.run_candidate(cfg,MAT,CSV,direct,3,17,max_updates=2)
                uninterrupted=torch.load(Path(direct)/"K03/seed_17/resume.pt",weights_only=False)
                for name,value in checkpoint["model"].items():
                    self.assertTrue(torch.equal(value,uninterrupted["model"][name]),name)
            cfg["learning_rate"]=.1
            with self.assertRaises(ValueError):f.run_candidate(cfg,MAT,CSV,tmp,3,17,resume=True,max_updates=1)


if __name__=="__main__": unittest.main()
