"""Bounded mechanism checks on explicit fixtures; no scientific accuracy claims."""
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from microscan.checkpoint import load_checkpoint, save_checkpoint
from microscan.models import MSRNN, tiny_config
from microscan.training import train, infer, load_deployment, profile, prepare_batch


def manifest_fixture(tmp_path):
    samples = []
    for index, split in enumerate(("train", "train", "validation")):
        name = f"sample_{index}"
        signal = np.full((10, 4, 4), (index + 1) * .001 + .002j, np.complex128)
        target = np.zeros((2, 16, 16, 16), np.float32)
        target[0] = 1
        target[:, 5:9, 5:9, 5:9] = np.asarray([45, .8])[:, None, None, None]
        np.save(tmp_path / f"{name}_s.npy", signal)
        np.save(tmp_path / f"{name}_x.npy", target)
        labels = np.zeros((16, 16, 16), np.int64)
        labels[5:9, 5:9, 5:9] = 1
        np.save(tmp_path / f"{name}_labels.npy", labels)
        np.save(tmp_path / f"{name}_valid.npy", np.ones_like(labels, bool))
        samples.append(dict(id=name, split=split, subject_id=name, head_id=name, geometry_id=name,
                            sequence_id=name, source_domain="explicit_mechanism_fixture", domain="calibrated_delta_s",
                            signal_path=f"{name}_s.npy", target_path=f"{name}_x.npy", label_path=f"{name}_labels.npy",
                            label_validity_path=f"{name}_valid.npy", label_schema="multiclass",
                            grid=dict(shape_zyx=[16,16,16], bounds_xyz_mm=[[-60,60]]*3, array_axes="ZYX", coordinate_axes="XYZ", units="mm"),
                            frequencies_ghz=[.6+.05*i for i in range(10)], physical_ports=[1,2,3,4], reference_frequency_ghz=.85))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(schema_version=1, samples=samples)))
    return path


def training_config(tmp_path, stage="enhance"):
    return dict(stage=stage, manifest=str(manifest_fixture(tmp_path)), output=str(tmp_path / "training"),
                device="cpu", verification_only=True, model=tiny_config(), epochs=1, batch_size=1, seed=21, amp=False, physics_mode="A")


def test_atomic_checkpoint_restores_optimizer_scheduler_scaler_rng(tmp_path):
    torch.manual_seed(19)
    model = torch.nn.Linear(2,1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model(torch.ones(1,2)).sum().backward(); optimizer.step(); scheduler.step()
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, {"o":optimizer},{"s":scheduler},scaler,epoch=2,step_in_epoch=3,global_step=8,
                    metadata={"run_identity":"test","physics_requested":"B","physics_active":"B","inference_requirements":{"phi_required":False}})
    expected = torch.rand(5)
    model.weight.data.zero_()
    state = load_checkpoint(path, model,{"o":optimizer},{"s":scheduler},scaler,expected_identity="test")
    assert torch.equal(torch.rand(5),expected)
    assert state["global_step"] == 8 and state["step_in_epoch"] == 3
    assert model.weight.abs().sum() > 0 and optimizer.state
    assert state["metadata"]["inference_requirements"]["phi_required"] is False
    with pytest.raises(ValueError,match="identity"):
        load_checkpoint(path,expected_identity="wrong")


def test_real_stage_bounded_update_resume_exact(tmp_path):
    torch.set_num_threads(1)
    config = training_config(tmp_path)
    report = train(config,max_steps=1)
    assert report["status"] == "bounded_interruption"
    state = load_checkpoint(report["checkpoint"],restore_random=False)
    assert state["metadata"]["preprocessing"]["fitted_geometry_ids"] == ["sample_0","sample_1"]
    report = train(config,resume=True)
    final = load_checkpoint(report["checkpoint"],restore_random=False)
    # Compare a fresh run with same seed and data, separate output.
    config["output"] = str(tmp_path / "uninterrupted")
    other = load_checkpoint(train(config)["checkpoint"],restore_random=False)
    assert final["global_step"] == other["global_step"] == 2
    assert all(torch.equal(value,other["model"][key]) for key,value in final["model"].items())


def test_stage_train_infer_segmentation_and_joint(tmp_path):
    torch.set_num_threads(1)
    config = training_config(tmp_path)
    base = train(config,max_steps=1)
    config.update(stage="segmentation",initialize_from=base["checkpoint"],output=str(tmp_path/"seg"),
                  segmentation={"widths":[4,8,8],"classes":5,"label_mode":"multiclass"})
    report = train(config,max_steps=1)
    state = load_checkpoint(report["checkpoint"],restore_random=False)
    inverse_keys = [k for k in state["model"] if k.startswith("inverse.")]
    prior = load_checkpoint(base["checkpoint"],restore_random=False)
    assert all(torch.equal(state["model"][k],prior["model"][k]) for k in inverse_keys)
    output = tmp_path / "infer"
    result = infer(report["checkpoint"],config["manifest"],output,split="validation")
    assert result["cases"] == 1
    with np.load(output/"case_00000.npz") as saved:
        assert saved["segmentation_probabilities"].shape == (1,5,16,16,16)
        assert "calibrated" in saved and "decluttered" in saved
    assert (output/"case_00000_components.json").is_file()
    validity = json.loads((output/"case_00000_validity.json").read_text())
    assert validity["absolute_valid"] and not validity["probability_calibration_validated"]
    assert not json.loads((output/"case_00000_tracking.json").read_text())["baseline_valid"]
    from microscan.training import evaluate
    metrics = evaluate(report["checkpoint"],config["manifest"],split="validation")
    assert metrics["refined_mean"] and metrics["per_sample"][0]["lesion_metrics"]
    config.update(stage="joint",initialize_from=report["checkpoint"],output=str(tmp_path/"joint"),freeze_schedule=[{"epoch":0,"train":["inverse","segmentation"]}])
    assert train(config,max_steps=1)["global_step"] == 1


def test_required_physics_and_unselected_grid_fail_closed(tmp_path):
    config = training_config(tmp_path)
    config["physics_mode"] = "B"
    with pytest.raises(ValueError,match="Phi"):
        train(config,max_steps=1)
    config["physics_mode"] = "A"
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"status":"pending_selection"}))
    config["frequency_config_path"] = str(frozen)
    with pytest.raises(ValueError,match="frozen"):
        train(config,max_steps=1)


def test_prepare_masks_hidden_nan_and_frozen_slots():
    signals = torch.ones(1,2,10,4,4)
    signals[:,:,4:] = float("nan")
    mask = torch.ones(1,1,10,4,4,dtype=torch.bool)
    batch = prepare_batch({"signals":signals,"mask":mask},None,{"frequency_config":{"indices_zero_based":[0,1,2,3]}})
    assert torch.isfinite(batch["signals"]).all() and not batch["mask"][:,:,4:].any()


def test_cuda_profile_refuses_unavailable_device():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match="CUDA unavailable"):
            profile({"model":tiny_config()},device="cuda",steps=1)


def test_phi_and_gan_stage_commands_resume_and_unqualified_report(tmp_path):
    torch.set_num_threads(1)
    config = training_config(tmp_path, "phi")
    manifest = json.loads(Path(config["manifest"]).read_text())
    for sample in manifest["samples"]:
        sample["physics_context"] = dict(tx_positions_mm=[[0,0,0],[1,0,0],[2,0,0],[3,0,0]],
                                         rx_positions_mm=[[0,0,0],[1,0,0],[2,0,0],[3,0,0]],
                                         source_excitation=[[1,0]]*4,constant_spectrum=True,
                                         layout_id="fixture",calibration_id="fixture",source_id="fixture")
        sample["clean_signal_path"] = sample["signal_path"]
        sample["clean_pairing"] = {"aligned":True,"pair_id":sample["id"]}
    Path(config["manifest"]).write_text(json.dumps(manifest))
    config["physics"] = {"operator":{"width":2,"latent_dim":8},"nondispersive_labels_verified":True}
    result = train(config,max_steps=1)
    assert result["global_step"] == 1
    from microscan.training import qualify
    report = qualify(result["checkpoint"],config["manifest"],{"scientific":False,"property_bounds":[[0,100],[0,5]]},output=tmp_path/"qualification.json")
    assert report["qualified"] is False and "synthetic_fixture_not_scientific_qualification" in report["reasons"]
    config.update(stage="gan",output=str(tmp_path/"gan"),gan={"channels":[4,8],"discriminator_channels":[4,8],"res_blocks":1},gan_paired=True)
    result = train(config,max_steps=2)
    # Resume after last batch still runs actual held-out validation, no undefined training terms.
    resumed = train(config,resume=True)
    state = load_checkpoint(resumed["checkpoint"],restore_random=False)
    assert state["epoch"] == 1 and state["extra"]["gan_trainer"]
    assert set(state["schedulers"]) == {"generator","discriminators"}


def test_resume_complete_checkpoint_does_not_reopen_initializer(tmp_path):
    torch.set_num_threads(1)
    config = training_config(tmp_path)
    initial = train(config,max_steps=1)["checkpoint"]
    config.update(stage="segmentation",initialize_from=initial,output=str(tmp_path/"seg_resume"),segmentation={"widths":[4,8,8]})
    train(config,max_steps=1)
    Path(initial).unlink()
    assert train(config,resume=True)["status"] == "completed"
