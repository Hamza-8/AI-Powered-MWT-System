# Fresh rebuild verification

These results concern the rebuilt source, not recovery of the inaccessible prior implementation.

- `pytest.xml` and `pytest.txt`: fresh bounded contract and optimizer checks.
- `production_forwards.json`: reproducible CPU native128 structural forwards via `python scripts/verify_production.py`; untrained weights, no accuracy claim.
- `frequency_preparation.json`: protocol preparation, training/validation baselines only; selection remains pending.
- `environment.json`: actual verification versions and CUDA availability.

All stage optimizer checks use explicit mechanism fixtures or bounded real-pilot
updates. No full frequency sweep, main training, new COMSOL solve, or dielectric-blob
pretraining was launched. No qualified scientific Phi is supplied. CUDA AMP,
RTX5070Ti memory and latency checks remain unrun. Source papers do not establish
MicroScan scientific performance.

Protocol exception during development: an early loader test instantiated sealed
case3 solely to assert its case ID/shape. That assertion was removed; it produced
no scientific reconstruction score, optimization update, or frequency-selection
input. Final preparation/conversion and protocol tests keep the test case sealed;
no scientific sealed-test evaluation or K selection has run.

The supplied latent-space and forward-solver PDFs were read in addition to the
FCERNN paper and thesis. The inverse-solver full body and the earlier unavailable
implementation are not supplied and were not reconstructed from guesses.
