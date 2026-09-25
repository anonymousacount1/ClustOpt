# External benchmark

Run it with `python experiments/external_benchmark/run.py --help`. The step order is in [`docs/reproduction.md`](../../docs/reproduction.md), Level 3.

- **Protocol and arm definitions:** `configs/external_benchmark/`.
- **Baselines:** [`baselines/README.md`](../../baselines/README.md).
- **Frozen results:** [`results/external/`](../../results/external/README.md).
- **CLUSTOPT as reported (C4) on one dataset:** `python -m clustopt.pipeline --input <file>`; see the README.
- **Arms that cannot be re-run.** The auxiliary policy-selector arms C2 and C5 cannot be re-run, because their models are not distributed. All other CLUSTOPT arms (C0, C1, C3, C4) run; the baseline arms need the upstream code.
