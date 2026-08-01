# M0 — render strategy benchmark

Implementation_plan.md §11 blocks every later milestone on this measurement. It exists
to replace §4's *estimates* with numbers from the machine that will actually render.

## What it measures

| Configuration | What it represents |
|---|---|
| `v2_baseline_cycles_fullscene` | what the old plan did — full frame, ground plane, world, Cycles 2 samples |
| `v3_cycles_<n>s_fullframe` | actor-only, transparent background, full frame |
| `v3_cycles_<n>s_roi` | actor-only, cropped to the actors' union bounding box |
| `v3_eevee_fullframe` | EEVEE Next, actor-only, full frame |
| `v3_eevee_roi` | EEVEE Next, actor-only, ROI cropped — the intended v3 default |

For each it records the **first frame** separately from the **steady-state median**. The
difference is shader/kernel compilation, which is the cost §6.2's persistent process
removes.

It also reports whether EEVEE Next can render headless at all — the open risk in §5.6,
since Colab needs a working EGL context.

## Go/no-go

The best v3 configuration must be **≥ 5× faster** than the v2 baseline. Below that we stop
and re-plan rather than build on a bad assumption. The driver exits non-zero on `no_go`.

## Running it

Locally (validates the harness; CPU numbers are not the real answer):

```bash
python backend/tools/run_m0_benchmark.py --quick --label local_smoke
```

Full local sweep:

```bash
python backend/tools/run_m0_benchmark.py --label local
```

On Colab — **this is the measurement that matters** — after the notebook has installed
Blender 4.5 and cloned the repo:

```bash
python backend/tools/run_m0_benchmark.py --label colab_t4
```

Commit the resulting `backend/benchmarks/m0_colab_t4.json`. It is the input to the M1 render
profile and to the §4 budget revision.

## Results so far

| Label | Machine | Baseline | Best v3 | Speedup | Verdict |
|---|---|---:|---:|---:|---|
| `local_smoke` | Windows, **CPU-only Cycles**, quick mode | 8.28 s/f | 0.38 s/f (`eevee_roi`) | ~20× | GO |
| `colab_t4` | *pending — run this* | | | | |

The local run is CPU-only and 2 frames per configuration, so treat it as harness validation
plus a directional signal, not the production number. It did already confirm two things:

- **ROI + actor-only is worth roughly 5.5×** on Cycles alone (8.28 → 1.49 s/frame).
- **EEVEE's first frame cost 3.37 s against a 0.38 s steady state** — an 8.9× penalty that
  is paid once per process. That is the §6.1 overhead argument, measured.
