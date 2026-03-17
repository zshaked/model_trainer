# Beyond Transformers — Autonomous Research Program

## Mission

Design, implement, and validate a neural network architecture that challenges
transformer assumptions using dynamical, biologically-inspired computational
primitives: pole-parameterized dynamics, predictive coding, adaptive thresholds,
RL-guided routing, and Hebbian consolidation.

## Rules for the Agent

1. **Only modify `train.py`**. Never touch `prepare.py` or `loop.py`.
2. **One change per experiment**. Make a single, testable modification.
3. **The metric is `val_bpb`** (validation bits per byte). Lower is better.
4. **Each experiment runs for TIME_BUDGET seconds** (set in prepare.py). Do not change this.
5. **Do NOT ask "should I keep going?"** The human may be asleep. Run until interrupted.
6. **If you run out of ideas**, re-read train.py, review results.tsv, and think harder.

## Workflow

```
while True:
    1. Read train.py, results.tsv, git log
    2. Formulate a hypothesis
    3. Edit train.py (one change)
    4. Run: python loop.py --once --tag "<description>"
    5. Check output: was it kept or reverted?
    6. Record observations and move to next hypothesis
```

## Current Phase: Phase 1 — Pole-Parameterized Units

### What Exists
- `PoleUnit`: Recurrent unit with learnable complex poles (σ + iω)
- σ constrained < 0 via `-softplus(raw_sigma)` for stability
- Complex recurrence: `h[t] = exp(σ + iω) * h[t-1] + W_in * x[t]`
- 4-layer model with feedforward and RMS norm

### Research Questions (Phase 1)
- Does the pole constraint (σ < 0) hold through training?
- Do different tasks induce different pole distributions?
- What is the right discretization scheme?
- Can we improve throughput while maintaining pole dynamics?

### Ideas to Try (Phase 1)
- [ ] Adjust pole initialization spread
- [ ] Try different discretization (ZOH, bilinear, Euler)
- [ ] Vary number of layers (2, 4, 8)
- [ ] Vary hidden dimension (128, 256, 512)
- [ ] Add frequency-band attention between units
- [ ] Try learned dt (time step) per layer
- [ ] Experiment with different nonlinearities (SiLU, Swish)
- [ ] Multi-head pole structure (groups of poles at different scales)

### Future Phases (do not implement yet)
- **Phase 2**: Adaptive threshold / hyperpolarization
- **Phase 3**: Predictive coding layers
- **Phase 4**: RL routing policy
- **Phase 5**: Hebbian consolidation
- **Phase 6**: Full integration and benchmark

## Signals That Phase 1 Is Working
- Poles form interpretable clusters (fast poles early, slow poles late)
- Training is stable without gradient clipping or normalization hacks
- Performance is competitive with a simple RNN/transformer baseline
- Pole statistics show learning (not stuck at initialization)
