# IndicConformer Hindi adaptation on RTX 3050 6GB — staged plan

## What actually caused the slowdown

Your full fine-tune wasn't just "too many parameters." Two things compound on a
Hybrid RNNT-CTC model specifically:

1. **Optimizer state for 600M params.** Adam keeps ~2 extra tensors per parameter
   (momentum + variance), so full fine-tuning needs several GB just for optimizer
   state, before activations. That's most of your 6GB gone before a batch even runs.
2. **The RNNT joint tensor.** RNNT loss materializes a `[batch, T, U, vocab]` tensor
   — that's not a fixed cost, it scales with audio length × transcript length ×
   vocab size, and it's usually the actual memory/time bottleneck in transducer
   training, not the encoder itself. NeMo has a documented, built-in control for
   this: `joint.fused_batch_size`, which computes the loss in smaller sub-chunks
   instead of one giant tensor. This is a real, existing config field (not
   something invented for this) — trades a bit of speed for a lot of memory
   headroom, recommended ratio to `batch_size` is 1:8–1:16 for constrained GPUs.

Neither of those is primarily about "the encoder is too big to backprop through."
So the fix isn't only LoRA-style parameter reduction — it's **freeze the
optimizer-state cost (encoder) AND bound the joint tensor (fused_batch_size) AND
stop padding long/short clips together (bucketing + duration cap)**.

## Chosen approach

**Freeze the Conformer encoder's original weights. Attach small LinearAdapters
inside it (if this NeMo build's encoder class is adapter-registered — verified by
`inspect_model.py`, not assumed). Fully unfreeze the small RNNT decoder+joint+CTC
head.** This is a real, documented NeMo mechanism (`model.add_adapter`,
`LinearAdapterConfig`, `ASR_with_Adapters` tutorial), not a NeMo-NLP PEFT feature
grafted on where it doesn't belong — it was built specifically for Conformer
CTC/RNNT/Hybrid models.

Because it's a Hybrid model, the CTC head and RNNT head **share the same
encoder**. Adapting the shared encoder improves both decoding paths even though
CTC-head gradients are far cheaper to compute than RNNT-joint gradients — that's
why `--ctc_loss_weight` defaults to 0.3 rather than 0: some CTC-weighted signal
plus the RNNT signal is cheaper than RNNT alone, and both push the same shared
adapters. I looked for a clean "skip RNNT loss entirely, train CTC-only" flag in
the Hybrid model's `training_step` — it always computes both losses and combines
them via `aux_ctc.ctc_loss_weight`; there's no built-in switch to skip the RNNT
loss computation itself. Doing that requires subclassing `training_step`, which
I've deliberately left out of the default script since it depends on internal
method names that can silently drift between NeMo point releases. If Stage 1
still doesn't fit in memory even with `fused_batch_size` at its minimum, that
override is the next lever — ask and I'll write it against whatever
`inspect_model.py` shows for your exact install.

If `inspect_model.py` shows your installed encoder class is **not**
adapter-registered, `finetune.py` auto-falls-back to: freeze encoder completely
(no adapters), train decoder+joint+ctc_decoder only. Still satisfies your
"freeze most of the encoder" and "train prediction/joint if appropriate" options
— just without the encoder-side adaptation.

## Is 5 epochs necessary?

No — for this kind of narrow domain/vocabulary adaptation (not training from
scratch), the literature and NeMo's own adapter tutorial converge in **1–3
epochs** over the adaptation subset; going further risks overfitting a few
thousand samples and forgetting general Hindi ASR capability, which is explicitly
one of your stated goals to avoid. Stage 1 uses 1 epoch as a sanity check, Stage 2
uses 2–3.

## Staged plan

I'm not going to hand you fabricated iterations/sec or wall-clock numbers — I
don't have your GPU in front of me, and your own smoke test already shows this
model's real throughput varies a lot by config. Instead, each stage tells you
what to measure before moving on.

| | Stage 1 — sanity | Stage 2 — realistic | Stage 3 — optional expand |
|---|---|---|---|
| Samples | 500–1000 (`select_subset.py --n_stage1`) | 5k–10k (`--n_stage2`) | up to full 52k, only if Stage 2 WER improves and doesn't regress general Hindi |
| Frozen | encoder backbone weights | encoder backbone weights | same |
| Trainable | adapters (if available) + decoder + joint + ctc_decoder | same | same, possibly larger adapter dim |
| Adapter dim | 16 | 32 | 32–64 |
| batch_size | 4 | 4–6 (raise only if Stage 1 VRAM had headroom) | same as Stage 2 once stable |
| grad_accum | 2 (effective 8) | 4 (effective 16–24) | same |
| fused_batch_size | 4 | 4 (raise to 8 only if Stage 1 had VRAM to spare) | same |
| max_duration | 10s | 12s | 12–15s |
| epochs | 1 | 2–3 | 1–2 (fine-tuning an already-adapted model) |
| What to check before advancing | Did it complete without OOM? What was actual it/s from the PTL progress bar? Did WER (via `eval_wer.py`) improve at all vs. the original model on a 100-sample slice? | Full WER/CER on `test_manifest.json`. Compare against original model's WER on the same set — regression means adapter dim or LR is off, not that the approach is wrong. | Only proceed if Stage 2 is a clear win; diminishing returns are likely given 5k–10k is already a reasonable adaptation set. |

**VRAM**: expect Stage 1 to leave meaningful headroom on 6GB, since Adam state
for a ~600M-param model with only adapters+decoder+joint+ctc_decoder trainable
(check `inspect_model.py` output for the exact trainable-param count on your
checkpoint) is a small fraction of what full fine-tuning needed. If Stage 1 OOMs
anyway, first move is `--fused_batch_size 2`, then `--batch_size 2 --grad_accum 4`
(same effective batch, smaller per-step memory), then `--max_duration 8`.

## Exact commands

```bash
cd /mnt/d/Final_year_project
conda activate indicconformer

# 0. Inspect the actual installed API — do this first, every time you change
#    NeMo/torch versions. Tells you whether adapters are usable and what the
#    real trainable-param split looks like on this exact checkpoint.
python inspect_model.py --nemo_model indicconformer_stt_multi_hybrid_rnnt_600m.nemo

# 1. Build stratified subsets
python select_subset.py \
    --manifest train_manifest.json \
    --out_stage1 subset_stage1.json --n_stage1 800 \
    --out_stage2 subset_stage2.json --n_stage2 8000 \
    --max_duration 15.0

# 2. Stage 1 — sanity run
python finetune.py \
    --nemo_model indicconformer_stt_multi_hybrid_rnnt_600m.nemo \
    --train_manifest subset_stage1.json \
    --val_manifest test_manifest.json \
    --output_dir checkpoints/stage1 \
    --stage 1 --max_epochs 1 --batch_size 4 --grad_accum 2 \
    --fused_batch_size 4 --max_duration 10.0 --adapter_dim 16

# Check WER before vs after on a quick 100-sample slice
python eval_wer.py --nemo_model indicconformer_stt_multi_hybrid_rnnt_600m.nemo \
    --test_manifest test_manifest.json --limit 100
python eval_wer.py --nemo_model checkpoints/stage1/indicconformer_hi_adapted_stage1.nemo \
    --test_manifest test_manifest.json --limit 100

# Qualitative before/after
python compare_inference.py \
    --original indicconformer_stt_multi_hybrid_rnnt_600m.nemo \
    --adapted checkpoints/stage1/indicconformer_hi_adapted_stage1.nemo \
    --test_manifest test_manifest.json --n 10

# 3. Stage 2 — only after Stage 1 looks sane
python finetune.py \
    --nemo_model indicconformer_stt_multi_hybrid_rnnt_600m.nemo \
    --train_manifest subset_stage2.json \
    --val_manifest test_manifest.json \
    --output_dir checkpoints/stage2 \
    --stage 2 --max_epochs 3 --batch_size 4 --grad_accum 4 \
    --fused_batch_size 4 --max_duration 12.0 --adapter_dim 32 --num_buckets 5

# Full WER/CER on the whole test set
python eval_wer.py --nemo_model checkpoints/stage2/indicconformer_hi_adapted_stage2.nemo \
    --test_manifest test_manifest.json

# Resume an interrupted stage
python finetune.py \
    --nemo_model indicconformer_stt_multi_hybrid_rnnt_600m.nemo \
    --train_manifest subset_stage2.json --val_manifest test_manifest.json \
    --output_dir checkpoints/stage2 --stage 2 --max_epochs 3 \
    --resume_from checkpoints/stage2/last.ckpt
```

## Avoiding overwriting the original model

`finetune.py` never writes to `--nemo_model`'s path. It always saves to
`{output_dir}/indicconformer_hi_adapted_stage{N}.nemo`, a new file. The original
`indicconformer_stt_multi_hybrid_rnnt_600m.nemo` is only ever opened with
`restore_from` (read-only). If you want belt-and-suspenders protection, `chmod
444 indicconformer_stt_multi_hybrid_rnnt_600m.nemo` before you start.

## What I verified vs. what to verify yourself

Verified against current NeMo source/docs before writing this: the
`add_adapter`/`LinearAdapterConfig` adapter mechanism for Conformer-based ASR
models, the `joint.fused_batch_size` memory control, `validation_ds.compute_eval_loss`,
and `aux_ctc.ctc_loss_weight`. What I *can't* verify without your machine: whether
your specific 1.23.0rc0 build's encoder class is registered as adapter-compatible
(some intermediate NeMo builds shipped this only for newer model families like
Canary) — that's exactly what `inspect_model.py`'s "Adapter support check" section
tells you, and `finetune.py` reads that same check at runtime and falls back
automatically rather than crashing.
