# LP1-Conditioned Vector Diffusion

This directory trains a conditional DDPM for the two continuous features in the LP1 CSV files:

```text
x = [x_top10mean, y_rank450mean_ascending]
condition = LP1 count
```

It is a vector model, not a spatial lattice model. The denoiser is an MLP and expects inputs of shape `(batch, 2)`.

## Conditional Prior

The LP1 control is normalized from the supplied endpoint values:

```text
LP1 0  -> -1
LP1 16 -> -0.5
LP1 32 ->  0
LP1 48 ->  0.5
LP1 64 ->  1
```

The training data fit a linear prior mean in standardized feature coordinates:

```text
m(c) = intercept + slope * c
p(x_T | c) = Normal(m(c), I)
```

## Train Five LP1 Conditions

```bash
conda run -n ph_diffusion python ExpTM_GC/ph_vector_diffusion/train.py \
  --condition 0 ExpTM_GC/ph_vector_diffusion/data/tip3_o_degree_top10_vs_rank450_301_400_LP1_0_first100frames_for_collaborators.csv \
  --condition 16 ExpTM_GC/ph_vector_diffusion/data/tip3_o_degree_top10_vs_rank450_301_400_LP1_16_first100frames_for_collaborators.csv \
  --condition 32 ExpTM_GC/ph_vector_diffusion/data/tip3_o_degree_top10_vs_rank450_301_400_LP1_32_first100frames_for_collaborators.csv \
  --condition 48 ExpTM_GC/ph_vector_diffusion/data/tip3_o_degree_top10_vs_rank450_301_400_LP1_48_first100frames_for_collaborators.csv \
  --condition 64 ExpTM_GC/ph_vector_diffusion/data/tip3_o_degree_top10_vs_rank450_301_400_LP1_64_first100frames_for_collaborators.csv \
  --output-dir ExpTM_GC/ph_vector_diffusion/runs/lp1_five_conditions \
  --epochs 100
```

## Sample And Plot

```bash
conda run -n ph_diffusion python ExpTM_GC/ph_vector_diffusion/sample.py \
  --checkpoint ExpTM_GC/ph_vector_diffusion/runs/lp1_five_conditions/model.pt \
  --output-csv ExpTM_GC/ph_vector_diffusion/runs/lp1_five_conditions/generated.csv \
  --control 0 16 32 48 64 \
  --num-samples-per-control 1000

conda run -n ph_diffusion python ExpTM_GC/ph_vector_diffusion/plot_scatter_svg.py \
  --input-csv ExpTM_GC/ph_vector_diffusion/runs/lp1_five_conditions/generated.csv \
  --output-svg ExpTM_GC/ph_vector_diffusion/runs/lp1_five_conditions/generated_scatter.svg
```

To plot raw training data, pass each source using `--condition LP1 CSV` instead of `--input-csv`.

## Alternative pH Parameterization

The same scripts also support a pH condition by passing `--control-name pH` and using pH values with `--condition`. The checkpoint then records the condition as `pH`, generated CSVs use a `pH` column, and the plotting script should receive `--control-name pH`.
