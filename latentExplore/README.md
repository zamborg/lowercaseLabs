# Latent Exploration project

This is a space for the latent-exploration project focusing on toxicity.

**Hypothesis:** Nonlinear decision boundary for latent space representaitons of toxicity illustrates potential failure modes of language-model toxicity detectors.

# TODO:

- [ ] Get realtoxicityprompts loaded (and confirm other datasets we'd like)
- [ ] Chose a huggingface (or other) detector we'd like to evaluate (persumably toxicity) and load this detector
- [ ] Setup initial experiment:
  - [ ] Score each of the toxicity prompts with the detector (or I suppose it makes more sense to score a sample generation from the prompt -- which can be pulled from some prior analysis)
  - [ ] Identify "toxic" and "non-toxic" samples
  - [ ] Embed the samples using the toxicity detector and define pairwise sets of "toxic" and "non-toxic" samples
  - [ ] Interpolate between the samples (in the embedding space) and compute the toxicity score of the interpolated samples (write a function to do this)
  - [ ] Visualize the interpolated samples (in the embedding space) and their toxicity scores
    - [ ] This could be a 2D plot with the interpolated samples (in pca) as points and the toxicity score as the color of the points
    - [ ] This could be a 3D plot with the interpolated samples as 2d points and height as the score

# Notes:

N/A atm (need to talk to Prof. Beidi about the validity of this method.)
