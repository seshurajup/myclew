# The ARChitects’ Solution

Hey Kaggle & ARC community!

The competition has been an exciting ride, especially in the last few intense weeks.

We’re excited to give you two compact “solution cards” below that summarize the core ideas of our submissions.  
**Or, check out our full technical report here:** [lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)

In the technical report we talk about the full story and entire journey: our approaches, our total compute budget, what else we’ve been working on, what surprised us by working really well, what didn’t quite work (yet!?) and what our next steps would have been if we had another week or two.

ARC was the perfect project to pour a lot of energy and passion into, spending nearly every waking hour on the home stretch for fine-tuning and experimenting.  
So: we’re all looking forward to what comes next!!

The ARChitects — *Daniel, Jan & David* 👋

<div style="display: flex; flex-direction: row; gap: 8px;">
  <img
    src="https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/The%20ARChitects%20-%20Technical%20Report/image%2012.png"
    alt="The ARChitects Logo"
    style="flex: 0 1 auto; height:20em"
  />
  <img
    src="https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/The%20ARChitects%20-%20Technical%20Report/arcfront.jpg"
    alt="Photo of all of us @ ICML '25, Vancouver"
    style="flex: 0 1 auto; height: 20em;"
  />
</div>

# Solution Cards

## Recursive Masked Diffusion Approach - “Final submission”

| Aspect | Summary |
| --- | --- |
| **Base model** | Used a masked-diffusion LLM: LLaDA-8B fine-tuned for ARC-style tasks. |
| **Masking & Loss Strategy** | Standard cross-entropy loss to reconstruct masked positions.  |
| **Soft Masking + Recursive Latent Sampling** | We found that adding the mask tokens to every token let’s the model refine it’s own pedictions.</br>After an initial guess, feed model output back in (with soft-masking), iteratively refine predictions in a loop until convergence or stability. This effectively turns the model into a continuous, recursive solver. |
| **Why Recursion Helps** | The models token embeddings are continuous. We found that you can blend tokens (token algebra) and feed back outputs. Soft-masking seems to tell the model “this position needs improvement.” This enables iterative self-improvement, rather than one-shot generation. |
| **Adaptations for ARC** | Replaced original 1D positional encoding with a 2D positional encoding (based on “Golden Gate RoPE”) to better match the 2D grid nature of ARC tasks. |
| **Datasets & Training** | Same data pipeline as AR model from last year: used datasets like ReARC, ARC-GEN-100K, official ARC1 & ARC2, ARC-Heavy, ConceptARC. Pretraining: ~175k steps (batch size 8) on 8×H100 GPUs. Test-time fine-tuning: for each task, 128 steps on L4 GPU (batch size 1). |
| **Shape Prediction** | Because output grids may vary in size, we trained a second LLaDA model (finetuned) to predict the correct output “shape” (grid size) given input examples, i.e. where delimiters go. |
| **Final Submission Performance** | With known shape: ~30.5% ±1% on held-out eval set (after 102 inference steps, that is two rounds of 51 with cold restart). Shape predictor ≈ 85% ±2% correct on eval set. </br> Combined we expected ~26%, but real public score was 21.67%, private score was 16.53%. </br></br>**Our best submission (using a different hyperparameter selection mechanism) achieved a public score of 19.17% and a private score of 19.17%.** </br> We didn’t choose that submission due to the lower public leaderboard score. ;D  |

---

## Autoregressive (AR) Approach - “First half of competition”

| Aspect | Summary |  |
| --- | --- | --- |
| **Summary: Improvements vs prior year** | Task-specific test-time fine-tuning rather than multi-task fine-tuning, speculative decoding + prefix caching,  lowering probability cutoff (from 17% → 7%) to explore more candidates, increased scoring augmentations (8 → 32). |  |
| **Base Model** | Used an autoregressive model (from previous year) built on Mistral‑NeMo‑Minitron‑8B‑Base. </br> We had a better pre-trained model from a late submission last year that we didn’t publish, which included Arc-Heavy in its training data. |  |
| **Sampling Method** | Faster depth-first search (DFS) using speculative decoding (predicting 16 to 32 tokens ahead) to generate even more candidate solutions to select from. |  |
| **Scoring / Selection** | Still “Product-of-Experts” scoring: apply multiple augmentations and aggregate neg-log-likelihoods across them to choose the most probable candidate. |  |
| **Performance / Result** | Maxed-out performance at 16.94% on public ARC2 leaderboard (as of Aug 11). After that, we transitioned to the masked diffusion model. |  |
| **Limitations Observed** | AR model struggled with puzzle-type tasks or tasks requiring global restructuring (e.g., diagonal lines, simulation tasks, etc). Augmentations lacked expressiveness to fully capture global constraints. |  |

---

## Key Insights & Tradeoffs

- The **AR approach** that we used in last year’s ARC 2024 and for the first, larger part of the competition is fast, well-understood, and effective for many tasks, especially when coupled with aggressive sampling + scoring. But we found it struggles with tasks that require **global reasoning** or structural transformations, because it generates sequentially and cannot revisit prior decisions.
- The **masked diffusion + recursion** approach that we used for our final submission is more expressive and flexible: because the model can “rewrite” the entire output grid (via the novel soft-mask reconstruction and recursive refinement technique), it better handles puzzles requiring restructuring. However, it comes at increased computational cost (many inference steps due to refinement recursion), and the final score is sensitive to the shape predictor that we haven’t included into the main model due to time-reasons.
- We found the “token algebra & soft-masking” trick is especially powerful to leverage the continuous embedding space of modern LLMs to enable iterative improvement.

## **Read our full technical report here:** [lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)