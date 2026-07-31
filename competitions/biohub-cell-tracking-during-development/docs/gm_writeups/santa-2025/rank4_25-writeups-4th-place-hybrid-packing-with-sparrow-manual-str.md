# Introduction

First of all, we would like to extend our sincere gratitude to the organizers for hosting this contest. Over the long two-month contest period, the problem was deep and highly engaging. We are deeply honored that this work resulted in our first gold medal.

This write-up is organized as follows.

- First, we describe the algorithms we employed to obtain our final solutions.
- Next, we present representative layouts of our best solutions for various values of N.

One note of caution: because we repeatedly refined the current best solutions using multiple algorithms, we cannot reliably determine which specific procedure led to the final solution for each N.

# Main Algorithms

Our solution pipeline mainly consists of two steps.

- STEP1: Construct well-structured solutions via Sparrow + manual work + compositional exploration
- STEP2: Iteratively improve solutions by replacing partial regions

Within each step, we also used a fairly standard Simulated Annealing procedure to refine solutions.

## STEP1: Construct Well-Structured Solutions via Sparrow + Manual Work + Compositional Exploration

Sparrow is a powerful open-source tool (implemented in Rust) for 2D irregular strip packing.  
We used Sparrow with minimal customization (aside from providing an initial solution). Since Sparrow targets general strip packing, the following restrictions were helpful in promoting better structure for this instance.

- Restrict angles to 8 orientations: 22.5, 67.5, 112.5, ..., 337.5 (configurable via `allowed_orientation` in Sparrow)
- For items in the central region where we want regularity, restrict angles to only 22.5 and 202.5
- Load an existing best solution, slightly shift left/right edge items outward (about 0.2), and rerun

Example output from Sparrow  
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F99b6ad947c235fb9f54da8f49d2e0796%2Fimage-10.png?generation=1770031521647770&alt=media" width="400" />

These settings reflect periodic placements (constructed from angles a° and 180+a°), as well as angles aligned with walls (about 23.5°) and around tree tips (about 40°-50°, depending on the definition). In practice, they help the search efficiently discover well-structured layouts.

Sparrow performed well for small N, but its output became less sufficient once N exceeded roughly 30. In parallel with running Sparrow, we explored dense regular structures; together, these efforts led us to a three-row periodic arrangement, which we further enhanced by adding items at its corners. We then manually refined solutions using a custom visualizer. To streamline this process, the visualizer included features such as data import, copy-and-paste, range selection, rotation, and reflection.

### Three-row periodic placement

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F37afebedc1bfa12d2b783d368973b9a2%2Fimage-3.png?generation=1770031800933621&alt=media" width="400" />

### Adding corners

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2Fa2961e86d93fc3e6b0873801f03075b7%2Fimage.png?generation=1770031822481137&alt=media" width="800" />
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F9c7e33f8969bbcf3a3005e4259aca114%2Fimage-1.png?generation=1770031834729891&alt=media" width="800" />

(and more patterns...)

### Visualizer

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F1a50c14128d13138f760f357053a663f%2Fimage-2.png?generation=1770031853250187&alt=media" width="400" />

We also observed that, for some N, combining two or more regular structures can be particularly effective. Specifically, we implemented this via the following process.  
As a premise, we maintain a database of placements indexed by (width, height) under the following policy.

- Store width and height in buckets of size 0.01
- When registering a new placement, within the same (width, height) bucket, keep the placement with the larger number of trees

We repeatedly update this database through various operations.

- Randomly combine 2-3 trees, then create placements by arranging them vertically and/or horizontally
- Choose two placements and compose them vertically/horizontally to create a new placement (also creating variants such as reflection, rotation, small shifts, etc.)
- Choose one placement and run a local search (hill-climbing-like) that slightly perturbs trees to reduce the overall size

As a result, solutions derived from this method often look like combinations of several regular patterns.  
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2Fd630f6ac4a87ffc8f414d120e2be15b5%2Fimage-11.png?generation=1770031895695471&alt=media" width="400" />

## STEP2: Iterative Improvement by Replacing Partial Regions

In the final stage of the contest, we adopted a framework that takes the current best solution as input and seeks further improvement. This framework grew out of our earlier, more hands-on manual refinements; as a pragmatic compromise to make that process repeatable at scale, we implemented a simplified, automated variant.  
The procedure follows this flow.

1. From a best solution at some N', specify a triangular or rectangular region near a square corner, and select the k items contained in it
2. Paste those k items onto a corner of the best solution for N, and remove k-1 or k or k+1 items among existing ones in descending order of overlap area
3. Resolve the overlaps and apply simulated annealing

This copy-and-paste operation from another N rarely yields an effective replacement in a single try. In practice, we repeat steps (1) and (2) many times (e.g. around 2000 times), use an overlap-area-based heuristic as the primary screening signal, and proceed to step (3) only with the most promising candidate.

## Simulated Annealing

The simulated annealing procedure used throughout is fairly standard: each move slightly perturbs the position of a single tree, with no special transitions. As such, it is not effective as a standalone solver; however, it is useful for refining local optima around candidate structures.
Our simulated annealing implementation was largely standard, with only minor (and still fairly conventional) tweaks; for completeness:

- Decrease temperature exponentially
- Decrease move amplitude exponentially according to the elapsed fraction of time

# Configurations in the Final CSV

For ease of presentation, we group the solutions in our final submission into several broad types based on their original construction. These categories should be viewed as a rough guide, as continuous refinement can make the distinctions less clear. Here we include only visualizations for each type.

## Completely chaotic (mainly small N, e.g. N=17,23)

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F7e735a042373c2023cb9dbc1b36cab30%2Fimage-4.png?generation=1770031920944160&alt=media" width="500" />
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F552efb7a9e59383965284bbef78df43f%2Fimage-5.png?generation=1770031934717748&alt=media" width="500" />

## Three-row periodic base (many N, e.g. N=140,192 and rotated ones N=58,170)

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F17b2baf1d2df118bf5b3c5a2f1d4cecf%2Fimage-6.png?generation=1770031954475435&alt=media" width="500" />
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F45721d1c91fb0df22d1e0db28c61b3ff%2Fimage-7.png?generation=1770031968717320&alt=media" width="500" /><img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F29d7f6a0c255b6620b333d8b2f6ade79%2Fimage-8.png?generation=1770031979134483&alt=media" width="500" />
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F6a1c0f458c0fae1f43fc943a0909a9c7%2Fimage-9.png?generation=1770031991121292&alt=media" width="500" />

## Perfect lattice placement (e.g. N=156)

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2Fd13ed9a33de3bad200ce5f813a0b6844%2Fimage-12.png?generation=1770032005790679&alt=media" width="500" />

## Combination of two or more patterns (e.g. N=59,111,153)

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2F65b8450a20b14298546a710d6af9a1a3%2Fimage-13.png?generation=1770032019178562&alt=media" width="500" />
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2Fdee3d01255619d3df5a09c711593bf09%2Fimage-14.png?generation=1770032031689770&alt=media" width="500" />
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F27595169%2Fd860228d1388194b77a6eb7a02dde58f%2Fimage-15.png?generation=1770032053559214&alt=media" width="500" />