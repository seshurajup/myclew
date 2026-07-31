# 1st place: genetic algorithm and GPU relaxation

Link to notebook demonstrating the solution - reading through this notebook first may be helpful in understanding this writeup: https://www.kaggle.com/code/jeroencottaar/santa-2025-1st-place-solution

Separate notebook that visualizes all 200 solutions: https://www.kaggle.com/code/jeroencottaar/santa-2025-1st-place-first-share

Github: https://github.com/jcottaar/packing

First off, I'd like to thank the organizers for a very fun and accessible competition. It's really cool to see such a diversity of solutions for a classic problem. 

The key aspects of my solution:
- The core of the solution is a genetic algorithm, with hierarchical layers of interaction between individuals. There are several mechanisms in place to allow exchange of genetic material while preventing a single solution from dominating the population.
- This is combined with a heavily optimized GPU-based relaxation scheme to find local minima. This allows far more moves to be accepted.
- Finally, symmetry plays a key role in reducing degrees of freedom for large numbers of trees. Specifically, I constrained solutions with even number of trees to have 180° symmetry and used tesselated seeds (optimizing only the edge).

I used AI assistance to explore literature and brainstorm ideas (typical chat: https://chatgpt.com/share/6988df69-de94-8004-ac56-54f983822f01 ), and to write parts of the code. This writeup itself is only lightly edited with AI. I used a variety of models through online APIs and Github Copilot, with a focus on ChatGPT for discussion and various Claude models for code.

In terms of compute, I spent about $250 on vast.ai, which paid for 1000 hours of RTX 5090 time. About half was to find the actual solution, and the other half for testing and development. This cost estimate does not include the use of AI tools discussed above.

Let's dive into the actual solution!

## Overview

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F14984949%2F1938d2dd7b8dc1b106fe4713a1f8aaa1%2FPasted%20image%2020260208121706.png?generation=1770550328192445&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F14984949%2Fdc2b5d05cff88d0481e0bd053e99b52a%2FPasted%20image%2020260208122812.png?generation=1770550351621631&alt=media)

The overall solution structure is outlined in the figures above. These describe how we find a single solution for a given number of trees. There was some overarching management involved as well, such as running various seeds and various solution types for a given number of trees; this was mostly manual work.

## Initial solution

The initial solutions per island serve as starting point for the genetic algorithm; as we'll see later, we also apply resets once in a while, which involves generating a new initial solution. There are several variations possible; before we dive in, here's a visual overview:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F14984949%2Fa84f6171b0e5b059ed5715d5e6358267%2FPasted%20image%2020260208103732.png?generation=1770550431523563&alt=media)

The most basic solution just has the trees randomly scattered and rotated within a square of some initial size (chosen to be somewhat bigger than the expected optimal size). This works up to N around 40, but for higher N we need to introduce more constraints to reduce the degrees of freedom.

For even numbers of trees, I noticed the algorithm tended to converge to rotationally symmetric solutions (with 180° rotation); I'm not sure if they're truly optimal or if this is related to the nature of the algorithm. This showed a clear opportunity: position only half the trees randomly, and determine the other half by rotating the initial ones by 180°. This symmetry is then also enforced throughout the genetic algorithm (by explicitly separating the phenotype - the actual solution - from the genotype - which has only half of the trees). I also played around with 90° symmetry; it actually finds some really nice solutions, but none of them were optimal.

With this symmetry, we can go up to N around 70 (and only for even numbers of trees!). I then noticed that many solutions were tesselated, i.e. the bulk of the solution follows a crystal structure. So the next opportunity is to use this in our initial solution. In this case we seed most of the trees from a predetermined crystal, and scatter the remaining trees randomly around the edge. When in this mode, we will not touch the inner trees during the genetic-algorithm moves. However, we do still move these trees during the GPU relaxation step.

Tesselated solutions allow us to go all the way to N=200 reasonably efficiently. I mostly stuck to continuing to enforce 180° symmetry for even numbers of trees.

One important aspect is to choose the correct crystal, as well as its position and orientation. I'll discuss only the crystal itself here. I spent quite some time hunting for good crystals, including using the full GA algorithm with periodic boundary conditions. In the end I only found 2-tree solutions, but they're still quite diverse. These are the ones I found (there's many more, but they can be reached smoothly from these archetypes):

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F14984949%2F46c1abfe9d14a2cfebc4c72bca5166b4%2FPasted%20image%2020260207124303.png?generation=1770550418565268&alt=media)

I tried using various crystals, but in the end it's really hard to beat 'Perfect dimer':
- It's the tightest packing.
- It has a 'straight edge' (the y=0 line in the figure), meaning it can line up to the top and bottom of the square nicely, and we only have to deal with filling in the left and right edges.
- It's quite deformable, i.e. we can change the angle and aspect ratio of the unit cell without making the packing too much worse. This means that the crystal can 'squeeze' around our edge solution.

## Offspring generation

The first step in each GA iteration is to enlarge the population to 3750 individuals by generating offspring. There are two main categories:
- **Mutate**: apply some modification to a single individual
- **Crossover**: combine two individuals

The following mutation moves are used (always only one):
- **Move**: move a random tree to a random position and rotation.
- **Jiggle**: apply an offset to the position and rotation of one or more trees. There are four variations of this move, which vary in the number of trees they move and how far the moves are.
- **Twist**: rotate trees close to a certain point by a randomly selected angle. The further we are from the point, the less the rotation is applied (to preserve some degree of continuity).
- **Translate**: add a random offset to all trees, wrapping trees around the boundaries if they move outside.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F14984949%2F46702178136cd3bc5f91ee19575ae346%2FPasted%20image%2020260208113858.png?generation=1770550454725338&alt=media)

Crossover moves will remove a section of trees from one solution and insert a similarly shaped section from a mate solution. Most important here is the mate selection. There are two sources:
- Any individual on the same island is a suitable mate.
- Lower-scoring champions from nearby islands on the ring are suitable mates. We only allow lower-scoring mates to give young islands time to develop; otherwise they'd quickly be overwritten by more mature solutions. If an island hasn't improved its score in a while this rule is dropped, and it's allowed to mate with any champions from nearby islands.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F14984949%2Fbe50e8873b3b67c44e29f67251f47ccd%2FScreenshot%202026-02-08%20124412.png?generation=1770551206041190&alt=media)

The exact subset of moves used and how they are configured varies per solution type (tesselated and/or symmetric). For tesselated solutions, moves are only applied to the edges.

## GPU relaxation

Out of the 3750 individuals generated per island, we pick the best 1500 (as in those with the least overlap). Most solutions are still far from feasible (i.e. trees overlap badly). So we search for a local minimum of the following cost function; note that a 0 value means we have a feasible solution:

$$\text{cost} = \alpha\sum_{i=1}^N\sum_{j=1}^N \text{overlap}_{ij}^2$$

$$+ \beta\sum_{i=1}^N\text{outside}_i^2$$

Here, "outside" refers to how far a tree pokes outside the square boundary (0 if it's fully inside), and "overlap" refers to the overlap between a pair of trees (0 if not overlapping). Several overlap functions are implemented, but the most important one is exact separation distance (i.e. the smallest distance that one tree needs to be moved to resolve the overlap). This is too expensive to compute in-line, so I precompute it (using Minkowski geometry) and store the results in a lookup table. This lookup table is 3D (relative X,Y,angle), and is used with trilinear interpolation.

The cost function is minimized using LBFGS. This means we need to compute the cost function and its gradients to all tree positions; this is implemented directly in CUDA. My LBFGS implementation is adapted from PyTorch, modified to work on batches (i.e. to solve multiple sets of trees simultaneously).

Here's what this relaxation process looks like visually:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F14984949%2F5726130d3207bfb967b5da5bd20189a3%2Frelaxation_animation.gif?generation=1770550587055567&alt=media)

The entire sequence in the animation above can be done 30,000 times per second (for 100 trees) on an RTX 5090.

## Selection

Next, we need to apply a selection step to keep our best individuals. This step is applied per island individually; recall that each island has 1500 individuals at this point. We will reduce this to 111.

The first 27 survivors are simply the 27 with the least overlap (according to the cost function above). It's possible that some of them are identical.

The remaining 84 survivors are chosen based on a mix of overlap and genetic diversity (defined below). Specifically, we pick the least-overlap individuals that has a genetic diversity over a given threshold to the already selected individuals. In other words, we reject any individuals that are too close to the ones already selected.

So how do we define genetic diversity between solutions A and B? It's the smallest set of transformations (i.e. translation and rotation per tree) we can apply to A to make it identical to B (or a mirror/90 degree rotation of B). This can be computed using the Hungarian algorithm, though in this step we use an approximation for speed reasons. Both the Hungarian algorithm and this approximation are implemented in CUDA.

## Management

Finally, I apply some administrative steps.

An individual island can be reset, i.e. reinitialized from a new random starting solution, in two cases:
- If its score has been stuck for a while (typically ~100 generations), and it's not currently the best island.
- If its champion (best individual) is identical to that of another island. This is computed with the same genetic diversity as described above (this time using the full Hungarian algorithm).

For any island of which the overlap falls under a certain threshold, we consider it as having found a valid solution for its current square boundary size. We reduce the square boundary size for that island, for all individuals. So different islands can be working with different square sizes, but the same square size always applies within an island.

Finally, if the entire ensemble has not improved its solution in several generations, we stop the algorithm (this is the only stopping criterion). At this point the solution typically has some overlap, so we apply a 'legalize' step. This is similar to the GPU relaxation above, but adapted to end up at a true zero-overlap solution.

## What didn't work

- Calculating approximate separation distance in-line rather than using a lookup table - I tried (also available in CUDA):
	- An approximation of separation distance based on a convex breakdown
	- Overlapping area
- Other island connectivity patterns, such as star, tree or hypercube
- Tournament selection instead of the diversity-based selection
- Applying an initial 'jiggling' step to the solutions
- Squeeze the trees inwards rather than occasionally reducing the square size
	- Explanation: The current solution fixes the square size, searches for a low-overlap solution, and then reduces the square size. An alternative is to have the square size as a solution parameter (next to the tree positions), and include it in the cost function. This adds an inward squeezing force. This didn't work for the GA, but this concept is still used during the 'legalize' step.

## What I didn't get around to trying

- Properly study best known methods (such as sparrow) and integrate their approaches - probably a big one, since people were able to significantly improve on my solution almost immediately after I posted the CSV
- Statistically study the effectiveness of the various moves and their hyperparameters
- Annealing in the relaxation step: apply some random moves during the relaxation, especially near areas affected by the GA moves
- Build a GUI to try manual edits, integrated with the GPU relaxation
- Personalities: variation between islands or individuals, such as different move selections
- Crossover between solutions for different numbers of trees
- Less random starting layouts
- Add combination moves (a few moves in sequence)
- Find more 'manual' solutions, based on one or more stacked crystals with no GA step (like N=156)