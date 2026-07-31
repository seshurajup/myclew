# #2 solution 🤖 Quantum Uncertainty 🤖

We want to thank Kaggle and CHAMPs for organizing such an awesome competition: 

- No leakage.
- Same distribution in train, private and public test (very stable CV vs. LB, public LB vs. private LB). This is very relevant in other competitions given so many participants when the 3 distributions are different some winners (not all, e.g. [CPMP approach to manually align test ~ train in the Microsoft malware comp](https://www.kaggle.com/c/microsoft-malware-prediction/discussion/84069#latest-499864)) are just lucky they fit private test distribution by chance; not the case here. 
- Inspiring and useful science problem hopefully used for good purposes as described in the context (new drugs, etc.).

Neither my teammate @pavelgonchar nor I had any previous domain expertise and we made the decision early on that we would tackle this problem using a very pure deep learning way: letting the model build the features for us, not the other way around (b/c obviously we were at a disadvantage if we tried to become quantum experts in 1 month… hence our team name 🤖Quantum Uncertainty🤖 we didn't know if our yet-to-be-developed approach was going to work).

Our solution had two major parts: 1) the input representation and 2) deep learning architecture.

**Input representation**

This is in our opinion the key part: we take a molecule and a source atom and move it so the source atom is @ (0,0,0). For each molecule we create N molecule siblings (N being as many source atoms are defined for that molecule), and each molecule sibling is translated so its source is at (0,0,0). 

The `x` (input) are three arrays of dimension 29 (maximum number of atoms): 

1) `x,y,z` position of each atom, 
2) `atom type` index (C=0, H=1, etc…)
3) `j-coupling type` index (1JHC=0,'2JHH=1,etc.)

Padding is done by placing -1 in `atom type` index and `j-coupling type` for molecules which have less than 29 atoms.

The `y` (ground truth) is just an array of dimension 29 containing j-couplings located at target atom indices.

Note that there is no graph information nor any other manually engineered features.

**Data augmentation**

We did two types of data augmentation:
- Rotations: which worked and were useful in our first attempt model: pointnet-based, but proved worthless in the final models (atomic transformer).
- J-coupling symmetry: as described [in this discussion](https://www.kaggle.com/c/champs-scalar-coupling/discussion/94706#latest-563148)

**First attempt: Pointnet-inspired architecture (got up to -2.28200 LB)**

Our input representation is basically a point cloud: an unordered set of elements with absolute positions `x,y,z` and two attributes `atom type` and `j-coupling type`. We modified the [Pointnet](https://arxiv.org/abs/1612.00593) architecture to regress j-couplings. Training was a bit unstable and we tried many variations of the architecture, swapping FC layers by linear (fixed) projections (Hadamard), adding coulomb matrix as input, etc. 

While this worked OK and got us to -2.28200 LB (ensembling a few models) we felt that this architecture was limited by the extreme pooling/bottleneck operation so we decided to explore other architectures: meet the Atomic Transformer.

**Final architecture: meet the Atomic Transformer**

You may know that the recent NLP revolution is mostly due to the transformer architecture described in the [Attention is all you need paper](https://arxiv.org/abs/1706.03762). The vanilla transformer architecture uses a very clever technique to add positional encodings that are needed for position-dependent input, such as language. 

Our input representation is a set, which means we can (and should) remove positional encoding. Prior to this competition we had no experience with transformers either but there's a section in [Lex Fridman MIT podcast interviewing Orion Vinyals](https://www.youtube.com/watch?v=Kedt2or9xlo) where he mentions the inherent position invariance of a barebone transformer encoder layer. This immediately triggered the idea of using transformer layers (encoders) stacked taking as an input `x,y,z` (normalized but otherwise as-is), and `atom type` and `j-coupling type` embeddings; just concatenated… nothing fancy. 

The dimension of the embeddings was such that the total dimension of the input vectors was `d_model` (as normally reference in transformer literature). We started with 256 and got immediately great results surpassing our pointnet-inspired architecture so we followed this path.

We trained a total of 14 models, with varying dimensions from 512 to 2048 and layers from 6 to 24. Each model parameter size ranged from ~12M to ~100M (biggest model).

We trained some models from scratch, others we fine-tuned. We also fine-tuned a few models on the troublesome j-couplings: reaching -2.12 CV on 1JHC on and -2.19 CV on 1JHN.

Our best score is an ensemble of 14 models achieving private LB of -3.22349, and our best single model achieved private LB of -3.16234, again just with `x,y,z`, `atom type` and `j-coupling type` inputs (no QM9, etc.).

**What didn't work**

Many things! We tried:

- Multi-task learning using contributions and other organization provided values.
- Dropout: We tried multiple attempts to add dropout at various stages (embeddings, encoder layers, pre-decoder, etc.). None of them worked.
- Knock-out: We added a variation in which as input we deleted 10% of the input atoms, the idea being that the model would build an internal representation of the missing atoms. Surprisingly this worked in that the model still converged nicely but failed to reduce train ~ val gap.
- Rotations and TTA in Atomic Transformer: it didn't reduce train ~ val gap and didn't produce meaningful TTA gains.
- Deep decoder: Our decoder is just a projection of ~ the model dimension to 1 (scalar coupling). We tried adding more expressive power to the decoder but this didn't help.
- Fp16 training. This worked for models of dimensions 256 but as training evolved gave `NaN`s despite numerous attempts to fix it.

**Source code**

We will make source code available once we do clean up. It's a single jupyter notebook using FastAI. Be patient.

**Computational resources**

We had more ideas than computational resources, even if our computational resources were not tiny:
- 3 x 2080 Ti + 128 Gb RAM + 16c32t processor
- 2 x 1080 Ti + 64 Gb RAM + 8c16t processor
- Rented 8+ 2080 Ti + 64 Gb RAM + 16c32t processor (multiple machines rented as needed)

**Final thoughts**

This was our most fun and hardest competition so far:
- Challenging problem
- Most teams in top 5 had domain experts (although we went *domainless* as part of our strategy)
- Hungry computational resources.

Even if we lost #1 position just a few hours before competition end we feel very excited we were able to achieve such [useful results for the organizers] (https://www.kaggle.com/c/champs-scalar-coupling/discussion/98375#latest-569312). In retrospect we believe a single model of the Atomic Transformer may achieve ever better results with further training.

Best - Pavel &amp; Andres

p.s. No graph NNs. We though graphs as manually engineered features that the model can infer by itself.