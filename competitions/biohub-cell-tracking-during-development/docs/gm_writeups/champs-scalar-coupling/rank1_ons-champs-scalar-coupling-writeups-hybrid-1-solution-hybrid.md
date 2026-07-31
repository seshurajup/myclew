# #1 Solution - hybrid

Hi everyone,
Here's a brief writeup of the method we used for the #1 entry.  I was hoping to post this a bit sooner, and apologies for the delay.

##Update: 9/13/19
We have posted the code for our method, all of which is available under MIT license at:
https://github.com/boschresearch/BCAI_kaggle_CHAMPS
The main code for the model is available in the `src/` directory, though for those interested, the `models/` directory contains slight variants on this code that were used in the ensemble (mainly earlier versions of the same architecture) so that you can recreate the predictions exactly.

## Introduction
First, a little bit of background on the team.  This project was done at Bosch Research, specifically as collaboration between two groups, one at Bosch Corporate Research, and one at the Bosch Center for AI (BCAI).  Our team consisted of both some ML experts and domain experts.  To introduce our team:

- Jonathan Mailoa and Mordechai Kornbluth are both Research Engineers working out of the Boston lab of Bosch Research.  They are domain experts on DFT and ML approach to molecular simulation, and have worked a great deal on molecular modeling, including lately some work with GNNs.

- Myself (Zico Kolter, I'm a faculty member working in machine learning at CMU, but work in industry one day a week at BCAI in Pittsburgh), Devin Willmott (Research Scienst at BCAI), and Shaojie Bai (my student at CMU, but doing this while while interning at BCAI) were all coming to the competition from the ML side.  I had actually done a bit of (pre-deep-learning, so ancient history) work in ML for molecular modeling, though we didn't end up using many of those methods.

##Overall architecture
Our overall approach is what I would call a kind of "soft" graph transformer.  We wrote the model all from scratch for this work, instead of building upon any existing code.  The model processes an entire molecule at once, simultaneously making a prediction for each of the scalar couplings in the molecule (we hadn't considered the per-atom approach that Quantum Uncertainty used, and frankly it sounds like that may be a pretty competitive approach, given that they did nearly as well with much less physical information).

Unlike a traditional graph model, though, we're really processing the data as more of a "meta-graph".  In constrast to most graph methods for molecules, where atoms are nodes and bonds are edges, in our graph each atom, bond (both chemical bonds, and non-chemical bonds, i.e., just pairs of atoms are included in the model), and even triplets or quads, if desired, all becomes nodes for the graph transformer.  This means that each molecule has on the order of ~500 nodes (depending on whether we include all the bonds or not, or whether we include triplets or quads, which only would be included for chemcial bonds).  At each layer of the network, we maintain an embedding of for each node in the graph, of dimension d ~= 600-750 in most of our models.

Following the standard transformer architectures, at  each layer of the network, we use self-attention layer that mixes the embeddings between the nodes.  The "standard" scaled self-attention layer from the transformer paper would be something like (forgive the latex-esq notation formatted as code ... I'm entirely unprepared to describe model architectures without being able to write some form of equation):

`Z' = W_1 Z softmax(Z^T W_2^T W_3 Z)`

where W_1, W_2, and W_3 are weights of the layer.  However, following the general practice of graph transformer architectures, we instead use a term

`Z' = W_1 Z softmax(Z^T W_2^T W_3 Z - gamma*D)`

where D is a distance matrix defined by the graph.  For a "hard" graph transformer, this will work like the mask in normal self-attention layers, and be infinite for nodes that are not connected, and zero for nodes that are connected (and the gamma term would be fixed to one, say).  In our "soft" version of the graph transformer, however, D was just the squared distance matrix between nodes in the graph, and gamma was a learnable parameters: as gamma went to zero, this would become a standard transformer with no notion of distance between objects, whereas as gamma went to infinity, it would become a hard graph transformer.  To be even more precise, in the final architecture we used a multi-head version of this self-attention layers, as is also common in transformer models.

As a final note, for this to work, we needed to define a distance measure between all the nodes in the graph.  For e.g., atom-to-atom distances, we just used the actual distance between atoms, for atom-to-bond distances, we would use the the minimum distance from the atom to the two atoms in the bond, with similar extensions for triplets, quads, etc.

After the self-attention layer, we used the normal fully-connected and layer-norm layers standard to transformer architectures, and used models of depth ranging from 14-16 (depending on available memory).  After the final embeddings, we had separate heads that would predict the final scalar coupling for the nodes that corresponded to fairs for which we needed the coupling value, using a simple two layer MLP for each type (or actually, for several sub-types of the bonds, which we'll mention below).

## Input features and embeddings
The input representation (i.e., the first-layer embeddings for all nodes in the network), 
As our input representation, for each type of node in the network, we would include a kind of  hierarchical embedding, where were had different levels of specificity for the different atoms, bonds, etc.

As an example, for each bond (again, really meaning just a pair of atoms ... I'm referring to pairs generally as bonds even if they are not chemical bonds in the molecule), we described it in terms of the two atoms belonging to the bond, but also in in terms of the number of bonds that each atom would have.  Thus, each bond could be described by multiple given types at subtypes: first by just the type of atoms in the bond, then by the type and total number of bonds that each atom had, and then by a few additional properties such as the bond order, etc.

This lead to substantially more coupling "types" than just the 8 that were used in the competition, and we actually had separate final layers for 33 different types of bonds, rather than the 8 in the competition (for instance, the 1JCH had very different properties depending on the number of bonds the C atom had), which definitely improved our predictions slightly.

In addition to the "discrete" embedding, each node type would have associated with it one or two scalar constants that we would embed with a Fourier encoding, much like positional encoding in a standard sequential Tranformer model.  For atoms, this consisted of the partial charge of the atom, as given by the OpenBabel library (*correction: original post said this was from RDKit, but RDKit was used for bond orders and conenctions, whereas OpenBabel was used for charges), just using some simple rules based upon graph structure; for bonds it was the distance between the two atoms; for triplets the angle between the center atom and the other others; and for quads the dihedral angle between the two planes formed by the center bond and the two other bonds (quads didn't end up helping too much for this particular task, though, so were left out most of our final models).

## Ensembling
In this end, we trained 13 models that we used for the final ensemble, which basically just corresponded to different iterations and versions of the same basic structure (at times we also included a few models based upon a more standard graph neural network approach from the PyTorch Geometric library, though they weren't included in the final ensemble).  We timed about 4 final models to complete on the final day of competition, including the model which eventually got the best performance, which is why we managed to sneak into the top spot on the very last day.  As I had mentioned in my last post, there really wasn't anything that much happening during the last ~4 days where we moved up the rankings, nor were we "holding anything back": our models simply kept improving each day, and we'd submit our best version of the ensemble, which kept bumping us up day by day.

Our best single model got about -3.08 on the public leaderboard, which makes me actually quite surprised, given that Quantum Uncertainty's best model was substantially better.  But I think the fact that we predicted entire molecules at once actually may have increased the variance of predictions across all molecules, but therefore also seemingly made it work much better with the ensembling several different models.  By taking a straight median across predictions from the best models, for instance, we could get to the ~-3.22 range, and with a slightly more involved blending scheme (using the median of all 13 models to determine which 9 models seemed best, then taking the mean of a few different medians of the different model predictions), we were able to achieve our score of -3.245 on the private leaderboard.

## Other random notes
- We used small amounts of dropout at each layer, as in standard transformer models, though found that it was best to use a very small amount of dropout.
- At the very end of the competition, we did find that for our model a kind of cutout procedure (where we would randomly drop out two atoms from the network, plus all bonds, triplets, etc, that contained this atom), worked as a very effective regularizer.
- We didn't use QM7/QM9 or in fact any of the extra data that was included in the competition besides the structures and train/test files (so just atoms and bonds).
- We used RDKit/xyz2mol and a few other packages to parse the atomic structure to e.g., the bond and neighbor configuration feautres.  Jonathan had a post about this earlier listing the packages we used.
- I'm not even going to attempt to list all the things we tried that didn't work, but there was lots :-).  Quad / dihedral angle information, for instance, actually seemed to _hurt_ generalization performance, as did including simple Coulomb forces at the bond level.
- As you'd expect, we had a fair amount of compute resources for the work.  Most of the models were trained on 4x RTX2080 Ti systems (we had 5 of these available through the month we were working on the project), with a handful also trained on six single V100s we got access to in the last week.

## Final thoughts
I want to thank the CHAMPS team for putting on an amazing competition.  As many others have pointed out, the stability between the private/public leaderboards demonstrates an understanding of how to run a machine learning contest that sadly seems to be missing from many of the other contests I looked over on Kaggle previously.

I also again want to thank the Quantum Uncertainty team, who as I mentioned before, were our goalposts the entire competition.  After reading their solution I'm coming away more convinced about Transformers as the architecture that's going to be dominant across many different domains, not just sequence models (despite the fact that I will never, ever, forgive the original paper for the monstrosity that is the "query", "key", and "value" terminology for self-attention layers ;-) ).  I also think their per-atom transformer is an awesome idea, and something I wish we had thought of ... I think most likely it took us using a _lot_ of domain knowledge and engineering to make back up the difference that their per-atom approach got.  And while it's wild to me that a non-rotationally invariant model would do so well (since we only used distance as a feature at the bond level, our model is rotationally invariant), it's impossible to argue with results.  Their model is excellent, and I think it actually goes to show there is substantial room for improvement still in the performance we can get on this task.

Thanks again,
Zico, Devin, Shaojie, Jonathan, and Mordechai