# Thanks
First of all, a big Thank You to the organizers for this outstanding competition. This competition is very challenging and interesting in many points of view.

Secondly, congratulations to all teams which completed this competition, whether you are in the medals or not. We learned a lot from your solutions and your discussions. We didn’t have any expertise in neither chemistry nor in graph neural net before this competition, this experience has been very enlightening for the 3 of us.

Also big kudos to the top 4 teams, your usage of Transformers are quite eye opening (honestly we did think about it but was not audacious/confident enough to test it out). Maybe this competition will open a new paradigm of Deep Learning for molecular properties - Transformer is all you need :D

A last thank you to my teemmates and coworkers Lam Dang and Thanh Tu Nguyen :) It was very fun competing with you.

# Solution
Without further ado, here is a highlight of our solution:
- On macro level our best submission is a 2 layer stacking:
- The base level consists of different variant of the general Graph Neural Net with edge, node and global representation with some variations (cf. Architectures below)
- It was implemented with pytorch and pytorch_geometric.
- The 2nd level is some metamodel trained on our validation set of 5000 molecules : 1 linear stacking model and 1 LGBM (cf. Stacking section below)
- The final submission is a blend of 2 meta model

## Architecture:
The final architecture is based on the paper https://arxiv.org/abs/1812.05055.
We tried different variations to improve this architecture, here is a summary of what worked and what didn’t work:
- Normalization: We found that LayerNorm worked better that BatchNorm for this data and helps improve convergence
- Softplus vs ReLU: Softplus did provide a ~ 0.1 boost of logMAE for our models vs a ReLU baseline
- Edge to node message gating: We found that adding some gating mechanism to the edge representation before the scatter_mean (see torch_geomrtric) for node update helps
- Edge to edge convolution: Guillaume implemented something that seem to work very well. He noticed after a feature importance test that the most important one was by far was the angle between an edge and the edge with the closest atom to the first edge. To integrate this angle feature for more than the closest edge, we updated each edge with a convolution of the edge in question and its neighboring edges in the graph (more specifically the neighboring edges that chemically connects two atoms), and putting in this convolution the angle of the edge vectors. This architecture tweek made our architecture 5 times slower but gave us a 0.15 improvement compared to the best model without it.
- 1 prediction tail per type: All types share a GNN “body”, but we found that having different MLP for each type helps.
- In some variants, before feeding into output MLP layers, we pool all the edges and nodes in the chemical bond path from atom_0 to atom_1. It seems to have helped in the beginning of the competition but our best model did not use it.
- For our architectures, we found that having a representation of the link between atom_0 and atom_1 is important. Also including the global representation as inputs of the top layers is important

## Stacking:
- Our single best model is the one with edge to edge convolution which gives us -2.9. But we have various models around -2.7 which are the variants of it. By stacking all (20 models) we got -3.13 on LB. 
- Another thing we found out at the last day helps improve our score from -3.13 to - 3.15 is adding checkpoints of our models to stacking pipeline. So finally, we have 50 predictions to do stacking.

Our final result is a blend of LGBM and HuberRegressor. 
- LGBM: 20 GroupKFold on all bond types together
- HuberRegressor: 20 GroupKFold on every bond type separately. 

## Computation:
We have : 
- 1 GTX 1080ti x 2 months + 1 RTX 2080ti x 1 month
- 1 RTX 2080ti x 3 month
- 1 V100 x 2 month (rented) + 4 V100 x 2 weeks (rented)