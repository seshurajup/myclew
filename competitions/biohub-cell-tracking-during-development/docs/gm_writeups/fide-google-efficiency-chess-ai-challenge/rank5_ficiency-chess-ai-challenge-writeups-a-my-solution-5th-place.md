# My solution (5th place)

The source code for my submission is available [HERE](https://github.com/Rafbillpc/kaggle-chess).

I'll keep this writeup short, as I don't have much to say that was not already covered by the writeup of the top solutions.

My solution is a modification of the Obsidian chess engine. Ignoring the size of the nnue net, I was able to fit the compressed executable in ~25Kb and the memory usage below the 5Mb budget, without any significant loss in strength.

The code is compiled to a shared library that is dynamically loaded by the python script. I think that this saves some memory, because the same stack can be used by python and the shared library.

I didn't implement pondering.

The main original part of my submission is the structure of the nnue: it is 768->128x2->16->32->1, but the feature transformer 768->128 only uses as much space as a 768x64 matrix.

This is accomplished by sharing some weights: the 768x128 matrix has the following shape:
```
 +-------+-------+
 |       |       |
 |   A   |   B   |
 |       |       |
 +-------+-------+
 |       |       |
 |   B   |   A   |
 |       |       |
 +-------+-------+
```
where the pair of features (x, x+384) corresponds to a piece on a square in file (A-D), and the same piece on the horizontally mirrored square (in file E-H).

That has the effect of building in the structure of the network some knowledge about the rules of chess, namely that the rules of chess are symmetrical (if you ignore castling, which is not really handled by the network anyway).

The resulting network is not as strong as with an ordinary 768->128 feature transformer, but it stronger than with an ordinary 768->64 feature transformer.