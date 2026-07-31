# Overview
I am absolutely in the stars 😉 at reaching #3 in this competition. Despite a huge amount of research and analysis up front in the first few days,  I was not at all confident of reaching even the top 100 given that most of my analysis seemed to point at the competition leaving quite a bit down to the luck of the private draw and some extremely strong starter notebooks (@cdeotte 👀) being published that I thought were already pretty close to the Bayes' noise floor! 

As many others found, most strong models even of different families tended to agree on the "core" set of rows where the galaxies, stars and quasars could be cleanly separated. After all, in real astronomical data redshift alone is a very strong deterministic discriminator (acting as a proxy for cosmological distance). Broadly speaking:
- Stars (z=0)
- Galaxies (0<z<1)
- Quasars (z>1)

In may ways, this would not have made for a very interesting competition, and so the fact that the synthetic generating process is not fully structure-preserving (it matches the feature marginals well, but seems to blur or discard joint dependencies) ensured a clean separation only using redshift would not be so easy. (We saw a similar effect in S6E5 where across lap dependencies were scrambled.)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F341622%2Fd547172f8d96f25d651b3857e6604db9%2Fconcept_shift.png?generation=1783253044663301&alt=media)

Instead, as many found, there was considerable blurring of the redshift so galaxies could easily be confounded with stars at low z, and with quasars around z~1.0.

The difficulty/problem though is that this same blurring of redshift/the radial dimension, effectively creates degenerate regions as photometry alone is insufficient to discriminate between the stellar objects!

In the end, I believe my good finish came down to quite a bit of luck, but also the strength of:
1.  Adding as much "breadth" to my model as possible, analogous to adding de-correlated signals in the systematic investment world .

2. A small lift from understanding that while radial distance was scrambled (redshift as the proxy) the sky-plane was preserved (via Right Ascension and Declination) and taking advantage of that via both target encoding, and using HEALPix ("Hierarchical Equal Area isoLatitude Pixelization").

## Breadth

In systematic investing, the fundamental law of active management states that IR = IC x sqrt(Breadth).

Any strong models  (that I found) tended to be highly correlated with each other, so the real trick (for me YMMV) was in adding *weaker* decorrelated legs to my ensemble. I actually ended up with an ensemble of around 180 models (apologies at the time of writing I don't have full access to my files) which I am actually not sure was necessary particularly on this competition, but I was determined to do better than in the F1 pit stop prediction competition where I saw that many winners did use very large ensembles! The real work was probably done with only a few of the weaker decorrelated legs, including surprisingly (for me) a GMM Bayes Classifier(!)

## "Voxels" out, "HEALPix" in.

I think the only other interesting point of note that actually led to some lift, was exploiting the fact that the sky-plane itself was preserved, even though the radial distance was scrambled. I was not the only one to notice this in the end and there were some good discussion posts and public notebooks investigating this (although the good notebooks sadly tended to be lost in the morass of low quality blending notebooks, that somehow always get voted up). This is a good example of what I thought was an interesting notebook: [Spatial KNN Class-Fraction Features](https://www.kaggle.com/code/omadon/s6e6-spatial-knn-class-fraction-features) that I was concerned at the time would make any small "edge" I might possibly have here common knowledge and really deserved more visibility.

I had actually even tried the idea of creating "voxels" by generateding the equivalent of 3d co-ordinates using ASTROpy and adding a cosmological model to convert redshift to radial distance. (This was before I understood that this was doomed to failure due to the DGP!)

Readers may find it interesting (although It did not necessarily lead to any real lift above the kNN class-fraction type approach, that I had also independently tried) that one way to take advantage of the SDSS17 data which I felt was unique or original (or at least undiscussed) was to use HEALPix to divide up the sky-plane into small pixels. Basically, the idea being that given Ra and Declination, we can map each stellar object to a HEALPix pixel ID and use that as a spatial feature for target encoding.

Congratulations to @cdeotte for some impressive starter notebooks, @Optimistix for back-to-back #1 wins and all the others who respected their CV and gained 200-300 places in the public shake-up!

I apologise for not including more pictures and code examples, but I do not have access to my original files at the time of writing so can only write this up mostly from memory. I actually have a *lot* of analysis that could be shared, but the nature of this competition was such that most of the analysis was in the category of interesting, but ultimately null and not actually helpful to producing any lift!

May the folds ever be with you.