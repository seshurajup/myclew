## Acknowledgements

I would like to thank the competition hosts and Kaggle for the opportunity to work on such an impactful and interesting problem.  
Thank you @jaejohn and @theoviel for sharing strong solutions at the beginning of the competition.  
Credit to Claude models for the code assistance.

## Overview 

My strategy for the competition was to build upon the provided strong solutions TBM, RNAPro and Protenix by incorporating into the modeling new challenges added in Part 2: proteins, DNA, ligands, multiple chains and long sequences.  
Another aspect I considered important was processing speed in order to be able to include diverse predictions in the submission within the runtime constraints.  
The solution combines two predictions from TBM, two predictions from Protenix, and one prediction from RNAPro for all targets. There is no additional logic for selecting specific predictions per RNA sequence.  
RNAPro and Protenix inferences run in parallel on the two T4 GPUs. 

## Template-based modeling (TBM)

For template-based modeling, a LightGBM model predicting TM-score for a given query-template pair is introduced. From the competition train data, the 1000 most recent query RNA sequences and 200 (per query) template sequences with a preceding temporal cutoff are used to create a training dataset of ~198K query-template pairs. TM-score for each pair is computed and used as a label. The model features are:

* RNA sequences similarity: `alignment_score`, `percent_identity`, and `len_diff_ratio`  
* Text embeddings similarity: cosine similarity between query description and template description embeddings.  The embeddings are created with this model: https://huggingface.co/NeuML/pubmedbert-base-embeddings
* Proteins similarity: alignment scores with a BLOSUM62 substitution matrix. For each query protein, the max alignment score with template proteins is taken. The min, mean, and max of the resulting vector are added as features. For processing speed considerations, only up to 6 query proteins are compared with up to 20 template proteins, and protein sequences are cropped to a max length of 768\. These features are named `protein_similarity_matrix_(min|mean|max)` in the model.  
* DNA similarity: `dna_similarity_matrix_(min|mean|max)` features calculated with the logic described above for proteins using alignment scores without a substitution matrix.  
* Ligands similarity: `ligand_similarity_matrix_(min|mean|max)` features using Tanimoto similarities of Morgan fingerprints calculated with the logic described above for proteins.  
* Composition count features: `num_query_(proteins|dna|ligands)` and `num_template_(proteins|dna|ligands)`  
* Chains-related features: `(query|template)_num_all_chains`, `(query|template)_num_unique_chains`, and `chain_counts_match`

The top-2 templates according to the scores predicted by this model are used for the final submission.  

The script for running this step is `srna3d.scripts.create_tbm_submission.`  
The script for creating the training dataset is `srna3d.scripts.create_tmscore_dataset`. The dataset was created locally.  
Notebook how to run it: https://www.kaggle.com/code/stefanstefanov/srna3d2-create-tmscore-dataset  
The training dataset: https://www.kaggle.com/datasets/stefanstefanov/srna3d2-tmscore-dataset  
The model was also trained locally. The script for training is `srna3d.scripts.train_tmscore_lightgbm_model`   
Training notebook: https://www.kaggle.com/code/stefanstefanov/srna3d2-train-tmscore-model  

## RNAPro

One RNAPro model prediction is generated using the top-5 templates from the TBM step as input. To speed up inference, diffusion steps are decreased to 100, MSA depth is limited to 2048, and `fast_layernorm` and `triattention` triangle attention kernels are used, as inference is faster with them on longer sequences. As long sequences are split into chunks, the MSA and templates are also sliced to be consistent with the predicted chunk start and end index.

## Protenix

Two predictions are generated with the latest Protenix model `protenix_base_20250630_v1.0.0`. In addition to the RNA sequence, proteins, DNA, and ligands are also added as input to the Protenix model. To fit in the T4 GPU memory and time constraints, proteins, DNA, and ligands are limited to a maximum of 6 per group. A `max_len_non_rna` parameter is added and set to `128`. This non-RNA sequence length budget is divided equally among protein and DNA sequences, and they are cut accordingly. The idea is to model the effect of short protein and DNA sequences on the RNA 3D structure, at least for targets with a small number of proteins or DNA. Similar to RNAPro, MSA depth is limited to 2048, and `fast_layernorm` and `triattention` triangle attention kernels are used.

## Long Sequences

For the RNAPro and Protenix models, RNA with a sequence length above 448 is split in two steps:

1) If it has multiple chains, it is first split into chains, and for each chain, an overlap of 96 from the next chain is added.  
   For the last chain, the overlap is added circularly from the first chain with the idea to try to capture potential circularity in structure.  
   For faster prediction, repeated chain pairs are skipped. For example, `9MME` which has U:8 chains with a total length of 4640 (8x580) is split and passed to the next step as one sequence of length 676 (580 from the first U + 96 from the second U chain).  
   If the target has multiple chains, but its total sequence length is below 448, no such splitting is performed.
This splitting step is performed by the `srna3d.scripts.split_into_chains` script.

2) Sequences from the previous step that are longer than 448 are split into chunks of length 448, having a 96-base overlap with the next chunk. This step is performed by the `srna3d.scripts.split_sequences_into_chunks` script.

Chunk predictions are first combined from chunks to chains using the `srna3d.scripts.combine_chunked_predictions` script. Afterward, they are combined from chains to full sequences with the `srna3d.scripts.combine_chain_predictions` script, applying Kabsch alignment on the overlapping regions. There is an option to randomly select and combine chain predictions from different samples, but it isn’t applied; for RNAPro, one sample is generated, and for Protenix, two samples are generated.

The solution includes various limits due to GPU memory and runtime constraints. It would be interesting to see what improvements this and other solutions could achieve with faster GPUs having more memory.

## References

Rao, G. John, RNAPro inference with TBM notebook: https://www.kaggle.com/code/jaejohn/rnapro-inference-with-tbm

Viel, Theo, et al. RNAPro Inference notebook: https://www.kaggle.com/code/theoviel/stanford-rna-3d-folding-pt2-rnapro-inference

Lee, Youhan, et al. (2025). Template-based RNA structure prediction advanced through a blind code competition. *bioRxiv*. doi: 10.64898/2025.12.30.696949.   
https://github.com/NVIDIA-Digital-Bio/RNAPro

Zhang, Yuxuan, et al. (2026). Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction. *bioRxiv*. doi: 10.64898/2026.02.05.703733.  
https://github.com/bytedance/Protenix