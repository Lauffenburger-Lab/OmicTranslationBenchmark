## Algorithms for the pre-processing of the raw data
1. preprocessingL1000.R: Script to pre-process and estimate the quality of the raw transcriptomic data for a pair of cell lines.
2. AddSequentiallyCells.R: Script to add one more cell line each time.
3. controlSigs.R: Pre-process data to add control transcriptomic signatures.
4. createSample.R: Function is used to create a sampled dataset to be used later in performance analysis for Figure 2. It is called inside the preprocessingL1000.R script.
5. distance_scores.R: This function calls the SCoreGSEA function (it calculates a Kolmogorov-Smirnof-based distance) and returns an NxN matrix with the distance between each column/sample. It is used to calculate a GSEA-based distance between samples based on some omic or enrichment profile.

## Folder structure
1. preprocessed_data: It contains the pre-processed data from the above scripts to be used later for training models and other downstream analyses of the study.

The produced data are used to create the figures stored in the figures folder and the results in the results folder.
