## Post-processing of embeddings and model's results.
1. CompareCohenD.R: Compare the latent space separation, based on Cohen's d, of different approaches (Supplementary Figure 14)
2. EncodedControlsAnalysis.R: Visualize in 2D the embeddings of controls, ccle, and the trained covariates (Figure 3f-3g)
3. GenesetsPerformance.R: Evaluate the performance in calculating gene set enrichment based on the predicted values from the model (Supplementary Figure 3)
4. LatentSpaceSeparation.R: Generate plots for estimating latent space embeddings separation (Figure 4a-4e)
5. PerformanceAnalysis.R: Script to perform the analysis and generate the panels for Figure 2.
6. SimilarityOfTrainValSets.R: Script to generate Supplementary Figure 5 and estimate the similarity of train and validation set, to make sure there is no leakage of information when evaluating models.
7. enrichment_calculations.R: Function to perform gene set enrichment for different types of genesets: KEGG pathways, GO Terms, MSIG genesets, even TFs but by using GSEA.
8. getRanksImportant.R: Script to perform analysis of genes importance for Figure 4.

