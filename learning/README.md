## Machine and Deep learning algorithms of the project
1. TranslationalAutoencoder_DEEPCELLSTATE_benchmarkGenes.ipynb: Jupyter notebook to train an autoencoder with the DeepCellState approach[^1], evaluate it with 10-fold cross-validation and get embeddings in the latent space.
2. TranslationalAutoencoder_benchmarkGenes.ipynb: Jupyter notebook to train multiple encoders and decoders for each cell line, evaluate it with 10-fold cross validation and get embeddings in the latent space.
3. TranslationalAutoencoder_benchmarkMI.ipynb: Jupyter notebook to train multiple encoders and decoders for each cell line coupled with a Mutual Information similarity approach[^2],[^3], evaluate it with 10-fold cross-validation and get embeddings in the latent space.
4. TranslationalAutoencoder_important.py: Script containing code for importance calculation using the integrated gradients approach[^4], for many of the panels in Figure 4. 
5. TranslationalAutoencoder_important.ipynb: Jupyter notebook to calculate gene importance for various tasks by utilizing an integrated gradients approach[^4] from the Captum library[^5].
6. models.py: Contains classes to define layers for the autoencoder and the encoder and decoder models.
7. evaluationUtils.py: Contains functions for evaluation of the models.
8. trainingUtils.py: Contains function to train an autoencoder.
9. TranslationalAutoencoder_CPA.py: Contains code to train an autoencoder with the CPA-based approach[^6]
10. CellLineSpecificAnalysis.ipynb: Implement cell-line specific analysis (train machine learning models like random forest and ANN-classifier or perform PCA and t-SNE analysis of raw data and pre-trained embeddings)
11. implementFIT_sklearn.ipynb: Implement FIT approach[^7].  
12. EncodeControls.py: Script containing code to get embeddings for control samples from the L1000 dataset.
13. EncodeCCLE.py: Script containing code to get CCLE embeddings.
14. TranslationalAutoencoder_CPA_nodecoder.py: Script to train a CPA-based model without decoders.
15. TranslationalAutoencoder_jointlyTrainedClassifier.py: Script containing code to train a model with one global latent space and simultaneously training a classifier to predict cell line label, which is a contradicting learning task.
16. TransCompRModel.py: Script containing code to train a model based on TransCompR[^8].
17. CellLineClassification_TransCompR_DCS.ipynb: Script containing code to train classifiers on TransCompR and DeepCellState embeddings for Figure 1e.
18. CPA_PerformaceAnalysis_lands.py: Script containing code to run the performance analysis from Figure 2, for the CPA-based approach, using the landmark genes.
19. CPA_PerfomanceAnalysis.py: Script containing code to run the performance analysis from Figure 2, for the CPA-based approach, but using the 10,086 genes. 
20. CPA_PerformancePretrainClass.py: Script containing code to pre-train the adverse classifier for the performance analysis from Figure 2, for the CPA-based approach, using the landmark genes.
21. IntermediateTranslators.ipynb: Jupyter notebook to train CPA-based model but with intermediate ANNs to add the cell line effect.

## References
[^1]: Umarov, Ramzan, Yu Li, and Erik Arner. "DeepCellState: An autoencoder-based framework for predicting cell type specific transcriptional states induced by drug treatment." PLoS Computational Biology 17.10 (2021): e1009465.
[^2]: Sun, Fan-Yun, et al. "Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization." arXiv preprint arXiv:1908.01000 (2019).
[^3]: Veličković, Petar, et al. "Deep graph infomax." arXiv preprint arXiv:1809.10341 (2018).
[^4]: Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic attribution for deep networks." International conference on machine learning. PMLR, 2017.
[^5]: Kokhlikyan, Narine, et al. "Captum: A unified and generic model interpretability library for pytorch." arXiv preprint arXiv:2009.07896 (2020).
[^6]: Lotfollahi, Mohammad, et al. "Compositional perturbation autoencoder for single-cell response modeling." BioRxiv (2021).
[^7]: Normand R, Du W, Briller M, Gaujoux R, Starosvetsky E, Ziv-Kenet A, Shalev-Malul G, Tibshirani RJ, Shen-Orr SS. Found In Translation: a machine learning model for mouse-to-human inference. Nat Methods. 2018 Dec;15(12):1067-1073. doi: 10.1038/s41592-018-0214-9. Epub 2018 Nov 26. PMID: 30478323.
[^8]: Brubaker, D. K. et al. An interspecies translation model implicates integrin signaling in infliximab-resistant inflammatory bowel disease. Science Signaling 13, eaay3258 (2020).


