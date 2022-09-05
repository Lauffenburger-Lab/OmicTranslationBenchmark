# This function is copied from this repository: 
# https://github.com/BioSysLab/deepSIBA
# This repository is part of this publication:
# https://doi.org/10.1039/D0MO00129E

# It is not modified or changed apart from
# adding some comments for explanation.

# This function takes as input: (i) a numeric table (num_table)
# which contains lists (each column) with scores for each element
# (it can be gene expression data, pathway enrichment scores ect)
# (ii) a number (threshold_count) of how many 
# top and bottom (after ranking) elements of the list to compare
# and (iii) the names of the samples (each column of the list)
# It calls the SCoreGSEA function (it calculates a 
# Kolmogorov-Smirnof based distance) and 
# returns a NxN matrix with the distance between each column/sample. 

distance_scores <- function(num_table, threshold_count, names) {
  library(GeneExpressionSignature)
  library(tidyverse)
  
  ### rank the table
  table_ranked <- apply(X = -num_table, MARGIN = 2, 
                        FUN = rank, ties.method = "random")
  
  ### create the phenodata
  pheno2 <- as.data.frame(colnames(num_table))
  rownames(pheno2) <- colnames(num_table)
  pheno_new <- new("AnnotatedDataFrame",data=pheno2)
  ### create expression set
  expr_set <- new("ExpressionSet",exprs = table_ranked, 
                  phenoData=pheno_new)
  ### calculate distances
  distances <- ScoreGSEA(expr_set , threshold_count,"avg")
  colnames(distances) <- names
  rownames(distances) <- names
  return(distances)
}
