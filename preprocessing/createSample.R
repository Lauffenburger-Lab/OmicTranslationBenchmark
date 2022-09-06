# This function is used to create a sampled dataset.
# It takes as input the (data) dataframe containing the total data, 
# (pairedInfo) the dataframe of paired conditions,
# (n1) the number of drugs,
# (n2) the number of extra random conditions
# (n3) the number of extra conditions from the same drugs,
# (cells) the cell-line pair in question,
# (cmapCor) the sample wise GeX correlation matrix of the data,
# (maxIter) the maximum iterations for sampling data until the conditions have low correlation.
# It returns a sampled data-frame.

createSample <- function(data,paired,n1,n2,n3,cells,cmapCor,maxIter=100){
  
  library(tidyverse)
  
  max_n1 <- nrow(paired)
  if (n1>max_n1){
    n1 <- max_n1
  }
  data <- data %>% filter(cell_iname %in% cells)
  paired <- paired %>% mutate(pair_id=paste0(sig_id.x,'_',sig_id.y))
  total_drugs <- unique(data$cmap_name)
  #cmapCor <- reshape2::melt(cmapCor)
  
  # Sample n1 drugs
  cmapCor_n1 <- cmapCor %>% mutate(pair_id=paste0(Var1,'_',Var2))
  cmapCor_n1 <- cmapCor_n1 %>% filter(!(pair_id %in% paired$pair_id))
  minCor <- 100
  for (j in 1:maxIter){
    if (length(unique(paired$cmap_name))<n1){
      drugs <- unique(paired$cmap_name)
    }else{
      drugs <- sample(unique(paired$cmap_name),n1)
    }
    conditions_n1 <- paired %>% filter(cmap_name %in% drugs)
    #cmap1 <- t(cmap[,unique(conditions_n1$sig_id.x)])
    #cmap2 <- t(cmap[,unique(conditions_n1$sig_id.y)])
    #conds_cor <- cor(c(cmap1),c(cmap2))
    conds_cor <- cmapCor_n1 %>% filter(Var1 %in% conditions_n1$sig_id.x & Var2 %in% conditions_n1$sig_id.y)
    if (mean(conds_cor$value)<minCor){
      sigs_n1 <- unique(c(conditions_n1$sig_id.x,conditions_n1$sig_id.y))
      drugs_n1 <- unique(conditions_n1$cmap_name)
      minCor <- mean(conds_cor$value)
    }
  }
  print(paste0('Paired conditions correlation:',minCor))
  
  # Sample n2 other drugs
  other_drugs <- total_drugs[which(!(total_drugs %in% drugs_n1))]
  minCor <- 100
  for (j in 1:maxIter){
    drugs <- sample(other_drugs,n2)
    conditions_n2 <- data %>% filter(cmap_name %in% drugs)
    conditions_n2 <- sample_n(conditions_n2,n2)
    #cmap1 <- t(cmap[,unique(conditions_n2$sig_id.x)])
    #cmap2 <- t(cmap[,unique(conditions_n2$sig_id.y)])
    #conds_cor <- cor(c(cmap1),c(cmap2))
    conds_cor <- cmapCor %>% filter(Var1 %in% conditions_n2$sig_id & Var2 %in% conditions_n2$sig_id)
    if (mean(conds_cor$value)<minCor){
      sigs_n2 <- unique(conditions_n2$sig_id)
      drugs_n2 <- unique(conditions_n2$cmap_name)
      minCor <- mean(conds_cor$value)
    }
  }
  print(paste0('Extra drugs correlation:',minCor))
  
  # Sample n3 other drugs
  minCor <- 100
  for (j in 1:maxIter){
    drugs <- sample(c(drugs_n1,drugs_n2),n3)
    conditions_n3 <- data %>% filter(cmap_name %in% drugs) %>% filter(!(sig_id %in% sigs_n1) & !(sig_id %in% sigs_n2))
    if (nrow(conditions_n3)<n3){
      drugs <- c(drugs,sample(total_drugs[which(!(total_drugs %in% c(drugs_n1,drugs_n2)))],n3-nrow(conditions_n3)))
      conditions_n3 <- data %>% filter(cmap_name %in% drugs) %>% filter(!(sig_id %in% sigs_n1) & !(sig_id %in% sigs_n2))
    }
    if (nrow(conditions_n3)>n3){
      conditions_n3 <- sample_n(conditions_n3,n3)
    }
    conds_cor <- cmapCor %>% filter(Var1 %in% conditions_n3$sig_id & Var2 %in% conditions_n3$sig_id)
    if (mean(conds_cor$value)<minCor){
      sigs_n3 <- unique(conditions_n3$sig_id)
      drugs_n3 <- unique(conditions_n3$cmap_name)
      minCor <- mean(conds_cor$value)
    }
  }
  print(paste0('Extra conditions for same drugs correlation:',minCor))
  
  sigs <- unique(c(sigs_n1,sigs_n2,sigs_n3))
  data <- data %>% filter(sig_id %in% sigs)
  
  cmapCor <- cmapCor %>% filter(Var1 %in% data$sig_id & Var2 %in% data$sig_id)
  print(paste0('Average sample-wise correlation:',mean(cmapCor$value)))
  
  
  return(data)
  
}
