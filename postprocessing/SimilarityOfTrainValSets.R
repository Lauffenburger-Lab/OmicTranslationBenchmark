library(tidyverse)
library(lsa)
library(ggplot2)
library(ggpubr)
library(ggridges)
library(doFuture)
# parallel: set number of workers
cores <- 14
registerDoFuture()
plan(multisession,workers = cores)
library(doRNG)
#Process function to add condition ids and duplicate ids
process_embeddings <- function(embbedings,dataInfo,sampleInfo){
  dataInfo <- dataInfo %>% select(sig_id,cmap_name,duplIdentifier) %>% unique()
  
  sampleInfo <- left_join(sampleInfo,dataInfo)
  
  embbedings <- embbedings %>% rownames_to_column('sig_id')
  
  embs_processed <- left_join(embbedings,sampleInfo)
  
  return(embs_processed)
}

# Load samples info
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% mutate(quality_replicates = ifelse(is_exemplar_sig==1 & qc_pass==1 & nsample>=3,1,0))
sigInfo <- sigInfo %>% filter(pert_type=='trt_cp')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
# Filter based on TAS
sigInfo <- sigInfo %>% filter(tas>=0.3)
# Duplicate information
sigInfo <- sigInfo %>% mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>% mutate(dupl_counts = n()) %>% ungroup()
# Drug condition information
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))
# duplicateSigs <- sigInfo %>% filter(dupl_counts>1)
# duplicatesIndentity <- unique(duplicateSigs$duplIdentifier)

# load A375/HT29 cmap
cmap <- data.table::fread('../preprocessing/preprocessed_data/cmap_HT29_A375.csv',header=T)
colnames(cmap)[1] <- 'sig_id'
sigInfo <- sigInfo %>% filter(sig_id %in% cmap$sig_id)
gc()


## calculate all pairwise GSEA distances
# Specify the thresholds to check
# bottom and top regulated genes
thresholds <- c(30,50,100,200,300,400,
                500,600,700,800,900,1000)
# Initialize empty list for the results:
# Each element of the list (for each threshold)
# contains an NxN matrix with comparing all these
# samples. Each element of the matrix is the
# GSEA distance.
dist_all <- NULL
### SOS:
### RUN FIRST THE distance_scores.R
### SCRIPT TO LOAD THE FUNCTION!!!
### calculate distances: SEE distance_scores.R
# for information about the function inputs
dist_all <- foreach(thres = thresholds) %dorng% {
  distance_scores(num_table = t(cmap %>% column_to_rownames('sig_id')) ,
                  threshold_count = thres,names = cmap$sig_id)
}
# Transform list to array
distance <- do.call(cbind,dist_all)
distance <- array(distance,
                  c(dim=dim(dist_all[[1]]),length(dist_all)))
saveRDS(distance,'../preprocessing/preprocessed_data/gsea_distances_a375_ht29.rds')
# Get the average distance across thresholds
mean_dist <- apply(distance, c(1,2), mean, na.rm = TRUE)
colnames(mean_dist) <- cmap$sig_id
rownames(mean_dist) <- cmap$sig_id
### Convert matrix into data frame
# Keep only unique (non-self) pairs
mean_dist[lower.tri(mean_dist,diag = T)] <- -100
dist <- reshape2::melt(mean_dist)
dist <- dist %>% filter(value != -100)
# Merge meta-data info and distances values
dist <- left_join(dist,sigInfo,by = c("Var1"="sig_id"))
dist <- left_join(dist,sigInfo,by = c("Var2"="sig_id"))
dist$value <- dist$value/2

sim_all_a375 <- data.frame()
sim_all_ht29 <- data.frame()
for (i in 0:9){
# for (i in c(1,2,3)){
  # Load train, validation info
  trainInfo <- rbind(data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/train_a375_',i,'.csv'),header = T) %>% column_to_rownames('V1'),
                     data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/train_ht29_',i,'.csv'),header = T) %>% column_to_rownames('V1'))
  trainPaired = data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/train_paired_',i,'.csv'),header = T) %>% column_to_rownames('V1')
  trainInfo <- rbind(trainInfo,
                     trainPaired %>% select(c('sig_id'='sig_id.x'),c('cell_iname'='cell_iname.x'),conditionId),
                     trainPaired %>% select(c('sig_id'='sig_id.y'),c('cell_iname'='cell_iname.y'),conditionId))
  trainInfo <- trainInfo %>% unique()
  trainInfo <- trainInfo %>% select(sig_id,conditionId,cell_iname)
  
  valInfo <- rbind(data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_a375_',i,'.csv'),header = T) %>% column_to_rownames('V1'),
                   data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_ht29_',i,'.csv'),header = T) %>% column_to_rownames('V1'))
  valPaired = data.table::fread(paste0('../preprocessing/preprocessed_data/10fold_validation_spit/val_paired_',i,'.csv'),header = T) %>% column_to_rownames('V1')
  valInfo <- rbind(valInfo,
                   valPaired %>% select(c('sig_id'='sig_id.x'),c('cell_iname'='cell_iname.x'),conditionId),
                   valPaired %>% select(c('sig_id'='sig_id.y'),c('cell_iname'='cell_iname.y'),conditionId))
  valInfo <- valInfo %>% unique()
  valInfo <- valInfo %>% select(sig_id,conditionId,cell_iname)
  
  # Load embeddings of pre-trained
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev3_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev3_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev3_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev3_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')

  embs_proc_train <- process_embeddings(embs_train,sigInfo,trainInfo)
  embs_proc_test <- process_embeddings(embs_test,sigInfo,valInfo)
  ## calculate for A375 
  train <- data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev3_',i,'_a375.csv'),
                             header = T) %>% unique() %>% column_to_rownames('V1')
  train <- process_embeddings(train,sigInfo,trainInfo)
  train <- train[,1:(ncol(train)-4)] %>% column_to_rownames('sig_id')
  val <- data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev3_',i,'_a375.csv'),
                           header = T) %>% unique() %>% column_to_rownames('V1')
  val <- process_embeddings(val,sigInfo,valInfo)
  val <- val[,1:(ncol(val)-4)] %>% column_to_rownames('sig_id')
  sim_a375 <- as.matrix(cosine(t(rbind(train,val))))
  sim_a375 <- sim_a375[1:nrow(train),(nrow(train)+1):ncol(sim_a375)]
  colnames(sim_a375) <- rownames(val)
  rownames(sim_a375) <- rownames(train)
  sim_a375 <- reshape2::melt(sim_a375)
  sim_a375 <- sim_a375 %>% filter(!is.na(value))
  # Merge meta-data info and sim_a375ances values
  sim_a375 <- left_join(sim_a375,rbind(embs_proc_train,embs_proc_test) %>% 
                     select(sig_id,conditionId,duplIdentifier,cell_iname,cmap_name),by = c("Var1"="sig_id"))
  sim_a375 <- left_join(sim_a375,rbind(embs_proc_train,embs_proc_test) %>% 
                     select(sig_id,conditionId,duplIdentifier,cell_iname,cmap_name),by = c("Var2"="sig_id"))
  sim_a375 <- sim_a375 %>% filter(!is.na(value))
  png(paste0('../figures/TrainValSimilarities/embs_basev3_',i,'_a375.png'),
      width=9,height=6,units = "in",res=600)
  ggplot(sim_a375,aes(x=value)) + geom_histogram(bins = 200,fill = '#125b80',color='black') +
       ggtitle('Histogram of similarity between validation and train embeddings in A375')+
       xlab('cosine similarity')+ theme(base_family = "Arial") + 
    theme_pubr(base_family = "Arial",base_size = 14) + 
    theme(plot.title = element_text(hjust = 0.5))
  dev.off()
  
  ## calculate for HT29
  train <- data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev3_',i,'_ht29.csv'),
                             header = T) %>% unique() %>% column_to_rownames('V1')
  train <- process_embeddings(train,sigInfo,trainInfo)
  train <- train[,1:(ncol(train)-4)] %>% column_to_rownames('sig_id')
  val <- data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev3_',i,'_ht29.csv'),
                           header = T) %>% unique() %>% column_to_rownames('V1')
  val <- process_embeddings(val,sigInfo,valInfo)
  val <- val[,1:(ncol(val)-4)] %>% column_to_rownames('sig_id')
  sim_ht29 <- as.matrix(cosine(t(rbind(train,val))))
  sim_ht29 <- sim_ht29[1:nrow(train),(nrow(train)+1):ncol(sim_ht29)]
  colnames(sim_ht29) <- rownames(val)
  rownames(sim_ht29) <- rownames(train)
  sim_ht29 <- reshape2::melt(sim_ht29)
  sim_ht29 <- sim_ht29 %>% filter(!is.na(value))
  # Merge meta-data info and simances values
  sim_ht29 <- left_join(sim_ht29,rbind(embs_proc_train,embs_proc_test) %>% 
                     select(sig_id,conditionId,duplIdentifier,cell_iname,cmap_name),by = c("Var1"="sig_id"))
  sim_ht29 <- left_join(sim_ht29,rbind(embs_proc_train,embs_proc_test) %>% 
                     select(sig_id,conditionId,duplIdentifier,cell_iname,cmap_name),by = c("Var2"="sig_id"))
  sim_ht29 <- sim_ht29 %>% filter(!is.na(value))
  png(paste0('../figures/TrainValSimilarities/embs_basev3_',i,'_ht29.png'),
      width=9,height=6,units = "in",res=600)
  ggplot(sim_ht29,aes(x=value)) + geom_histogram(bins = 200,fill = '#125b80',color='black') +
    ggtitle('Histogram of similarity between validation and train embeddings in HT29')+
    xlab('cosine similarity')+ theme(base_family = "Arial") + 
    theme_pubr(base_family = "Arial",base_size = 14) + 
    theme(plot.title = element_text(hjust = 0.5))
  dev.off()
  
  sim_all_a375 <- rbind(sim_all_a375,sim_a375 %>% mutate(fold=i) )
  sim_all_ht29 <- rbind(sim_all_ht29,sim_ht29 %>% mutate(fold=i) )
}

png('../figures/TrainValSimilarities/embs_basev3_ridge_a375.png',
    width=9,height=9,units = "in",res=600)
ggplot(sim_all_a375, aes(x = value, y = as.factor(fold))) +
  geom_density_ridges(fill = '#125b80',color='black') +
  ggtitle('Distribution of similarity between validation and train embeddings in A375')+
  xlab('cosine similarity') + ylab('fold-split')+ theme(base_family = "Arial") + 
  theme_pubr(base_family = "Arial",base_size = 14) + 
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

png(paste0('../figures/TrainValSimilarities/embs_basev3_ridge_ht29.png'),
    width=9,height=9,units = "in",res=600)
ggplot(sim_all_ht29, aes(x = value, y = as.factor(fold))) +
  geom_density_ridges(stat = "binline",bins = 100,alpha = 0.8,
                      fill = '#125b80',color='black') +
  ggtitle('Distribution of similarity between validation and train embeddings in HT29')+
  xlab('cosine similarity') + ylab('fold-split')+ theme(base_family = "Arial") + 
  theme_pubr(base_family = "Arial",base_size = 14) + 
  theme(plot.title = element_text(hjust = 0.5))
dev.off()
