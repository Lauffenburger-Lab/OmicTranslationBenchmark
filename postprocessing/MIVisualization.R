library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggridges)
library(infotheo)
library(doFuture)
# parallel: set number of workers
cores <- 14
registerDoFuture()
plan(multisession,workers = cores)
library(doRNG)

#Process function to add condition ids and duplicate ids
process_embeddings <- function(embbedings,dataInfo,sampleInfo){
  dataInfo <- dataInfo %>% select(sig_id,cmap_name,duplIdentifier) %>% unique()
  
  sampleInfo <- suppressMessages(left_join(sampleInfo,dataInfo))
  
  embbedings <- embbedings %>% rownames_to_column('sig_id')
  
  embs_processed <- suppressMessages(left_join(embbedings,sampleInfo))
  
  return(embs_processed)
}

#Function to empirically calculate the Jensen-Shannon MI
JSDMI <- function(x,y,num_bins  = 10,estimation_method = "emp"){
  bins_x <- cut(x, breaks = num_bins, labels = FALSE)
  bins_y <- cut(y, breaks = num_bins, labels = FALSE)
  mi <- mutinformation(bins_x , bins_y , method=estimation_method)
  return(mi)
}

###Load data------------
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


### Estimate and visualize empirical MI------------------------
dmi_empirical_all <- data.frame()
for (i in 0:9){
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
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  
  # Estimate MI from embeddings
  print(paste0('Calculate empirical MI in fold : ',i))
  # mi_empirical <- readRDS(paste0('../results/MI_results/mi_estimations/OneGlobal/empirical_mi_a375_ht29_fold',i,'.rds'))
  embs_all <- rbind(embs_train,embs_test)
  print(paste0('Calculate empirical MI in fold : ',i))
  mi_empirical_list <- foreach(j = seq(1:nrow(embs_all))) %dopar% {
      apply(embs_all,1,JSDMI,y=as.matrix(embs_all[j,]))
  }
  mi_empirical <- do.call(cbind,mi_empirical_list)
  colnames(mi_empirical) <- rownames(embs_all)
  mi_empirical <- as.data.frame(mi_empirical) %>% rownames_to_column('sig_id.x') %>% gather('sig_id.y','MI',-sig_id.x) %>% mutate(type='empirical')
  mi_empirical <- left_join(mi_empirical,sigInfo %>% select(c('sig_id.x'='sig_id'),c('conditionId.x'='conditionId')))
  mi_empirical <- left_join(mi_empirical,sigInfo %>% select(c('sig_id.y'='sig_id'),c('conditionId.y'='conditionId')))
  mi_empirical <- mi_empirical %>% mutate(relationship = ifelse(conditionId.x==conditionId.y,'same condition','different condition'))
  print(paste0('Save empirical MI in fold : ',i))
  # saveRDS(mi_empirical,paste0('../results/MI_results/mi_estimations/OneGlobal/empirical_mi_a375_ht29_fold',i,'.rds'))
  
  dmi_empirical_all <- rbind(dmi_empirical_all,mi_empirical %>% mutate(fold=i))
  print(paste0('Finished fold ',i))
  
}
gc()
p1 <- ggplot(dmi_empirical_all, aes(x = MI, y = as.factor(fold),fill = relationship)) +
  geom_density_ridges(stat = "binline",bins = 200,alpha = 0.8,
                      color='black') +
  xlab('Empirical MI') + ylab('fold-split')+ 
  theme_pubr(base_family = "Arial",base_size = 20) + 
  theme(text = element_text(family = 'Arial'))
# print(p1)
ggsave('../article_supplementary_info/empirical_mutual_information_histograms.png',
       plot = p1,
       height = 12,
       width = 9,
       units = 'in')
postscript('../article_supplementary_info/empirical_mutual_information_histograms.eps',width = 12,height = 9)
ggplot(dmi_empirical_all, aes(x = MI, y = as.factor(fold),fill = relationship)) +
  geom_density_ridges(stat = "binline",bins = 200,
                      color='black') +
  xlab('Empirical MI') + ylab('fold-split')+ 
  theme_pubr(base_family = "Arial",base_size = 20) + 
  theme(text = element_text(family = 'Arial'))
dev.off()

### See the effect of uniformity------------------------
mi_all <- data.frame()
for (i in 0:9){
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
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  
  sds <- apply(rbind(embs_train,embs_test),2,sd)
  mus <- apply(rbind(embs_train,embs_test),2,mean)
  covariance_matrix <- cov(rbind(embs_train,embs_test))
  random_normal <- MASS::mvrnorm(n=85,mu = mus,Sigma = covariance_matrix)
  random_uniform <-  replicate(292, runif(n = 85,min = 1e-03,max=0.999))
  #add noise and remember 1:30 and 70:30 are the same
  random_normal <- rbind(random_normal,random_normal[1:15,] + 1e-3*rnorm(292))
  random_uniform <- rbind(random_uniform,random_uniform[1:15,] + 1e-3*rnorm(292))
  # Select also all the paired conditions of my data and randomly 30 paired and select also the rest to sum up to 100 sig_ids
  sample_paired <- sample_n(rbind(trainPaired,valPaired) %>% filter(sig_id.x!=sig_id.y),17)
  sample_unpaired <- sample_n(rbind(trainInfo,valInfo),70)
  embs <- rbind(embs_train,embs_test)
  embs <- embs[which(rownames(embs) %in% c(sample_paired$sig_id.x,sample_paired$sig_id.y,sample_unpaired$sig_id)),]
  embs <- distinct(embs)
  
  # Calculate MI of these vectors
  
  #First for my embeddings
  mi_embs_list <- foreach(j = seq(1:nrow(embs))) %dopar% {
      apply(embs,1,JSDMI,y=as.matrix(embs[j,]))
  }
  mi_embs <- do.call(cbind,mi_embs_list)
  colnames(mi_embs) <- rownames(embs)
  mi_embs <- as.data.frame(mi_embs) %>% rownames_to_column('sig_id.x') %>% gather('sig_id.y','MI',-sig_id.x) %>% mutate(type='sampled embeddings')
  mi_embs <- suppressMessages(left_join(mi_embs,sigInfo %>% select(c('sig_id.x'='sig_id'),c('conditionId.x'='conditionId'))))
  mi_embs <- suppressMessages(left_join(mi_embs,sigInfo %>% select(c('sig_id.y'='sig_id'),c('conditionId.y'='conditionId'))))
  mi_embs <- mi_embs %>% mutate(relationship = ifelse(conditionId.x==conditionId.y,'same condition','different condition'))
  #Secondly for random normal
  mi_normal_list <- foreach(j = seq(1:nrow(random_normal))) %dopar% {
    apply(random_normal,1,JSDMI,y=as.matrix(random_normal[j,]))
  }
  mi_normal <- do.call(cbind,mi_normal_list)
  colnames(mi_normal) <- paste0('id_',seq(1:100))
  rownames(mi_normal) <- paste0('id_',seq(1:100))
  df_info <- data.frame(sig_id.y = paste0('id_',seq(1:100)),
                        conditionId.y=c(paste0('cid_',seq(1:85)),paste0('cid_',seq(1:15))))
  mi_normal <- as.data.frame(mi_normal) %>% mutate(conditionId.x= c(paste0('cid_',seq(1:85)),paste0('cid_',seq(1:15))))
  mi_normal <- mi_normal %>% rownames_to_column('sig_id.x') %>% gather('sig_id.y','MI',-sig_id.x,-conditionId.x) 
  mi_normal <- suppressMessages(left_join(mi_normal,df_info))
  mi_normal <- mi_normal %>% mutate(type='random normal')
  mi_normal <- mi_normal %>% mutate(relationship = ifelse(conditionId.x==conditionId.y,'same condition','different condition'))
  mi_normal <- mi_normal %>% dplyr::select(all_of(colnames(mi_embs)))
  #Last for random uniform
  mi_uniform_list <- foreach(j = seq(1:nrow(random_uniform))) %dopar% {
    apply(random_uniform,1,JSDMI,y=as.matrix(random_uniform[j,]))
  }
  mi_uniform <- do.call(cbind,mi_uniform_list)
  colnames(mi_uniform) <- paste0('id_',seq(1:100))
  rownames(mi_uniform) <- paste0('id_',seq(1:100))
  df_info <- data.frame(sig_id.y = paste0('id_',seq(1:100)),
                        conditionId.y=c(paste0('cid_',seq(1:85)),paste0('cid_',seq(1:15))))
  mi_uniform <- as.data.frame(mi_uniform) %>% mutate(conditionId.x= c(paste0('cid_',seq(1:85)),paste0('cid_',seq(1:15))))
  mi_uniform <- mi_uniform %>% rownames_to_column('sig_id.x') %>% gather('sig_id.y','MI',-sig_id.x,-conditionId.x) 
  mi_uniform <- suppressMessages(left_join(mi_uniform,df_info))
  mi_uniform <- mi_uniform %>% mutate(type='random uniform')
  mi_uniform <- mi_uniform %>% mutate(relationship = ifelse(conditionId.x==conditionId.y,'same condition','different condition'))
  mi_uniform <- mi_uniform %>% dplyr::select(all_of(colnames(mi_embs)))
  
  # combine all data
  mi_all <- rbind(mi_all,rbind(mi_embs,mi_normal,mi_uniform) %>% mutate(fold=i))
  print(paste0('Finished fold ',i))
  
}
mi_all <- mi_all %>% mutate(relationship=ifelse(relationship=='same condition','same','different'))
mi_all$relationship <- factor(mi_all$relationship,levels = c('different','same'))
p3 <- ggboxplot(mi_all %>% mutate(fold=paste0('fold ',fold)),x='relationship',y='MI',color='type') + 
  ylab('Empirical MI') + xlab('condition') +
  facet_wrap(~fold)+
  theme(text = element_text(family='Arial',size=20))
print(p3)
# saveRDS(mi_all,'../results/MI_results/mi_estimations/mi_all.rds')
ggsave('../article_supplementary_info/empirical_mi_random_vs_embs.png',
       plot = p3,
       height = 12,
       width = 16,
       units = 'in')
postscript('../article_supplementary_info/empirical_mi_random_vs_embs.eps',width = 16,height = 12)
print(p3)
dev.off()
p4 <- ggboxplot(mi_all %>% mutate(fold=paste0('fold ',fold)) %>% 
                  filter(type!='sampled embeddings') %>% 
                  mutate(type=ifelse(type=='random uniform','uniform','normal')),
                x='type',y='MI',color='relationship') + 
  ylab('Empirical MI') + xlab('random distribution') +
  facet_wrap(~fold)+
  theme(text = element_text(family='Arial',size=20))
print(p4)
ggsave('../article_supplementary_info/empirical_mi_uniform_vs_normal.png',
       plot = p4,
       height = 12,
       width = 16,
       units = 'in')
postscript('../article_supplementary_info/empirical_mi_uniform_vs_normal.eps',width = 16,height = 12)
print(p4)
dev.off()

### Compare U2OS MI of embs with beta = 1.0 and beta = 10000 for uniform distribution-----------------------------------
### First for the strong mse-like loss
mi_u2os <- data.frame()
for (i in 0:4){
  trainInfo <- data.table::fread(paste0('../preprocessing/preprocessed_data/SameCellimputationModel/U2OS/train_',i,'.csv'))
  valInfo <- data.table::fread(paste0('../preprocessing/preprocessed_data/SameCellimputationModel/U2OS/val_',i,'.csv'))
  
  
  # Load state loss embs
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_uniform_mselike/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_uniform_mselike/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  embs_test_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_uniform_mselike/val_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_test_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_uniform_mselike/val_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_test <- rbind(embs_test_1,embs_test_2)
  embs <- rbind(embs_train,embs_test)
  embs <- sample_frac(embs)
  embs <- sample_n(embs,500)
  # Calculate MI of these vectors
  mi_embs_list <- foreach(j = seq(1:nrow(embs))) %dopar% {
    apply(embs,1,JSDMI,y=as.matrix(embs[j,]))
  }
  mi_embs_state_unif <- do.call(cbind,mi_embs_list)
  colnames(mi_embs_state_unif) <- rownames(embs)
  mi_embs_state_unif <- as.data.frame(mi_embs_state_unif) %>% rownames_to_column('sig_id.x')
  mi_embs_state_unif <- mi_embs_state_unif %>% mutate(sig_id.x=strsplit(sig_id.x,'___')) %>% unnest(sig_id.x) %>% filter(!grepl('gene',sig_id.x))
  mi_embs_state_unif <- mi_embs_state_unif %>% gather('sig_id.y','MI',-sig_id.x) %>% mutate(type='state loss uniform')
  mi_embs_state_unif <- mi_embs_state_unif %>% mutate(sig_id.y=strsplit(sig_id.y,'___')) %>% unnest(sig_id.y) %>% filter(!grepl('gene',sig_id.y))
  mi_embs_state_unif <- suppressMessages(left_join(mi_embs_state_unif,sigInfo %>% select(c('sig_id.x'='sig_id'),c('conditionId.x'='conditionId'))))
  mi_embs_state_unif <- suppressMessages(left_join(mi_embs_state_unif,sigInfo %>% select(c('sig_id.y'='sig_id'),c('conditionId.y'='conditionId'))))
  mi_embs_state_unif <- mi_embs_state_unif %>% mutate(relationship = ifelse(conditionId.x==conditionId.y,'same condition','different condition'))
  print('Finished uniform state loss embs')
  ### Repeat for state loss normal
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_normal_mselike/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_normal_mselike/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  embs_test_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_normal_mselike/val_embs1_fold',
                                          i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_test_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_normal_mselike/val_embs2_fold',
                                          i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_test <- rbind(embs_test_1,embs_test_2)
  embs <- rbind(embs_train,embs_test)
  embs <- sample_frac(embs)
  embs <- sample_n(embs,500)
  # Calculate MI of these vectors
  mi_embs_list <- foreach(j = seq(1:nrow(embs))) %dopar% {
    apply(embs,1,JSDMI,y=as.matrix(embs[j,]))
  }
  mi_embs_state_normal <- do.call(cbind,mi_embs_list)
  colnames(mi_embs_state_normal) <- rownames(embs)
  mi_embs_state_normal <- as.data.frame(mi_embs_state_normal) %>% rownames_to_column('sig_id.x')
  mi_embs_state_normal <- mi_embs_state_normal %>% mutate(sig_id.x=strsplit(sig_id.x,'___')) %>% unnest(sig_id.x) %>% filter(!grepl('gene',sig_id.x))
  mi_embs_state_normal <- mi_embs_state_normal %>% gather('sig_id.y','MI',-sig_id.x) %>% mutate(type='state loss normal')
  mi_embs_state_normal <- mi_embs_state_normal %>% mutate(sig_id.y=strsplit(sig_id.y,'___')) %>% unnest(sig_id.y) %>% filter(!grepl('gene',sig_id.y))
  mi_embs_state_normal <- suppressMessages(left_join(mi_embs_state_normal,sigInfo %>% select(c('sig_id.x'='sig_id'),c('conditionId.x'='conditionId'))))
  mi_embs_state_normal <- suppressMessages(left_join(mi_embs_state_normal,sigInfo %>% select(c('sig_id.y'='sig_id'),c('conditionId.y'='conditionId'))))
  mi_embs_state_normal <- mi_embs_state_normal %>% mutate(relationship = ifelse(conditionId.x==conditionId.y,'same condition','different condition'))
  print('Finished normal state loss embs')
  
  ### Repeat for current
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_current/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_current/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  embs_test_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_current/val_embs1_fold',
                                          i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_test_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_current/val_embs2_fold',
                                          i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_test <- rbind(embs_test_1,embs_test_2)
  embs <- rbind(embs_train,embs_test)
  embs <- sample_frac(embs)
  embs <- sample_n(embs,500)
  # Calculate MI of these vectors
  mi_embs_list <- foreach(j = seq(1:nrow(embs))) %dopar% {
    apply(embs,1,JSDMI,y=as.matrix(embs[j,]))
  }
  mi_embs <- do.call(cbind,mi_embs_list)
  colnames(mi_embs) <- rownames(embs)
  mi_embs <- as.data.frame(mi_embs) %>% rownames_to_column('sig_id.x')
  mi_embs <- mi_embs %>% mutate(sig_id.x=strsplit(sig_id.x,'___')) %>% unnest(sig_id.x) %>% filter(!grepl('gene',sig_id.x))
  mi_embs <- mi_embs %>% gather('sig_id.y','MI',-sig_id.x) %>% mutate(type='current approach')
  mi_embs <- mi_embs %>% mutate(sig_id.y=strsplit(sig_id.y,'___')) %>% unnest(sig_id.y) %>% filter(!grepl('gene',sig_id.y))
  mi_embs <- suppressMessages(left_join(mi_embs,sigInfo %>% select(c('sig_id.x'='sig_id'),c('conditionId.x'='conditionId'))))
  mi_embs <- suppressMessages(left_join(mi_embs,sigInfo %>% select(c('sig_id.y'='sig_id'),c('conditionId.y'='conditionId'))))
  mi_embs <- mi_embs %>% mutate(relationship = ifelse(conditionId.x==conditionId.y,'same condition','different condition'))
  print('Finished current embs')
  
  # combine all data
  mi_u2os <- rbind(mi_u2os,rbind(mi_embs,mi_embs_state_normal,mi_embs_state_unif) %>% mutate(fold=i))
  
  
  print(paste0('Finished fold ',i))
  
}
mi_u2os <- mi_u2os %>% mutate(relationship=ifelse(relationship=='same condition','same','different'))
mi_u2os$relationship <- factor(mi_u2os$relationship,levels = c('different','same'))
mi_u2os <- mi_u2os %>% mutate(type = ifelse(type=='current approach',
                                            'current',
                                            ifelse(type=='state loss normal',
                                                   'normal',
                                                   'uniform')))

p4 <- ggboxplot(mi_u2os %>% mutate(fold=paste0('fold ',fold)),
                x='relationship',y='MI',color='type') + 
  ylab('Empirical MI') + xlab('relationship') +
  guides(fill=guide_legend(title="prior loss approach"))+
  facet_wrap(~fold)+
  theme(text = element_text(family='Arial',size=20))
print(p4)
# saveRDS(mi_u2os,'../results/MI_results/mi_estimations/mi_u2os.rds')
ggsave('../article_supplementary_info/empirical_mi_different_priors.png',
       plot = p4,
       height = 12,
       width = 16,
       units = 'in')
postscript('../article_supplementary_info/empirical_mi_different_priors.eps',width = 16,height = 12)
print(p4)
dev.off()

### Find latent variable with most uniform like shape-------------------------------------------------------
histograms_unif_state <- NULL
histograms_normal_state <- NULL
histograms_current <- NULL
histograms_normal_discr <- NULL
histograms_normal_KLD <- NULL
histograms_unif_discr <- NULL
bin_size <-  20
for (i in 0:4){
  trainInfo <- data.table::fread(paste0('../preprocessing/preprocessed_data/SameCellimputationModel/U2OS/train_',i,'.csv'))
  valInfo <- data.table::fread(paste0('../preprocessing/preprocessed_data/SameCellimputationModel/U2OS/val_',i,'.csv'))
  
  # Load state loss embs
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_uniform_mselike/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_uniform_mselike/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  ks_list_unif <- apply(embs_train,2,ks.test,runif(10000))
  ks_inf <- NULL
  for (j in 1:length(ks_list_unif)){
    ks_inf[j] <- ks_list_unif[[j]]$statistic
  }
  ind <- which.min(ks_inf)
  tmp <- as.data.frame(embs_train[,ind])
  colnames(tmp) <- 'z'
  histograms_unif_state[[i+1]] <- ggplot(tmp,aes(x=z)) + 
    geom_histogram(bins = bin_size,alpha = 0.8,fill = '#125b80',color='black')+
    ggtitle(paste0('fold ',i))+
    xlab(paste0('Latent variable z',ind-1))+
    ylab('counts')+
    theme_pubr(base_family = "Arial",base_size = 18) + 
    theme(plot.title = element_text(hjust = 0.5))
  # print('Finished uniform state loss embs')
  ### Repeat for state loss normal
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_normal_mselike/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_normal_mselike/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  ks_list_normal <- apply(embs_train,2,ks.test,rnorm(10000))
  ks_normal <- NULL
  for (j in 1:length(ks_list_normal)){
    ks_normal[j] <- ks_list_normal[[j]]$statistic
  }
  ind <- which.min(ks_normal)
  tmp <- as.data.frame(embs_train[,ind])
  colnames(tmp) <- 'z'
  histograms_normal_state[[i+1]] <- ggplot(tmp,aes(x=z)) + 
    geom_histogram(bins = bin_size,alpha = 0.8,fill = '#125b80',color='black')+
    ggtitle(paste0('fold ',i))+
    xlab(paste0('Latent variable z',ind-1))+
    ylab('counts')+
    theme_pubr(base_family = "Arial",base_size = 18) + 
    theme(plot.title = element_text(hjust = 0.5))
  # print('Finished normal state loss embs')
  
  ### Repeat for current
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_current/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_current/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  ks_list_current <- apply(embs_train,2,ks.test,rnorm(10000))
  ks_current <- NULL
  for (j in 1:length(ks_list_current)){
    ks_current[j] <- ks_list_current[[j]]$statistic
  }
  ind <- which.min(ks_current)
  tmp <- as.data.frame(embs_train[,ind])
  colnames(tmp) <- 'z'
  histograms_current[[i+1]] <- ggplot(tmp,aes(x=z)) + 
    geom_histogram(bins = bin_size,alpha = 0.8,fill = '#125b80',color='black')+
    ggtitle(paste0('fold ',i))+
    xlab(paste0('Latent variable z',ind-1))+
    ylab('counts')+
    theme_pubr(base_family = "Arial",base_size = 18) + 
    theme(plot.title = element_text(hjust = 0.5))
  # print('Finished current embs')
  
  ### Repeat for normal KLD
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_KLD_normal/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_KLD_normal/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  ks_list_kld <- apply(embs_train,2,ks.test,rnorm(10000))
  ks_kld <- NULL
  for (j in 1:length(ks_list_kld)){
    ks_kld[j] <- ks_list_kld[[j]]$statistic
  }
  ind <- which.min(ks_kld)
  tmp <- as.data.frame(embs_train[,ind])
  colnames(tmp) <- 'z'
  histograms_normal_KLD[[i+1]] <- ggplot(tmp,aes(x=z)) + 
    geom_histogram(bins = bin_size,alpha = 0.8,fill = '#125b80',color='black')+
    ggtitle(paste0('fold ',i))+
    xlab(paste0('Latent variable z',ind-1))+
    ylab('counts')+
    theme_pubr(base_family = "Arial",base_size = 18) + 
    theme(plot.title = element_text(hjust = 0.5))
  
  ### Repeat for normal discriminator
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_discr_normal/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_discr_normal/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  ks_list_discr_normal <- apply(embs_train,2,ks.test,rnorm(10000))
  ks_discr_normal <- NULL
  for (j in 1:length(ks_list_discr_normal)){
    ks_discr_normal[j] <- ks_list_discr_normal[[j]]$statistic
  }
  ind <- which.min(ks_discr_normal)
  tmp <- as.data.frame(embs_train[,ind])
  colnames(tmp) <- 'z'
  histograms_normal_discr[[i+1]] <- ggplot(tmp,aes(x=z)) + 
    geom_histogram(bins = bin_size,alpha = 0.8,fill = '#125b80',color='black')+
    ggtitle(paste0('fold ',i))+
    xlab(paste0('Latent variable z',ind-1))+
    ylab('counts')+
    theme_pubr(base_family = "Arial",base_size = 18) + 
    theme(plot.title = element_text(hjust = 0.5))
  # print('Finished current embs')
  
  ## Repeat for uniform discriminator
  embs_train_1 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_discr_uniform/train_embs1_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes1')) %>%
    column_to_rownames('sig_id')
  embs_train_2 <- data.table::fread(paste0('../results/PriorLossAnalysis/embs_AutoTransOp/U2OS_discr_uniform/train_embs2_fold',
                                           i,'_beta10000.0.csv'),header = T) %>% 
    mutate(sig_id = paste0(sig_id,'___genes2')) %>%
    column_to_rownames('sig_id')
  embs_train <- rbind(embs_train_1,embs_train_2)
  ks_list_discr_uniform <- apply(embs_train,2,ks.test,runif(10000))
  ks_discr_uniform <- NULL
  for (j in 1:length(ks_list_discr_uniform)){
    ks_discr_uniform[j] <- ks_list_discr_uniform[[j]]$statistic
  }
  ind <- which.min(ks_discr_uniform)
  tmp <- as.data.frame(embs_train[,ind])
  colnames(tmp) <- 'z'
  histograms_unif_discr[[i+1]] <- ggplot(tmp,aes(x=z)) + 
    geom_histogram(bins = bin_size,alpha = 0.8,fill = '#125b80',color='black')+
    ggtitle(paste0('fold ',i))+
    xlab(paste0('Latent variable z',ind-1))+
    ylab('counts')+
    theme_pubr(base_family = "Arial",base_size = 18) + 
    theme(plot.title = element_text(hjust = 0.5))

  print(paste0('Finished fold ',i))
  
}


### Save uniform disciminator
histograms <- ggarrange(plotlist = histograms_unif_discr)
ggsave('../article_supplementary_info/prior_analysis/uniform_disciminator_embs_beta10000_hist_allfolds.eps',
       plot = histograms,
       device = cairo_ps,
       height = 9,
       width=12,
       units = 'in',
       dpi=600)
for (j in 1:length(histograms_unif_discr)){
  histogram <- histograms_unif_discr[[j]]
  ggsave(paste0('../article_supplementary_info/prior_analysis/uniform_disciminator_embs_beta10000_hist_fold',j-1,'.eps'),
         plot = histogram,
         device = cairo_ps,
         height = 9,
         width=9,
         units = 'in',
         dpi=600)
}


### Save normal disciminator
histograms <- ggarrange(plotlist = histograms_normal_discr)
ggsave('../article_supplementary_info/prior_analysis/normal_disciminator_embs_beta10000_hist_allfolds.eps',
       plot = histograms,
       device = cairo_ps,
       height = 9,
       width=12,
       units = 'in',
       dpi=600)
for (j in 1:length(histograms_normal_discr)){
  histogram <- histograms_normal_discr[[j]]
  ggsave(paste0('../article_supplementary_info/prior_analysis/normal_disciminator_embs_beta10000_hist_fold',j-1,'.eps'),
         plot = histogram,
         device = cairo_ps,
         height = 9,
         width=9,
         units = 'in',
         dpi=600)
}

### Save uniform state
histograms <- ggarrange(plotlist = histograms_unif_state)
ggsave('../article_supplementary_info/prior_analysis/uniform_stateLoss_embs_beta10000_hist_allfolds.eps',
       plot = histograms,
       device = cairo_ps,
       height = 9,
       width=12,
       units = 'in',
       dpi=600)
for (j in 1:length(histograms_unif_state)){
  histogram <- histograms_unif_state[[j]]
  ggsave(paste0('../article_supplementary_info/prior_analysis/uniform_stateLoss_embs_beta10000_hist_fold',j-1,'.eps'),
         plot = histogram,
         device = cairo_ps,
         height = 9,
         width=9,
         units = 'in',
         dpi=600)
}

### Save normal state
histograms <- ggarrange(plotlist = histograms_normal_state)
ggsave('../article_supplementary_info/prior_analysis/normal_stateLoss_embs_beta10000_hist_allfolds.eps',
       plot = histograms,
       device = cairo_ps,
       height = 9,
       width=12,
       units = 'in',
       dpi=600)
for (j in 1:length(histograms_normal_state)){
  histogram <- histograms_normal_state[[j]]
  ggsave(paste0('../article_supplementary_info/prior_analysis/normal_stateLoss_embs_beta10000_hist_fold',j-1,'.eps'),
         plot = histogram,
         device = cairo_ps,
         height = 9,
         width=9,
         units = 'in',
         dpi=600)
}

### Save current embs
histograms <- ggarrange(plotlist = histograms_current)
ggsave('../article_supplementary_info/prior_analysis/current_embs_beta10000_hist_allfolds.eps',
       plot = histograms,
       device = cairo_ps,
       height = 9,
       width=12,
       units = 'in',
       dpi=600)
for (j in 1:length(histograms_current)){
  histogram <- histograms_current[[j]]
  ggsave(paste0('../article_supplementary_info/prior_analysis/current_embs_beta10000_hist_fold',j-1,'.eps'),
         plot = histogram,
         device = cairo_ps,
         height = 9,
         width=9,
         units = 'in',
         dpi=600)
}

### Save KLD embs
histograms <- ggarrange(plotlist = histograms_normal_KLD)
ggsave('../article_supplementary_info/prior_analysis/KLD_embs_beta10000_hist_allfolds.eps',
       plot = histograms,
       device = cairo_ps,
       height = 9,
       width=12,
       units = 'in',
       dpi=600)
for (j in 1:length(histograms_normal_KLD)){
  histogram <- histograms_normal_KLD[[j]]
  ggsave(paste0('../article_supplementary_info/prior_analysis/kld_normal_embs_beta10000_hist_fold',j-1,'.eps'),
         plot = histogram,
         device = cairo_ps,
         height = 9,
         width=9,
         units = 'in',
         dpi=600)
}
