library(tidyverse)
library(ggplot2)
library(ggpubr)
library(infotheo)
library(doFuture)
# parallel: set number of workers
cores <- 16
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

### Start analysis to figure out how many bins you need-------
df_empirical_dmi_vector <- NULL
dmi_modeld_vector <- NULL
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
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev2_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev2_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev2_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev2_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  
  # Estimate MI from embeddings
  if (i<10){
    print(paste0('Calculate empirical MI in fold : ',i))
    mi_empirical <- readRDS(paste0('../results/MI_results/mi_estimations/CPA/empirical_mi_a375_ht29_fold',i,'.rds'))
  }else{
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
    saveRDS(mi_empirical,paste0('../results/MI_results/mi_estimations/CPA/empirical_mi_a375_ht29_fold',i,'.rds'))
  }
  dmi_empirical <- mi_empirical %>% group_by(relationship) %>% mutate(meanMI=mean(MI)) %>% ungroup()
  # pos_mi <- dmi_empirical %>% filter(relationship=='same condition') %>% select(meanMI) %>% unique()
  # pos_mi <- pos_mi$meanMI
  # neg_mi <- dmi_empirical %>% filter(relationship!='same condition') %>% select(meanMI) %>% unique()
  # neg_mi <- neg_mi$meanMI
  pos_mi <- dmi_empirical %>% filter(relationship=='same condition') %>% select(MI) %>% unique()
  pos_mi <- pos_mi$MI
  neg_mi <- dmi_empirical %>% filter(relationship!='same condition') %>% select(MI) %>% unique()
  neg_mi <- neg_mi$MI
  neg_mi <- sample(neg_mi,length(pos_mi),replace = F)
  tmp <- sapply(pos_mi,'-',neg_mi)
  df_empirical_dmi_vector <- c(df_empirical_dmi_vector,c(tmp))
  # ggboxplot(mi_empirical,x='relationship',y='MI')
  
  # ### read mi modeled
  # tmp <- data.table::fread(paste0('../results/MI_results/mi_estimations/CPA/DeltaMI_ht29_a375_fold',i-1,'.csv'),header = T) %>% select(-V1)
  # tmp <- as.matrix(tmp)
  # tmp <- c(tmp)
  # tmp <- tmp[which(tmp!=0)]
  # dmi_modeld_vector <- c(dmi_modeld_vector,tmp)
  
}
df_empirical_dmi <- as.data.frame(df_empirical_dmi_vector)
colnames(df_empirical_dmi)[1] <- 'DMI'
df_empirical_dmi <- df_empirical_dmi %>% mutate(fold = seq(0,9))
df_empirical_dmi <- df_empirical_dmi %>% mutate(type='empirical')
saveRDS(df_empirical_dmi,'../results/MI_results/mi_estimations/CPA/df_empirical_dmi_ht29_a375.rds')
dmi_modeld <- data.table::fread('../results/MI_results/mi_estimations/CPA/DeltaMI_ht29_a375.csv',header = T) %>% select(-V1)
dmi_modeld <- dmi_modeld %>% mutate(type='modeled')
df_dmi <- rbind(dmi_modeld,df_empirical_dmi)
ggdotplot(df_dmi,x='type',y='DMI',fill='type') + 
  theme(legend.position = 'none')

### Compare empirically and modeled MI------------------------
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
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev2_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev2_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev2_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev2_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_proc_train <- process_embeddings(embs_train,sigInfo,trainInfo)
  embs_proc_test <- process_embeddings(embs_test,sigInfo,valInfo)
  
  # load MI estimation
  # mi_positives <- data.table::fread(paste0('../results/MI_results/mi_estimations/CPA/estimated_mi_positives_ht29_a375_fold',i,'.csv'),header = T)
  # mi_positives <- distinct(mi_positives)
  # mi_positives <- mi_positives %>% column_to_rownames('V1')
  # mi_positives <- mi_positives %>% select(all_of(rownames(mi_positives)))
  # mi_positives <- mi_positives %>% rownames_to_column('sig_id.x') %>% gather('sig_id.y','MI',-sig_id.x) %>% mutate(type='modeled')
  # mi_positives <- left_join(mi_positives,sigInfo %>% select(c('sig_id.x'='sig_id'),c('conditionId.x'='conditionId')))
  # mi_positives <- left_join(mi_positives,sigInfo %>% select(c('sig_id.y'='sig_id'),c('conditionId.y'='conditionId')))
  # mi_positives <- mi_positives %>% filter(MI!=0)
  # mi_negatives <- data.table::fread(paste0('../results/MI_results/mi_estimations/CPA/estimated_mi_negatives_ht29_a375_fold',i,'.csv'),header = T)
  # mi_negatives <- distinct(mi_negatives)
  # mi_negatives <- mi_negatives %>% column_to_rownames('V1')
  # mi_negatives <- mi_negatives %>% select(all_of(rownames(mi_negatives)))
  # mi_negatives <- mi_negatives %>% rownames_to_column('sig_id.x') %>% gather('sig_id.y','MI',-sig_id.x) %>% mutate(type='modeled')
  # mi_negatives <- left_join(mi_negatives,sigInfo %>% select(c('sig_id.x'='sig_id'),c('conditionId.x'='conditionId')))
  # mi_negatives <- left_join(mi_negatives,sigInfo %>% select(c('sig_id.y'='sig_id'),c('conditionId.y'='conditionId')))
  # mi_negatives <- mi_negatives %>% filter(MI!=0)
  # mi_modeled <- rbind(mi_positives,mi_negatives)
  mi_modeled <- data.table::fread(paste0('../results/MI_results/mi_estimations/CPA/estimated_mi_ht29_a375_fold',i,'.csv'),header = T)
  mi_modeled <- distinct(mi_modeled)
  mi_modeled <- mi_modeled %>% column_to_rownames('V1')
  mi_modeled <- mi_modeled %>% select(all_of(rownames(mi_modeled)))
  mi_modeled <- mi_modeled %>% rownames_to_column('sig_id.x') %>% gather('sig_id.y','MI',-sig_id.x) %>% mutate(type='modeled')
  mi_modeled <- left_join(mi_modeled,sigInfo %>% select(c('sig_id.x'='sig_id'),c('conditionId.x'='conditionId')))
  mi_modeled <- left_join(mi_modeled,sigInfo %>% select(c('sig_id.y'='sig_id'),c('conditionId.y'='conditionId')))
  mi_modeled <- mi_modeled %>% mutate(relationship = ifelse(conditionId.x==conditionId.y,'same condition','different condition'))
  
  
  ggboxplot(rbind(mi_modeled,mi_empirical) %>% filter(relationship=='same condition'),x='type',y='MI',
            color = 'type',add='jitter')+
    theme(legend.position = 'none')
  
  
  # Estimate MI from embeddings
  embs_proc_all <- rbind(embs_proc_train,embs_proc_test)
  embs_proc_all <- embs_proc_all[,1:293]
  mi_empirical <- mutinformation(embs_proc_all[,2:293],embs_proc_all[,2:293],method='emp')
  
}
