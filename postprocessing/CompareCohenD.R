library(tidyverse)
library(gg3D)
#Process function to add condition ids and duplicate ids
process_embeddings <- function(embbedings,dataInfo,sampleInfo){
  dataInfo <- dataInfo %>% select(sig_id,cmap_name,duplIdentifier) %>% unique()
  
  sampleInfo <- left_join(sampleInfo,dataInfo)
  
  embbedings <- embbedings %>% rownames_to_column('sig_id')
  
  embs_processed <- left_join(embbedings,sampleInfo)
  
  return(embs_processed)
}

samples_separation <- function(processed_embbedings,save_name,
                               compare_level=c('duplicates',
                                               'equivalent condition',
                                               'cell',
                                               'drug',
                                               'cell-drug'),
                               metric=c("euclidean", "maximum", "manhattan",
                                        "canberra", "binary","cosine"),
                               show_plot=TRUE){
  library(tidyverse)
  embs <- processed_embbedings %>% column_to_rownames('sig_id') %>%
    select(-conditionId,-duplIdentifier,-cell_iname,-cmap_name)
  sample_info <- processed_embbedings %>% select(sig_id,conditionId,duplIdentifier,cell_iname,cmap_name)
  
  
  # calculate distance matrix
  if (metric=='cosine'){
    library(lsa)
    mat <- t(embs)
    dist <- 1 - cosine(mat)
  } else{
    dist <- as.matrix(dist(embs, method = metric))
  }
  
  # Conver to long format data frame
  # Keep only unique (non-self) pairs
  dist[lower.tri(dist,diag = T)] <- NA
  dist <- reshape2::melt(dist)
  dist <- dist %>% filter(!is.na(value))
  
  # Merge meta-data info and distances values
  dist <- left_join(dist,sample_info,by = c("Var1"="sig_id"))
  dist <- left_join(dist,sample_info,by = c("Var2"="sig_id"))
  dist <- dist %>% filter(!is.na(value))
  
  if (compare_level=='duplicates'){
    dist <- dist %>% mutate(is_same = (duplIdentifier.x==duplIdentifier.y))
    label <- 'Duplicate Signatures'
  }else if (compare_level=='equivalent condition'){
    dist <- dist %>% mutate(is_same = (conditionId.x==conditionId.y))
    label <- 'Same condition in different cell-line'
  }else if (compare_level=='cell'){
    dist <- dist %>% mutate(is_same = (cell_iname.x==cell_iname.y))
    label <- 'Same cell-line'
  }else if (compare_level=='drug'){
    dist <- dist %>% mutate(is_same = (cmap_name.x==cmap_name.y))
    label <- 'Same drug'
  } else if (compare_level=='cell-drug'){
    dist <- dist %>% mutate(is_same = (paste0(cmap_name.x,cell_iname.x)==paste0(cmap_name.y,cell_iname.y)))
    label <- 'Same drug,same cell-line'
  }
  
  dist <-dist %>% mutate(is_same=ifelse(is_same==T,
                                        label,'Random Signatures')) %>%
    mutate(is_same = factor(is_same,
                            levels = c('Random Signatures',
                                       label)))
  p <- ggplot(dist,aes(x=value,color=is_same,fill=is_same)) +
    geom_density(alpha=0.2) +
    labs(col = 'Type',fill='Type',title="Distance distribution in the latent space",x=paste0(metric,' distance'), y = "Density")+
    theme_classic() + theme(text = element_text(size=10),plot.title = element_text(hjust = 0.5))
  # png(paste0(save_name,'_',compare_level,'_seperation_latent_space.png'),width=12,height=8,units = "in",res=600)
  # print(p)
  # dev.off()
  
  if (show_plot==T){
    print(p)
  } 
  return(dist)
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

plotList <- NULL
distrList <- NULL
df_effsize <- data.frame()
#df_effsize_train <- data.frame()
for (i in 0:9){
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
  embs_train_cov <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev3_',i,'_a375.csv'),header = T),
                         data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev3_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test_cov <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev3_',i,'_a375.csv'),header = T),
                        data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev3_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  
  embs_train_2IntEncs2Dec <- rbind(data.table::fread(paste0('../results/MI_results/embs/IntermediateTranslationLandmarks2Dec_1000ep/train/trainEmbs_base_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/IntermediateTranslationLandmarks2Dec_1000ep/train/trainEmbs_base_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test_2IntEncs2Dec <- rbind(data.table::fread(paste0('../results/MI_results/embs/IntermediateTranslationLandmarks2Dec_1000ep/validation/valEmbs_base_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/IntermediateTranslationLandmarks2Dec_1000ep/validation/valEmbs_base_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_train_2IntEncs1Dec <- rbind(data.table::fread(paste0('../results/MI_results/embs/IntermediateTranslationLandmarks_1000ep/train/trainEmbs_base_',i,'_a375.csv'),header = T),
                                   data.table::fread(paste0('../results/MI_results/embs/IntermediateTranslationLandmarks_1000ep/train/trainEmbs_base_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test_2IntEncs1Dec <- rbind(data.table::fread(paste0('../results/MI_results/embs/IntermediateTranslationLandmarks_1000ep/validation/valEmbs_base_',i,'_a375.csv'),header = T),
                                  data.table::fread(paste0('../results/MI_results/embs/IntermediateTranslationLandmarks_1000ep/validation/valEmbs_base_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  
  embs_proc_train_cov <- process_embeddings(embs_train_cov,sigInfo,trainInfo)
  embs_proc_test_cov <- process_embeddings(embs_test_cov,sigInfo,valInfo)
  
  embs_proc_train_2IntEncs2Dec <- process_embeddings(embs_train_2IntEncs2Dec,sigInfo,trainInfo)
  embs_proc_test_2IntEncs2Dec <- process_embeddings(embs_test_2IntEncs2Dec,sigInfo,valInfo)
  
  embs_proc_train_2IntEncs1Dec <- process_embeddings(embs_train_2IntEncs1Dec,sigInfo,trainInfo)
  embs_proc_test_2IntEncs1Dec <- process_embeddings(embs_test_2IntEncs1Dec,sigInfo,valInfo)
  
  # Check distributions in the latent space----
  dist_train_cov <- samples_separation(embs_proc_train_cov,
                                      compare_level='cell',
                                      metric = 'cosine',
                                      show_plot = F)
  dist_train_cov <- dist_train_cov %>% mutate(model='CPA-based model')
  dist_train_cov <- dist_train_cov %>% mutate(set='Train')
  dist_test_cov <- samples_separation(embs_proc_test_cov,
                                     compare_level='cell',
                                     metric = 'cosine',
                                     show_plot = F)
  dist_test_cov <- dist_test_cov %>% mutate(model='CPA-based model')
  dist_test_cov <- dist_test_cov %>% mutate(set='Validation')
  
  dist_train_2IntEncs2Dec <- samples_separation(embs_proc_train_2IntEncs2Dec,
                                       compare_level='cell',
                                       metric = 'cosine',
                                       show_plot = F)
  dist_train_2IntEncs2Dec <- dist_train_2IntEncs2Dec %>% mutate(model='Intermediate encoders with 2 decoders')
  dist_train_2IntEncs2Dec <- dist_train_2IntEncs2Dec %>% mutate(set='Train')
  dist_test_2IntEncs2Dec <- samples_separation(embs_proc_test_2IntEncs2Dec,
                                      compare_level='cell',
                                      metric = 'cosine',
                                      show_plot = F)
  dist_test_2IntEncs2Dec <- dist_test_2IntEncs2Dec %>% mutate(model='Intermediate encoders with 2 decoders')
  dist_test_2IntEncs2Dec <- dist_test_2IntEncs2Dec %>% mutate(set='Validation')
  
  dist_train_2IntEncs1Dec <- samples_separation(embs_proc_train_2IntEncs1Dec,
                                       compare_level='cell',
                                       metric = 'cosine',
                                       show_plot = F)
  dist_train_2IntEncs1Dec <- dist_train_2IntEncs1Dec %>% mutate(model='Intermediate encoders with 1 decoder')
  dist_train_2IntEncs1Dec <- dist_train_2IntEncs1Dec %>% mutate(set='Train')
  dist_test_2IntEncs1Dec <- samples_separation(embs_proc_test_2IntEncs1Dec,
                                      compare_level='cell',
                                      metric = 'cosine',
                                      show_plot = F)
  dist_test_2IntEncs1Dec <- dist_test_2IntEncs1Dec %>% mutate(model='Intermediate encoders with 1 decoder')
  dist_test_2IntEncs1Dec <- dist_test_2IntEncs1Dec %>% mutate(set='Validation')
  
  all_dists <- bind_rows(dist_train_cov,dist_test_cov,
                         dist_train_2IntEncs2Dec,dist_test_2IntEncs2Dec,
                         dist_train_2IntEncs1Dec,dist_test_2IntEncs1Dec)
  
  d1_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set!='Train') %>% 
                                            filter(is_same=='Same cell-line') %>% select(value)),
                                as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(is_same!='Same cell-line')%>% select(value)),
                                ci=0.95)
  d2_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='Intermediate encoders with 2 decoders')%>% filter(set!='Train') %>%
                                            filter(is_same=='Same cell-line') %>% select(value)),
                                as.matrix(all_dists %>% filter(model=='Intermediate encoders with 2 decoders') %>% filter(set!='Train')
                                          %>% filter(is_same!='Same cell-line')%>% select(value)),
                                ci=0.95)
  d3_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='Intermediate encoders with 1 decoder')%>% filter(set!='Train') %>%
                                            filter(is_same=='Same cell-line') %>% select(value)),
                                as.matrix(all_dists %>% filter(model=='Intermediate encoders with 1 decoder')%>% filter(set!='Train')
                                          %>% filter(is_same!='Same cell-line')%>% select(value)),
                                ci=0.95)
  d1_train = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set=='Train') %>% 
                                            filter(is_same=='Same cell-line') %>% select(value)),
                                as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set=='Train')%>% 
                                            filter(is_same!='Same cell-line')%>% select(value)),
                                ci=0.95)
  d2_train = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='Intermediate encoders with 2 decoders')%>% filter(set=='Train') %>%
                                            filter(is_same=='Same cell-line') %>% select(value)),
                                as.matrix(all_dists %>% filter(model=='Intermediate encoders with 2 decoders') %>% filter(set=='Train')
                                          %>% filter(is_same!='Same cell-line')%>% select(value)),
                                ci=0.95)
  d3_train = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='Intermediate encoders with 1 decoder')%>% filter(set=='Train') %>%
                                            filter(is_same=='Same cell-line') %>% select(value)),
                                as.matrix(all_dists %>% filter(model=='Intermediate encoders with 1 decoder')%>% filter(set=='Train')
                                          %>% filter(is_same!='Same cell-line')%>% select(value)),
                                ci=0.95)
  
  all_dists_val <- all_dists %>% filter(set!='Train') %>% 
    mutate(effsize = ifelse(model=='CPA-based model',abs(d1_val$Cohens_d),
                            ifelse(model=='Intermediate encoders with 2 decoders',
                                   abs(d2_val$Cohens_d),abs(d3_val$Cohens_d))))
  all_dists_train <- all_dists %>% filter(set=='Train') %>% 
    mutate(effsize = ifelse(model=='CPA-based model',abs(d1_train$Cohens_d),
                            ifelse(model=='Intermediate encoders with 2 decoders',
                                   abs(d2_train$Cohens_d),abs(d3_train$Cohens_d))))
  all_dists <- rbind(all_dists_train,all_dists_val)
  all_dists  <- all_dists %>% mutate(effsize = paste0('Cohen`s d: ',round(effsize,3)))
  cohen_df <- distinct(all_dists %>% select(model,effsize,set))
  df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
                        filter(effsize!='Cohen`s d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% select(model,split,set,'Cohen`s d'))
  
  violin_separation <- ggplot(all_dists, aes(x=model, y=value, fill = is_same)) + 
    geom_violin(position = position_dodge(width = 1),width = 1)+geom_boxplot(position = position_dodge(width = 1),width = 0.05,
                                                                             outlier.shape = NA)+
    scale_fill_discrete(name="Embedding distance distribution in the composed latent space",
                        labels=c("Random Signatures","Same cell-line"))+
    ylim(0,2)+
    xlab("")+ylab("Cosine Distance")+ 
    theme(axis.ticks.x=element_blank(),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),text = element_text(family = "Arial",size = 20),legend.position = "bottom")+
    theme_minimal(base_family = "Arial",base_size = 20) +
    geom_text(aes(x=model,y=max(all_dists  %>% select(value))+0.1, label=effsize),
              data=cohen_df ,inherit.aes = FALSE)+
  facet_wrap(~ set)
  #stat_compare_means(aes(group=is_same), method = "t.test")
  violin_separation <- violin_separation + theme(legend.position = "bottom")
  plotList[[i+1]] <- violin_separation
  
  message(paste0('Done split ',i))
}

library(ggpubr)
png(file="../figures/MI_results/comparison_of_different_basalLatent_separation_approaches_lands.png",width=16,height=16,units = "in",res=600)
p <- ggarrange(plotlist=plotList,ncol=2,nrow=5,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',10),seq(1,10)),
               font.label = list(size = 10, color = "black", face = "plain", family = NULL),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 14))
dev.off()

colnames(df_effsize)[4] <- 'value'
my_comparisons <- list(c("CPA-based model","Intermediate encoders with 2 decoders"),
                       c("CPA-based model","Intermediate encoders with 1 decoder"),
                       c("Intermediate encoders with 2 decoders","Intermediate encoders with 1 decoder"))
p  <- ggplot(df_effsize,aes(x=model,y=value,fill=model)) + 
  geom_boxplot(position = position_dodge(width = 1),width = 0.5) +
  ylab('Cohen`s d')+
  stat_compare_means(comparisons = my_comparisons,label.y = c(4.5, 4, 2)) + #,label = "p.signif"
  theme(panel.background = element_rect(fill = "white",
                                        colour = "white",
                                        size = 0.5, linetype = "solid"),
        panel.grid.major = element_line(size = 1, linetype = 'solid',
                                        colour = "#EEEDEF"), 
        panel.grid.minor = element_line(size = 1, linetype = 'solid',
                                        colour = "#EEEDEF"),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        text = element_text(family = "serif",size = 20),legend.position = "bottom")+
  facet_wrap(~ set)
print(p)

png(file="../figures/MI_results/comparison_of_different_cohen_basalLatent_separation_approaches_lands.png",width=12,height=8,units = "in",res=600)
print(p)
dev.off()
