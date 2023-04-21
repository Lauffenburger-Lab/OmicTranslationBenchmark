library(tidyverse)
library(gg3D)
library(ggpubr)
#Process function to add condition ids and duplicate ids
process_embeddings <- function(embbedings,dataInfo,sampleInfo){
  dataInfo <- dataInfo %>% select(sig_id,cmap_name,duplIdentifier) %>% unique()
  
  sampleInfo <- suppressMessages(left_join(sampleInfo,dataInfo))
  
  embbedings <- embbedings %>% rownames_to_column('sig_id')
  
  embs_processed <- suppressMessages(left_join(embbedings,sampleInfo))
  
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
    dist <- 1 - suppressMessages(cosine(mat))
  } else{
    dist <- as.matrix(dist(embs, method = metric))
  }
  
  # Conver to long format data frame
  # Keep only unique (non-self) pairs
  dist[lower.tri(dist,diag = T)] <- NA
  dist <- reshape2::melt(dist)
  dist <- dist %>% filter(!is.na(value))
  
  # Merge meta-data info and distances values
  dist <- suppressMessages(left_join(dist,sample_info,by = c("Var1"="sig_id")))
  dist <- suppressMessages(left_join(dist,sample_info,by = c("Var2"="sig_id")))
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

### CPA analysis--------------------------------------------------------

plotList <- NULL
distrList <- NULL
df_effsize <- data.frame()
#df_effsize_train <- data.frame()
for (i in 0:9){
#for (i in c(2,7)){
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
  
  # Load basal embeddings of pre-trained
  embs_train_basal <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev2_',i,'_a375.csv'),header = T),
                          data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbs_basev2_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test_basal <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev2_',i,'_a375.csv'),header = T),
                         data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbs_basev2_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  
  embs_proc_train_basal <- process_embeddings(embs_train_basal,sigInfo,trainInfo)
  embs_proc_test_basal <- process_embeddings(embs_test_basal,sigInfo,valInfo)
  
  # Load embeddings of pre-trained
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbsv2_',i,'_a375.csv'),header = T),
                         data.table::fread(paste0('../results/MI_results/embs/CPA_approach/train/trainEmbsv2_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbsv2_',i,'_a375.csv'),header = T),
                        data.table::fread(paste0('../results/MI_results/embs/CPA_approach/validation/valEmbsv2_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_proc_train <- process_embeddings(embs_train,sigInfo,trainInfo)
  embs_proc_test <- process_embeddings(embs_test,sigInfo,valInfo)
  
  
  # Check distributions in the latent space----
  dist_train <- samples_separation(embs_proc_train,
                                       compare_level='cell',
                                       metric = 'cosine',
                                       show_plot = F)
  dist_train <- dist_train %>% mutate(model='CPA-based model')
  dist_train <- dist_train %>% mutate(set='Train')
  dist_train <- dist_train %>% mutate(space='latent')
  dist_train_basal <- samples_separation(embs_proc_train_basal,
                                   compare_level='cell',
                                   metric = 'cosine',
                                   show_plot = F)
  dist_train_basal <- dist_train_basal %>% mutate(model='CPA-based model')
  dist_train_basal <- dist_train_basal %>% mutate(set='Train')
  dist_train_basal <- dist_train_basal %>% mutate(space='basal latent')
  
  dist_test <- samples_separation(embs_proc_test,
                                      compare_level='cell',
                                      metric = 'cosine',
                                      show_plot = F)
  dist_test <- dist_test %>% mutate(model='CPA-based model')
  dist_test <- dist_test %>% mutate(set='Validation')
  dist_test <- dist_test %>% mutate(space='latent')
  dist_test_basal <- samples_separation(embs_proc_test_basal,
                                         compare_level='cell',
                                         metric = 'cosine',
                                         show_plot = F)
  dist_test_basal <- dist_test_basal %>% mutate(model='CPA-based model')
  dist_test_basal <- dist_test_basal %>% mutate(set='Validation')
  dist_test_basal <- dist_test_basal %>% mutate(space='basal latent')
  
  all_dists <- bind_rows(dist_train,dist_train_basal,dist_test,dist_test_basal)
  
  d_val_basal = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set!='Train') %>% 
                                            filter(space!='latent') %>% filter(is_same=='Same cell-line') %>% select(value)),
                                as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set!='Train') %>% 
                                            filter(space!='latent') %>%filter(is_same!='Same cell-line')%>% select(value)),
                                ci=0.95)
  d_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set!='Train') %>% 
                                                 filter(space=='latent') %>% filter(is_same=='Same cell-line') %>% select(value)),
                                     as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set!='Train') %>% 
                                                 filter(space=='latent') %>%filter(is_same!='Same cell-line')%>% select(value)),
                                     ci=0.95)
  d_train_basal = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set=='Train') %>% 
                                             filter(space!='latent') %>%filter(is_same=='Same cell-line') %>% select(value)),
                                  as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set=='Train')%>% 
                                              filter(space!='latent') %>%filter(is_same!='Same cell-line')%>% select(value)),
                                  ci=0.95)
  d_train = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set=='Train') %>% 
                                                   filter(space=='latent') %>%filter(is_same=='Same cell-line') %>% select(value)),
                                       as.matrix(all_dists %>% filter(model=='CPA-based model') %>% filter(set=='Train')%>% 
                                                   filter(space=='latent') %>%filter(is_same!='Same cell-line')%>% select(value)),
                                       ci=0.95)
  
  all_dists_val_basal <- all_dists %>% filter(set!='Train') %>%  filter(space!='latent') %>%
    mutate(effsize = abs(d_val_basal$Cohens_d))
  all_dists_val <- all_dists %>% filter(set!='Train') %>%  filter(space=='latent') %>%
    mutate(effsize = abs(d_val$Cohens_d))
  all_dists_train_basal <- all_dists %>% filter(set=='Train') %>%  filter(space!='latent') %>%
    mutate(effsize = abs(d_train_basal$Cohens_d))
  all_dists_train <- all_dists %>% filter(set=='Train') %>%  filter(space=='latent') %>%
    mutate(effsize = abs(d_train$Cohens_d))
  all_dists <- rbind(all_dists_train,all_dists_train_basal,all_dists_val,all_dists_val_basal)
  all_dists  <- all_dists %>% mutate(effsize = paste0('Cohen`s d: ',round(effsize,3)))
  cohen_df <- distinct(all_dists %>% select(model,effsize,set,space))
  df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
                        filter(effsize!='Cohen`s d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% 
                        select(model,split,set,space,'Cohen`s d'))
  
  violin_separation <- ggplot(all_dists, aes(x=set, y=value, fill = is_same)) + 
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
    geom_text(aes(x=set,y=max(all_dists  %>% select(value))+0.1, label=effsize),
              data=cohen_df ,inherit.aes = FALSE,size=5)+
    facet_wrap(~ space)
  #stat_compare_means(aes(group=is_same), method = "t.test")
  violin_separation <- violin_separation + theme(legend.position = "bottom")
  #print(violin_separation)
  plotList[[i+1]] <- violin_separation
  
  message(paste0('Done split ',i))
}


png(file="../figures/MI_results/cpa_compare_basal_and_composes_space.png",width=16,height=16,units = "in",res=600)
p <- ggarrange(plotlist=plotList,ncol=2,nrow=5,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),seq(1,10)),
               font.label = list(size = 10, color = "black", face = "plain", family = NULL),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 14))
dev.off()

colnames(df_effsize)[5] <- 'value'
df_effsize$space <- factor(df_effsize$space,levels = c("latent","basal latent"))
p  <- ggplot(df_effsize ,aes(x=space,y=value,fill=space)) + 
  geom_boxplot() +
  ylab('Cohen`s d')+
  stat_compare_means() + #,label = "p.signif"
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
        text = element_text(family = "Arial",size = 20),legend.position = "bottom")+
  facet_wrap(~ set)
print(p)
ggsave(
  '../figures/MI_results/cpa_compare_cohensd_basal_and_composed_space.eps',
  plot = p,
  device = cairo_ps,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)

png(file="../figures/MI_results/cpa_compare_cohensd_basal_and_composed_space.png",width=12,height=8,units = "in",res=600)
print(p)
dev.off()

### Plot distance distributions for two extreme cases of observed difference
diff <- NULL
for (i in 1:10){
  tmp <- df_effsize %>% filter(split==i)
  diff[i] <- mean(c(as.matrix(tmp %>% filter(set!='Train') %>% filter(space=='latent') %>% 
                                select(value))[,1]-as.matrix(tmp %>% filter(set!='Train') %>% filter(space!='latent') %>% 
                                                               select(value)),
                    as.matrix(tmp %>% filter(set=='Train') %>% filter(space=='latent') %>% 
                                select(value))[,1]-as.matrix(tmp %>% filter(set=='Train') %>% filter(space!='latent') %>% 
                                                               select(value))))
}
ind_max <- which.max(diff)
ind_min <- which.min(diff)

list_visualize <- plotList[c(ind_min,ind_max)]
p <- ggarrange(plotlist=list_visualize,ncol=2,nrow=1,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),c(ind_min,ind_max)),
               font.label = list(size = 20, color = "black", face = "plain", family = 'Arial'),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 20))
ggsave(
  '../figures/MI_results/cpa_compare_distance_basal_and_composed_space.eps', 
  device = cairo_ps,
  scale = 1,
  width = 18,
  height = 9,
  units = "in",
  dpi = 600,
)

### Normal global space--------------------------------------------------------
plotList <- NULL
distrList <- NULL
df_effsize <- data.frame()
#df_effsize_train <- data.frame()
for (i in 0:9){
  #for (i in c(2,7)){
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
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_proc_train <- process_embeddings(embs_train,sigInfo,trainInfo)
  embs_proc_test <- process_embeddings(embs_test,sigInfo,valInfo)
  
  
  # Check distributions in the latent space----
  dist_train <- samples_separation(embs_proc_train,
                                   compare_level='cell',
                                   metric = 'cosine',
                                   show_plot = F)
  dist_train <- dist_train %>% mutate(model='simlarity autoencoders')
  dist_train <- dist_train %>% mutate(set='Train')
  
  dist_test <- samples_separation(embs_proc_test,
                                  compare_level='cell',
                                  metric = 'cosine',
                                  show_plot = F)
  dist_test <- dist_test %>% mutate(model='simlarity autoencoders')
  dist_test <- dist_test %>% mutate(set='Validation')
  
  all_dists <- bind_rows(dist_train,dist_test)
  
  d_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set!='Train') %>% 
                                           filter(is_same=='Same cell-line') %>% select(value)),
                               as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% 
                                           filter(is_same!='Same cell-line')%>% select(value)),
                               ci=0.95)
  d_train = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set=='Train') %>% 
                                             filter(is_same=='Same cell-line') %>% select(value)),
                                 as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set=='Train')%>% 
                                             filter(is_same!='Same cell-line')%>% select(value)),
                                 ci=0.95)
  
  all_dists_val <- all_dists %>% filter(set!='Train') %>%  
    mutate(effsize = abs(d_val$Cohens_d))
  all_dists_train <- all_dists %>% filter(set=='Train') %>%  
    mutate(effsize = abs(d_train$Cohens_d))
  all_dists <- rbind(all_dists_train,all_dists_val)
  all_dists  <- all_dists %>% mutate(effsize = paste0('Cohen`s d: ',round(effsize,3)))
  cohen_df <- distinct(all_dists %>% select(model,effsize,set))
  df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
                        filter(effsize!='Cohen`s d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% 
                        select(model,split,set,'Cohen`s d'))
  
  violin_separation <- ggplot(all_dists, aes(x=set, y=value, fill = is_same)) + 
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
    geom_text(aes(x=set,y=max(all_dists  %>% select(value))+0.1, label=effsize),
              data=cohen_df ,inherit.aes = FALSE,size=8)
  violin_separation <- violin_separation + theme(legend.position = "bottom")
  #print(violin_separation)
  plotList[[i+1]] <- violin_separation
  
  message(paste0('Done split ',i))
}

png(file="../figures/MI_results/landmarks_similarity_trained_autoencoders_cell_separation.png",width=16,height=16,units = "in",res=600)
p <- ggarrange(plotlist=plotList,ncol=2,nrow=5,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),seq(1,10)),
               font.label = list(size = 10, color = "black", face = "plain", family = NULL),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 14))
dev.off()

colnames(df_effsize)[4] <- 'value'
p  <- ggplot(df_effsize ,aes(x=set,y=value,fill=set)) + 
  geom_boxplot() +
  ylab('Cohen`s d')+
  ylim(c(0,1))+
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
        text = element_text(family = "Arial",size = 20),legend.position = "bottom")
print(p)
ggsave(
  '../figures/MI_results/landmarks_similarity_trained_autoencoders_cell_coensd.eps',
  plot = p,
  device = cairo_ps,
  scale = 1,
  width = 9,
  height = 9,
  units = "in",
  dpi = 600,
)

png(file="../figures/MI_results/landmarks_similarity_trained_autoencoders_cell_coensd.png",width=9,height=9,units = "in",res=600)
print(p)
dev.off()

### Plot distance distributions for two extreme cases of observed difference
diff <- NULL
for (i in 1:10){
  tmp <- df_effsize %>% filter(split==i)
  diff[i] <- mean(c(as.matrix(tmp %>% filter(set!='Train') %>% select(value))[,1],
                    as.matrix(tmp %>% filter(set=='Train') %>% select(value))[,1]))
}
ind_max <- which.max(diff)
ind_min <- which.min(diff)

list_visualize <- plotList[c(ind_min,ind_max)]
p <- ggarrange(plotlist=list_visualize,ncol=2,nrow=1,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),c(ind_min,ind_max)),
               font.label = list(size = 20, color = "black", face = "plain", family = 'Arial'),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 20))
ggsave(
  '../figures/MI_results/landmarks_similarity_trained_autoencoders_cell_separation_extremes.eps', 
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)

#### See separation of same-condition for these splits-------------------------------
plotList <- NULL
df_effsize <- data.frame()
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
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_proc_train <- process_embeddings(embs_train,sigInfo,trainInfo)
  embs_proc_test <- process_embeddings(embs_test,sigInfo,valInfo)
  
  
  # Check distributions in the latent space----
  dist_train <- samples_separation(embs_proc_train,
                                   compare_level='equivalent condition',
                                   metric = 'cosine',
                                   show_plot = F)
  dist_train <- dist_train %>% mutate(model='simlarity autoencoders')
  dist_train <- dist_train %>% mutate(set='Train')
  
  dist_test <- samples_separation(embs_proc_test,
                                  compare_level='equivalent condition',
                                  metric = 'cosine',
                                  show_plot = F)
  dist_test <- dist_test %>% mutate(model='simlarity autoencoders')
  dist_test <- dist_test %>% mutate(set='Validation')
  
  all_dists <- bind_rows(dist_train,dist_test)
  
  d_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set!='Train') %>% 
                                           filter(is_same=='Same condition in different cell-line') %>% select(value)),
                               as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% 
                                           filter(is_same!='Same condition in different cell-line')%>% select(value)),
                               ci=0.95)
  d_train = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set=='Train') %>% 
                                             filter(is_same=='Same condition in different cell-line') %>% select(value)),
                                 as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set=='Train')%>% 
                                             filter(is_same!='Same condition in different cell-line')%>% select(value)),
                                 ci=0.95)
  
  all_dists_val <- all_dists %>% filter(set!='Train') %>%  
    mutate(effsize = abs(d_val$Cohens_d))
  all_dists_train <- all_dists %>% filter(set=='Train') %>%  
    mutate(effsize = abs(d_train$Cohens_d))
  all_dists <- rbind(all_dists_train,all_dists_val)
  all_dists  <- all_dists %>% mutate(effsize = paste0('Cohen`s d: ',round(effsize,3)))
  cohen_df <- distinct(all_dists %>% select(model,effsize,set))
  df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
                        filter(effsize!='Cohen`s d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% 
                        select(model,split,set,'Cohen`s d'))
  
  violin_separation <- ggplot(all_dists, aes(x=set, y=value, fill = is_same)) + 
    geom_violin(position = position_dodge(width = 1),width = 1)+geom_boxplot(position = position_dodge(width = 1),width = 0.05,
                                                                             outlier.shape = NA)+
    scale_fill_discrete(name="Embedding distance distribution in the composed latent space",
                        labels=c("Random Signatures","Same condition in different cell-line"))+
    ylim(0,2)+
    xlab("")+ylab("Cosine Distance")+ 
    theme(axis.ticks.x=element_blank(),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),text = element_text(family = "Arial",size = 20),legend.position = "bottom")+
    theme_minimal(base_family = "Arial",base_size = 20) +
    geom_text(aes(x=set,y=max(all_dists  %>% select(value))+0.1, label=effsize),
              data=cohen_df ,inherit.aes = FALSE,size=8)
  violin_separation <- violin_separation + theme(legend.position = "bottom")
  #print(violin_separation)
  plotList[[i+1]] <- violin_separation
  
  message(paste0('Done split ',i))
}
png(file="../figures/MI_results/landmarks_similarity_trained_autoencoders_samecondition_separation.png",width=16,height=16,units = "in",res=600)
p <- ggarrange(plotlist=plotList,ncol=2,nrow=5,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),seq(1,10)),
               font.label = list(size = 10, color = "black", face = "plain", family = NULL),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 14))
dev.off()

colnames(df_effsize)[4] <- 'value'
### Plot distance distributions for two extreme cases of observed difference
diff <- NULL
for (i in 1:10){
  tmp <- df_effsize %>% filter(split==i)
  diff[i] <- mean(c(as.matrix(tmp %>% filter(set!='Train') %>% select(value))[,1],
                    as.matrix(tmp %>% filter(set=='Train') %>% select(value))[,1]))
}
ind_max <- which.max(diff)
ind_min <- which.min(diff)

list_visualize <- plotList[c(ind_min,ind_max)]
p <- ggarrange(plotlist=list_visualize,ncol=2,nrow=1,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),c(ind_min,ind_max)),
               font.label = list(size = 20, color = "black", face = "plain", family = 'Arial'),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 20))
ggsave(
  '../figures/MI_results/landmarks_similarity_trained_autoencoders_samecondition_separation_extremes.eps', 
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)

#### See separation of duplicates for these splits------------------
plotList <- NULL
df_effsize <- data.frame()
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
  embs_train <- rbind(data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_proc_train <- process_embeddings(embs_train,sigInfo,trainInfo)
  embs_proc_test <- process_embeddings(embs_test,sigInfo,valInfo)
  
  
  # Check distributions in the latent space----
  dist_train <- samples_separation(embs_proc_train,
                                   compare_level='duplicates',
                                   metric = 'cosine',
                                   show_plot = F)
  dist_train <- dist_train %>% mutate(model='simlarity autoencoders')
  dist_train <- dist_train %>% mutate(set='Train')
  
  dist_test <- samples_separation(embs_proc_test,
                                  compare_level='duplicates',
                                  metric = 'cosine',
                                  show_plot = F)
  dist_test <- dist_test %>% mutate(model='simlarity autoencoders')
  dist_test <- dist_test %>% mutate(set='Validation')
  
  all_dists <- bind_rows(dist_train,dist_test)
  
  
  if (nrow(as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set=='Train') %>% 
                      filter(is_same=='Duplicate Signatures') %>% select(value)))>0 & nrow(as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set!='Train') %>% 
                                                                                                     filter(is_same=='Duplicate Signatures') %>% select(value)))>0){
    d_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set!='Train') %>% 
                                             filter(is_same=='Duplicate Signatures') %>% select(value)),
                                 as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% 
                                             filter(is_same!='Duplicate Signatures')%>% select(value)),
                                 ci=0.95)
    d_train = effectsize::cohens_d(as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set=='Train') %>% 
                                               filter(is_same=='Duplicate Signatures') %>% select(value)),
                                   as.matrix(all_dists %>% filter(model=='simlarity autoencoders') %>% filter(set=='Train')%>% 
                                               filter(is_same!='Duplicate Signatures')%>% select(value)),
                                   ci=0.95)
    
    all_dists_val <- all_dists %>% filter(set!='Train') %>%  
      mutate(effsize = abs(d_val$Cohens_d))
    all_dists_train <- all_dists %>% filter(set=='Train') %>%  
      mutate(effsize = abs(d_train$Cohens_d))
    all_dists <- rbind(all_dists_train,all_dists_val)
    all_dists  <- all_dists %>% mutate(effsize = paste0('Cohen`s d: ',round(effsize,3)))
    cohen_df <- distinct(all_dists %>% select(model,effsize,set))
    df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
                          filter(effsize!='Cohen`s d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% 
                          select(model,split,set,'Cohen`s d'))
    
    violin_separation <- ggplot(all_dists, aes(x=set, y=value, fill = is_same)) + 
      geom_violin(position = position_dodge(width = 1),width = 1)+geom_boxplot(position = position_dodge(width = 1),width = 0.05,
                                                                               outlier.shape = NA)+
      scale_fill_discrete(name="Embedding distance distribution in the composed latent space",
                          labels=c("Random Signatures","Duplicate Signatures"))+
      ylim(0,2)+
      xlab("")+ylab("Cosine Distance")+ 
      theme(axis.ticks.x=element_blank(),
            panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(),text = element_text(family = "Arial",size = 20),legend.position = "bottom")+
      theme_minimal(base_family = "Arial",base_size = 20) +
      geom_text(aes(x=set,y=max(all_dists  %>% select(value))+0.1, label=effsize),
                data=cohen_df ,inherit.aes = FALSE,size = 8)
    violin_separation <- violin_separation + theme(legend.position = "bottom")
    #print(violin_separation)
    plotList[[i+1]] <- violin_separation
  } else{
    plotList[[i+1]] <- NULL
  }
  
  message(paste0('Done split ',i))
}
png(file="../figures/MI_results/landmarks_similarity_trained_autoencoders_duplicates_separation.png",width=16,height=16,units = "in",res=600)
p <- ggarrange(plotlist=plotList,ncol=2,nrow=length(plotList)/2,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),unique(df_effsize$split)),
               font.label = list(size = 10, color = "black", face = "plain", family = NULL),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 14))
dev.off()

colnames(df_effsize)[4] <- 'value'
### Plot distance distributions for two extreme cases of observed difference
diff <- NULL
for (i in unique(df_effsize$split)){
  tmp <- df_effsize %>% filter(split==i)
  diff[i] <- mean(c(as.matrix(tmp %>% filter(set!='Train') %>% select(value))[,1],
                    as.matrix(tmp %>% filter(set=='Train') %>% select(value))[,1]))
}
ind_max <- which.max(diff)
ind_min <- which.min(diff)

list_visualize <- plotList[c(ind_min,ind_max)]
p <- ggarrange(plotlist=list_visualize,ncol=2,nrow=1,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),c(ind_min,ind_max)),
               font.label = list(size = 20, color = "black", face = "plain", family = 'Arial'),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 20))
ggsave(
  '../figures/MI_results/landmarks_similarity_trained_autoencoders_duplicates_separation_extremes.eps', 
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)

