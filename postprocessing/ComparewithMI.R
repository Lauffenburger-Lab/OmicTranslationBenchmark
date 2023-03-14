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

visualize_projection_results <- function(embs_reduced,processed_embbedings,
                                         method,
                                         iters = 10,
                                         no_ids=5,
                                         compare_level=c('duplicates',
                                                         'equivalent condition',
                                                         'cell',
                                                         'drug',
                                                         'cell-drug')){
  library(ggplot2)
  library(ggpubr)
  colnames(embs_reduced) <- c('Dim1','Dim2')
  if (method=='PCA'){
    dim_label <- 'PC '
  } else{
    dim_label <- 'Dimension '
  }
  embs_reduced <- embs_reduced %>% rownames_to_column('sig_id')
  processed_embbedings <- processed_embbedings %>% select(sig_id,cmap_name,cell_iname,duplIdentifier,conditionId) %>% unique()
  embs_reduced <- left_join(embs_reduced,processed_embbedings)
  
  if (compare_level=='duplicates'){
    embs_reduced <- embs_reduced %>% select(Dim1,Dim2,c('ID'='duplIdentifier'))
  }else if (compare_level=='equivalent condition'){
    embs_reduced <- embs_reduced %>% select(Dim1,Dim2,c('ID'='conditionId'))
  }else if (compare_level=='cell'){
    embs_reduced <- embs_reduced %>% select(Dim1,Dim2,c('ID'='cell_iname'))
  }else if (compare_level=='drug'){
    embs_reduced <- embs_reduced %>% select(Dim1,Dim2,c('ID'='cmap_name'))
  } else if (compare_level=='cell-drug'){
    embs_reduced <- embs_reduced %>% select(Dim1,Dim2,cmap_name,cell_iname) %>%
      mutate(ID=paste0(cmap_name,'_',cell_iname)) %>% select(-cmap_name,-cell_iname)
  }
  
  uniq_ids <- unique(dplyr::pull(embs_reduced %>% group_by(ID) %>% mutate(counts=n()) %>% filter(counts>1),ID))
  id_colors <- c('#BFBFBF',randomcoloR::randomColor(no_ids,luminosity ='bright'))
  plotList <- NULL
  for (i in 1:iters){
    sampleIDs <- sample(uniq_ids,no_ids)
    embs_sample <- embs_reduced %>% mutate(label=(ID %in% sampleIDs)) %>%
      mutate(label=ifelse(label==T,
                          ID,'others')) %>%
      mutate(label = factor(label,
                            levels = c('others',
                                       sampleIDs)))
    embs_sample <- embs_sample %>% mutate(alpha_label=ifelse(label=='others','transparent','solid'))%>%
      mutate(alpha_label=factor(alpha_label,levels = c('transparent','solid')))
    
    plotList[[i]]  <- ggplot(embs_sample,aes(Dim1,Dim2)) +geom_point(aes(col=label,alpha=factor(alpha_label),size=factor(alpha_label)))+
      scale_color_manual(values = id_colors)+ 
      scale_alpha_discrete(range = c(0.25,1))+
      scale_size_discrete(range = c(0.5,1.5))+
      ggtitle('') + xlab(paste0(dim_label,'1'))+ ylab(paste0(dim_label,'2'))+
      theme(text = element_text(size=15))+guides(alpha = "none",size="none")
    
    uniq_ids <- uniq_ids[!uniq_ids %in% sampleIDs]
  }
  
  #png(file=paste0(save_name,".png"),width=24,height=18,units = "in",res=300)
  
  p <- ggarrange(plotlist=plotList,ncol=2,nrow=ceiling(iters/2),common.legend = FALSE)
  #print(p)
  #annotate_figure(p, top = text_grob(paste0(method,' plot to visualize similar conditions'), color = "black",face = 'plain', size = 20))
  #dev.off()
  
  return(p)
  
}
cell_line_pca_visualize <- function(embs,processed_embbedings,dim=2,show_plot=T,colors=data.frame(col=c("#BFBFBF",
                                                                       "#47ed0b",
                                                                       "#a81443",
                                                                       "#78db0f",
                                                                       "#f96d93",
                                                                       "#bc1836", 
                                                                       "#d3bb32", 
                                                                       "#f461e1", 
                                                                       "#5fbee8",
                                                                       "#a54a29"))){
  
  
  pca <- prcomp(embs,scale=F)
  df_pca<- pca$x[,1:dim]
  df_pca <- as.data.frame(df_pca)
  
  col_names <- paste0(rep('PC',dim),seq(1,dim))
  colnames(df_pca) <- col_names
  embs_reduced <- df_pca %>% rownames_to_column('sig_id')
  embs_reduced <- left_join(embs_reduced,
                            processed_embbedings %>% select(sig_id,cmap_name,cell_iname,duplIdentifier,conditionId) %>% unique())
  embs_reduced <- embs_reduced %>% select(all_of(col_names),c('cell-line'='cell_iname'))
  
  uniq_cells <- unique(embs_reduced$`cell-line`)
  id_colors <- colors$col
  if (length(uniq_cells)>length(id_colors)){
    rand_colors <- randomcoloR::randomColor(1000,luminosity ='bright')
    rand_colors <- rand_colors[which(!(rand_colors %in% id_colors))]
    id_colors <- c(id_colors,rand_colors[1:(length(uniq_cells)-length(id_colors))])
  }
  
  if (dim==2){
    pca_plot <- ggplot(embs_reduced,aes(PC1,PC2)) +geom_point(aes(col=`cell-line`))+
      scale_color_manual(values = id_colors)+
      ggtitle('PCA plot of cell-line associated experiments in the latent space') + xlab(paste0('PC1'))+ ylab(paste0('PC2'))+
      theme(text = element_text(size=14))+guides(alpha = "none",size="none")
  }else{
    pca_plot <- ggplot(embs_reduced,aes(x=PC1,y=PC2,z=PC3,col=`cell-line`))+
      ggtitle('PCA plot of cell-line associated experiments in the latent space') +
      scale_color_manual(values = id_colors)+
      theme_void() +
      labs_3D(labs=c("PC1", "PC2", "PC3"),
              angle=c(0,0,0),
              hjust=c(0,2,2),
              vjust=c(2,2,-1))+
      axes_3D() +
      stat_3D()+
      theme(text = element_text(size=14),plot.title = element_text(hjust = 0.5))+
      guides(alpha = "none",size="none")
  }
  if (show_plot==T){
    print(pca_plot)
  }
  return(pca_plot)
}
visualize_embeddings_distribution <- function(embs,all=FALSE){
  if (all==T){
    library(ggridges)
    ggplot(as.data.frame(embs) %>% gather('latent_dim','value'),aes(x =value,y=latent_dim ,fill=latent_dim)) + 
      geom_density_ridges(scale = 0.9) + ylab('Latent Dimesnion') + theme(legend.position="none")
  }else{
    hist(as.matrix(embs),breaks=100,xlab='embedding value',main= 'Latent space embeddings distribution')
  }
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
  embs_train_mi <- rbind(data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_a375.csv'),header = T),
                         data.table::fread(paste0('../results/MI_results/embs/train/trainEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test_mi <- rbind(data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_a375.csv'),header = T),
                        data.table::fread(paste0('../results/MI_results/embs/validation/valEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  
  embs_train <- rbind(data.table::fread(paste0('../results/my_results//embs/train/trainEmbs_',i,'_a375.csv'),header = T),
                      data.table::fread(paste0('../results/my_results/embs/train/trainEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  embs_test <- rbind(data.table::fread(paste0('../results/my_results/embs/validation/valEmbs_',i,'_a375.csv'),header = T),
                     data.table::fread(paste0('../results/my_results/embs/validation/valEmbs_',i,'_ht29.csv'),header = T)) %>% unique() %>%
    column_to_rownames('V1')
  
  embs_proc_train_mi <- process_embeddings(embs_train_mi,sigInfo,trainInfo)
  embs_proc_test_mi <- process_embeddings(embs_test_mi,sigInfo,valInfo)
  
  embs_proc_train <- process_embeddings(embs_train,sigInfo,trainInfo)
  embs_proc_test <- process_embeddings(embs_test,sigInfo,valInfo)
  
  # Check distributions in the latent space----
  dist_train_mi <- samples_separation(embs_proc_train_mi,
                                      compare_level='cell',
                                      metric = 'cosine',
                                      show_plot = F)
  dist_train_mi <- dist_train_mi %>% mutate(model='Autoencoders with MI')
  dist_train_mi <- dist_train_mi %>% mutate(set='Train')
  dist_test_mi <- samples_separation(embs_proc_test_mi,
                                     compare_level='cell',
                                     metric = 'cosine',
                                     show_plot = F)
  dist_test_mi <- dist_test_mi %>% mutate(model='Autoencoders with MI')
  dist_test_mi <- dist_test_mi %>% mutate(set='Validation')
  
  dist_train <- samples_separation(embs_proc_train,
                                   compare_level='cell',
                                   metric = 'cosine',
                                   show_plot = F)
  dist_train <- dist_train %>% mutate(model='Autoencoders')
  dist_train <- dist_train %>% mutate(set='Train')
  
  dist_test <- samples_separation(embs_proc_test,
                                  compare_level='cell',
                                  metric = 'cosine',
                                  show_plot = F)
  dist_test <- dist_test %>% mutate(model='Autoencoders')
  dist_test <- dist_test %>% mutate(set='Validation')
  
  all_dists <- bind_rows(dist_train_mi,dist_test_mi,dist_train,dist_test)
  
  violin_separation <- ggplot(all_dists %>% filter(set=='Validation'), aes(x=model, y=value, fill = is_same)) + 
    geom_violin(position = position_dodge(width = 1),width = 1)+geom_boxplot(position = position_dodge(width = 1),width = 0.05,
                                                                             outlier.shape = NA)+
    scale_fill_discrete(name="Embedding distance distribution",
                        labels=c("Random Signatures","Same cell-line"))+
    ylim(0,max(all_dists$value))+
    xlab("")+ylab("Cosine Distance")+ 
    theme(axis.ticks.x=element_blank(),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),text = element_text(family = "Arial",size = 20),legend.position = "bottom")+
    theme_minimal(base_family = "Arial",base_size = 20)
  violin_separation <- violin_separation + theme(legend.position = "bottom")
  plotList[[i+1]] <- violin_separation
  
  #distrList[[i+1]]<-visualize_embeddings_distribution2(rbind(embs_test_mi%>%gather('key','value')%>%mutate(key='Autoencoder MI'),
  #                                                     embs_test%>%gather('key','value')%>%mutate(key='Autoencoder Simple')))
  
  # png(paste0('../figures/MI_results/compare_pca/pca_2d_cells_mi_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # pca_test_mi <- cell_line_pca_visualize(embs_test_mi,embs_proc_test_mi,dim=2,show_plot = F)
  # print(pca_test_mi)
  # dev.off()
  # 
  # png(paste0('../figures/MI_results/compare_pca/pca_2d_cells_simple_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # pca_test <- cell_line_pca_visualize(embs_test,embs_proc_test,dim=2,show_plot = F)
  # print(pca_test)
  # dev.off()  
  # 
  # png(paste0('../figures/MI_results/compare_latent_distr/latent_distr_mi_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # visualize_embeddings_distribution(embs_test_mi)
  # dev.off()
  # 
  # png(paste0('../figures/MI_results/compare_latent_distr/latent_distr_simple_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # visualize_embeddings_distribution(embs_test)
  # dev.off()
  
  
  message(paste0('Done split ',i))
}

library(ggpubr)
png(file="../figures/MI_results/samecell_vs_random_cosine_embs_comparison_valonly.png",width=10,height=16,units = "in",res=600)
p <- ggarrange(plotlist=plotList,ncol=2,nrow=5,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',10),seq(1,10)),
               font.label = list(size = 10, color = "black", face = "plain", family = NULL),
               hjust=-0.15)

annotate_figure(p, top = text_grob("Distributions of embedding distances in the latent space", 
                                   color = "black",face = 'plain', size = 14))
dev.off()

# Visualize in 2D----
library(factoextra)
#Run pca test
pca_test <- prcomp(embs_test_mi,scale=F)
fviz_eig(pca_test,ncp=50)
df_pca_test<- pca_test$x[,1:2]
df_pca_test <- as.data.frame(df_pca_test)


# Run t-SNE
library(Rtsne)
#perpl = DescTools::RoundTo(sqrt(nrow(embs_test_mi)), multiple = 5, FUN = round)
perpl=10
init_dim = 10
iter = 1000
emb_size = ncol(embs_test_mi)
tsne_all <- Rtsne(embs_test_mi, 
                  dims = 2, perplexity=perpl, 
                  verbose=TRUE, max_iter = iter,
                  initial_dims = init_dim,check_duplicates = F,
                  normalize = F,pca_scale = T,
                  num_threads = 15)
df_tsne_test <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
rownames(df_tsne_test) <- rownames(embs_test_mi)

colnames(df_tsne_test) <- c('Dim1','Dim2')
embs_reduced_test <- df_tsne_test %>% rownames_to_column('sig_id')
embs_reduced_test <- left_join(embs_reduced_test,
                               embs_proc_test_mi %>% select(sig_id,cmap_name,cell_iname,duplIdentifier,conditionId) %>% unique())
embs_reduced_test <- embs_reduced_test %>% select(Dim1,Dim2,c('ID'='duplIdentifier'))

uniq_ids <- unique(dplyr::pull(embs_reduced_test %>% group_by(ID) %>% mutate(counts=n()) %>% filter(counts>1),ID))
no_ids <- length(uniq_ids)
id_colors <- c('#BFBFBF',randomcoloR::randomColor(no_ids,luminosity ='bright'))

embs_sample <- embs_reduced_test %>% mutate(label=(ID %in% uniq_ids)) %>%
  mutate(label=ifelse(label==T,
                      ID,'others')) %>%
  mutate(label = factor(label,
                        levels = c('others',
                                   uniq_ids)))
embs_sample <- embs_sample %>% mutate(alpha_label=ifelse(label=='others','transparent','solid'))%>%
  mutate(alpha_label=factor(alpha_label,levels = c('transparent','solid')))

tsne_plot_test <- ggplot(embs_sample,aes(Dim1,Dim2)) +geom_point(aes(col=label,))+
  scale_color_manual(values = id_colors)+
  ggtitle('t-SNE plot of duplicate samples') + xlab('Dim 1')+ ylab('Dim 2')+
  theme(text = element_text(size=15))+guides(alpha = "none",size="none")

print(tsne_plot_test)
