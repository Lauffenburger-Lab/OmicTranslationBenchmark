library(tidyverse)

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
  png(paste0(save_name,'_',compare_level,'_seperation_latent_space.png'),width=12,height=8,units = "in",res=600)
  print(p)
  dev.off()
  
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
      scale_alpha_discrete(range = c(0.15,1))+
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
# Load samples info----
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

# Load train, validation info
trainInfo <- data.table::fread('../results/MI_results/trainInfo.csv',header = T) %>% column_to_rownames('V1')
valInfo <- data.table::fread('../results/MI_results/valInfo.csv',header = T) %>% column_to_rownames('V1')

# Load embeddings of pre-trained----
embs_train <- data.table::fread('../results/MI_results/embs_train.csv',header = T) %>% column_to_rownames('V1')
embs_test <- data.table::fread('../results/MI_results/embs_val.csv',header = T) %>% column_to_rownames('V1')

embs_proc_train <- process_embeddings(embs_train,sigInfo,trainInfo)
embs_proc_test <- process_embeddings(embs_test,sigInfo,valInfo)

# Check distributions in the latent space----
dist_train <- samples_separation(embs_proc_train,'train_cosine',
                                 compare_level='duplicates',metric = 'cosine')
dist_test <- samples_separation(embs_proc_test,'test_cosine',
                                compare_level='duplicates',metric = 'cosine')

# P.values from comparing
t_test_validation <- t.test(dplyr::pull(dist_test %>% filter(is_same=='Random Signatures') %>% select(value),value),
                            dplyr::pull(dist_test %>% filter(is_same!='Random Signatures') %>% select(value),value))
print(t_test_validation)

t_test_train <- t.test(dplyr::pull(dist_train %>% filter(is_same=='Random Signatures') %>% select(value),value),
                            dplyr::pull(dist_train %>% filter(is_same!='Random Signatures') %>% select(value),value))
print(t_test_train)

# Visualize in 2D----
library(factoextra)
#Run pca test
pca_test <- prcomp(embs_test,scale=F)
fviz_eig(pca_test,ncp=30)
df_pca_test<- pca_test$x[,1:2]
df_pca_test <- as.data.frame(df_pca_test)

# colnames(df_pca_test) <- c('Dim1','Dim2')
# embs_reduced_test <- df_pca_test %>% rownames_to_column('sig_id')
# embs_reduced_test <- left_join(embs_reduced_test,
#                                embs_proc_test %>% select(sig_id,cmap_name,cell_iname,duplIdentifier,conditionId) %>% unique())
# embs_reduced_test <- embs_reduced_test %>% select(Dim1,Dim2,c('ID'='duplIdentifier'))
# 
# uniq_ids <- unique(dplyr::pull(embs_reduced_test %>% group_by(ID) %>% mutate(counts=n()) %>% filter(counts>1),ID))
# no_ids <- length(uniq_ids)
# id_colors <- c('#BFBFBF',randomcoloR::randomColor(no_ids,luminosity ='bright'))
# 
# embs_sample <- embs_reduced_test %>% mutate(label=(ID %in% uniq_ids)) %>%
#     mutate(label=ifelse(label==T,
#                         ID,'others')) %>%
#     mutate(label = factor(label,
#                           levels = c('others',
#                                      uniq_ids)))
# embs_sample <- embs_sample %>% mutate(alpha_label=ifelse(label=='others','transparent','solid'))%>%
#     mutate(alpha_label=factor(alpha_label,levels = c('transparent','solid')))
  
# pca_plot_test <- ggplot(embs_sample,aes(Dim1,Dim2)) +geom_point(aes(col=label,alpha=factor(alpha_label),size=factor(alpha_label)))+
#   scale_color_manual(values = id_colors)+ 
#   scale_alpha_discrete(range = c(0.15,1))+
#   scale_size_discrete(range = c(0.5,1))+
#   ggtitle('PCA plot of duplicate samples') + xlab(paste0(dim_label,'1'))+ ylab(paste0(dim_label,'2'))+
#   theme(text = element_text(size=15))+guides(alpha = "none",size="none")
png(file="../figures/pca_plot_samecondition_test.png",width=24,height=18,units = "in",res=300)
pca_plot_test <- visualize_projection_results(df_pca_test,embs_proc_test,
                                              compare_level='equivalent condition',method='PCA')
annotate_figure(pca_plot_test, 
                top = text_grob('PCA plot to visualize similar conditions', color = "black",face = 'plain', size = 20))
#print(pca_plot_test)
dev.off()
#Run pca train
pca_train <- prcomp(embs_train,scale=F)
fviz_eig(pca_train,ncp=30)
df_pca_train<- pca_train$x[,1:2]
df_pca_train <- as.data.frame(df_pca_train)
png(file="../figures/pca_plot_samecondition_train.png",width=24,height=18,units = "in",res=300)
pca_plot_train <- visualize_projection_results(df_pca_train,embs_proc_train,
                                               compare_level='equivalent condition',method='PCA')
annotate_figure(pca_plot_train, 
                top = text_grob('PCA plot to visualize similar conditions', color = "black",face = 'plain', size = 20))
dev.off()

# Run t-SNE
library(Rtsne)
perpl = DescTools::RoundTo(sqrt(nrow(embs_train)), multiple = 5, FUN = round)
#Use the above formula to calculate perplexity (perpl). But if perplexity is too large for the number of data you have define manually
#perpl=2
init_dim = 15
iter = 2000
emb_size = ncol(embs_train)
tsne_all <- Rtsne(embs_train, 
                  dims = 2, perplexity=perpl, 
                  verbose=TRUE, max_iter = iter,initial_dims = init_dim,check_duplicates = F)
df_tsne_train <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
rownames(df_tsne_train) <- rownames(embs_train)


png(file="../figures/tsne_plot_samecondition_train.png",width=24,height=18,units = "in",res=300)
tsne_plot_train <- visualize_projection_results(df_tsne_train,embs_proc_train,
                                                compare_level='equivalent condition',method='t-SNE')
annotate_figure(tsne_plot_train, 
                top = text_grob('t-SNE plot to visualize similar conditions', color = "black",face = 'plain', size = 20))
dev.off()


perpl = DescTools::RoundTo(sqrt(nrow(embs_test)), multiple = 5, FUN = round)
#Use the above formula to calculate perplexity (perpl). But if perplexity is too large for the number of data you have define manually
#perpl=2
init_dim = 15
iter = 2000
emb_size = ncol(embs_test)
tsne_all <- Rtsne(embs_test, 
                  dims = 2, perplexity=perpl, 
                  verbose=TRUE, max_iter = iter,initial_dims = init_dim,check_duplicates = F)
df_tsne_test <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
rownames(df_tsne_test) <- rownames(embs_test)
# colnames(df_tsne_test) <- c('Dim1','Dim2')
# embs_reduced_test <- df_tsne_test %>% rownames_to_column('sig_id')
# embs_reduced_test <- left_join(embs_reduced_test,
#                                embs_proc_test %>% select(sig_id,cmap_name,cell_iname,duplIdentifier,conditionId) %>% unique())
# embs_reduced_test <- embs_reduced_test %>% select(Dim1,Dim2,c('ID'='duplIdentifier'))
# 
# uniq_ids <- unique(dplyr::pull(embs_reduced_test %>% group_by(ID) %>% mutate(counts=n()) %>% filter(counts>1),ID))
# no_ids <- length(uniq_ids)
# id_colors <- c('#BFBFBF',randomcoloR::randomColor(no_ids,luminosity ='bright'))
# 
# embs_sample <- embs_reduced_test %>% mutate(label=(ID %in% uniq_ids)) %>%
#   mutate(label=ifelse(label==T,
#                       ID,'others')) %>%
#   mutate(label = factor(label,
#                         levels = c('others',
#                                    uniq_ids)))
# embs_sample <- embs_sample %>% mutate(alpha_label=ifelse(label=='others','transparent','solid'))%>%
#   mutate(alpha_label=factor(alpha_label,levels = c('transparent','solid')))
# 
# tsne_plot_test <- ggplot(embs_sample,aes(Dim1,Dim2)) +geom_point(aes(col=label,alpha=factor(alpha_label),size=factor(alpha_label)))+
#   scale_color_manual(values = id_colors)+ 
#   scale_alpha_discrete(range = c(0.15,1))+
#   scale_size_discrete(range = c(0.5,1))+
#   ggtitle('t-SNE plot of duplicate samples') + xlab(paste0('Dim 1'))+ ylab(paste0(dim_label,'Dim 2'))+
#   theme(text = element_text(size=15))+guides(alpha = "none",size="none")

png(file="../figures/tsne_plot_samecondition_test.png",width=24,height=18,units = "in",res=300)
tsne_plot_test <- visualize_projection_results(df_tsne_test,embs_proc_test,
                                                compare_level='equivalent condition',method='t-SNE')
annotate_figure(tsne_plot_test, 
                top = text_grob('t-SNE plot to visualize similar conditions', color = "black",face = 'plain', size = 20))
#print(tsne_plot_test)
dev.off()
