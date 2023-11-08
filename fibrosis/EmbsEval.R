library(tidyverse)
library(gg3D)
library(ggsignif)
library(ggpubr)

samples_separation <- function(embbedings,
                               compare_level=c('equivalent condition',
                                               'diagnosis',
                                               'cell',
                                               'species'),
                               metric=c("euclidean", "maximum", "manhattan",
                                        "canberra", "binary","cosine"),
                               show_plot=TRUE){
  library(tidyverse)
  embs <- embbedings %>% column_to_rownames('id') %>%
    select(-species,-cell_type,-diagnosis)
  sample_info <- embbedings %>% select(id,species,cell_type,diagnosis) %>% mutate(conditionId = paste0(cell_type,'_',diagnosis))
  
  
  # calculate distance matrix
  if (metric=='cosine'){
    #library(lsa)
    #mat <- t(embs)
    mat <- as.matrix(embs)
    sim_matrix <- tcrossprod(mat) / (sqrt(rowSums(mat^2)) %*% t(sqrt(rowSums(mat^2))))
    #dist <- 1 - cosine(mat)
    dist <- 1 - sim_matrix
  } else{
    dist <- as.matrix(dist(embs, method = metric))
  }
  
  # Conver to long format data frame
  # Keep only unique (non-self) pairs
  dist[lower.tri(dist,diag = T)] <- NA
  dist <- reshape2::melt(dist)
  dist <- dist %>% filter(!is.na(value))
  
  # Merge meta-data info and distances values
  dist <- left_join(dist,sample_info,by = c("Var1"="id"))
  dist <- left_join(dist,sample_info,by = c("Var2"="id"))
  dist <- dist %>% filter(!is.na(value))
  
  if (compare_level=='equivalent condition'){
    dist <- dist %>% mutate(is_same = (conditionId.x==conditionId.y))
    label <- 'Same condition in different species'
  }else if (compare_level=='cell'){
    dist <- dist %>% mutate(is_same = (cell_type.x==cell_type.y))
    label <- 'Same cell type'
  }else if (compare_level=='species'){
    dist <- dist %>% mutate(is_same = (species.x==species.y))
    label <- 'Same species'
  } else if (compare_level=='diagnosis'){
    dist <- dist %>% mutate(is_same = (diagnosis.x==diagnosis.y))
    label <- 'Same diagnosis'
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
  
  if (show_plot==T){
    print(p)
  } 
  return(dist)
}


pca_visualize <- function(embbedings,dim=2,scale=F,show_plot=T,
                                  space_annotation = c('composed latent','basal latent'),
                                  colors=data.frame(col=c("#125b80",
                                                          "#cc8110",
                                                          "#0b7545",
                                                          "#a81443",
                                                          "#78db0f",
                                                          "#f96d93",
                                                          "#bc1836", 
                                                          "#d3bb32", 
                                                          "#f461e1", 
                                                          "#5fbee8",
                                                          "#a54a29"))){
  
  embs <- embbedings %>% column_to_rownames('id') %>%
    select(-species,-cell_type,-diagnosis)
  sample_info <- embbedings %>% select(id,species,cell_type,diagnosis) %>% mutate(conditionId = paste0(cell_type,'_',diagnosis))
  
  pca <- prcomp(embs,scale=scale)
  df_pca<- pca$x[,1:dim]
  df_pca <- as.data.frame(df_pca)
  
  col_names <- paste0(rep('PC',dim),seq(1,dim))
  colnames(df_pca) <- col_names
  embs_reduced <- df_pca %>% rownames_to_column('id')
  embs_reduced <- left_join(embs_reduced,
                            sample_info %>% select(id,species,cell_type,diagnosis) %>% 
                              mutate(conditionId = paste0(cell_type,'_',diagnosis)) %>% 
                              unique())
  embs_reduced <- embs_reduced %>% select(all_of(col_names),c('cell type'='cell_type'),species)
  
  uniq_cells <- unique(embs_reduced$`cell type`)
  id_colors <- colors$col
  if (length(uniq_cells)>length(id_colors)){
    rand_colors <- randomcoloR::randomColor(1000,luminosity ='dark')
    rand_colors <- rand_colors[which(!(rand_colors %in% id_colors))]
    id_colors <- c(id_colors,rand_colors[1:(length(uniq_cells)-length(id_colors))])
  }
  
  if (dim==2){
    pca_plot <- ggplot(embs_reduced,aes(PC1,PC2)) +geom_point(aes(col=`cell type`,shape=species))+
      scale_color_manual(values = id_colors)+
      ggtitle(paste0('PCA plot of single cell data in the ',space_annotation,' space')) + 
      xlab(paste0('PC1'))+ ylab(paste0('PC2'))+theme_minimal()+
      theme(text = element_text(size=14),plot.title = element_text(hjust = 0.5),
                    legend.text=element_text(size=14))
  }else{
    pca_plot <- ggplot(embs_reduced,aes(x=PC1,y=PC2,z=PC3,col=`cell type`,shape=species))+
      ggtitle(paste0('PCA plot of single cell data in the ',space_annotation,' space')) +
      scale_color_manual(values = id_colors)+
      theme_void() +
      labs_3D(labs=c("PC1", "PC2", "PC3"),
              angle=c(0,0,0),
              hjust=c(0,2,2),
              vjust=c(2,2,-1))+
      axes_3D() +
      stat_3D()+
      theme(text = element_text(size=14),plot.title = element_text(hjust = 0.5),
            legend.text=element_text(size=14))
  }
  if (show_plot==T){
    print(pca_plot)
  }
  return(pca_plot)
}
tsne_visualize <- function(embbedings,dim=2,scale=F,normalize=F,show_plot=T,init_dim=10,iter=1000,
                                   space_annotation = c('composed latent','basal latent'),
                                   colors=data.frame(col=c("#125b80","#cc8110",
                                                             "#0b7545",
                                                             "#a81443",
                                                             "#78db0f",
                                                             "#f96d93",
                                                             "#bc1836", 
                                                             "#d3bb32", 
                                                             "#f461e1", 
                                                             "#5fbee8",
                                                             "#a54a29"))){
  
  embs <- embbedings %>% column_to_rownames('id') %>%
    select(-species,-cell_type,-diagnosis)
  sample_info <- embbedings %>% select(id,species,cell_type,diagnosis) %>% mutate(conditionId = paste0(cell_type,'_',diagnosis))
  
  library(Rtsne)
  perpl = DescTools::RoundTo(sqrt(nrow(embs)), multiple = 5, FUN = round)
  emb_size = ncol(embs)
  set.seed(42)
  tsne_all <- Rtsne(embs, 
                    dims = dim, perplexity=perpl, 
                    verbose=F, max_iter = iter,initial_dims = init_dim,check_duplicates = T,
                    normalize = normalize,pca_scale = scale,
                    num_threads = 15)
  if (dim==2){
    df_tsne <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
  }else{
    df_tsne <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2],V3 =tsne_all$Y[,3])
  }
  rownames(df_tsne) <- rownames(embs)
  
  col_names <- paste0(rep('tSNE-',dim),seq(1,dim))
  colnames(df_tsne) <- col_names
  embs_reduced <- df_tsne %>% rownames_to_column('id')
  embs_reduced <- left_join(embs_reduced,
                            sample_info %>% select(id,species,cell_type,diagnosis) %>% 
                              mutate(conditionId = paste0(cell_type,'_',diagnosis)) %>% 
                                       unique())
  embs_reduced <- embs_reduced %>% select(all_of(col_names),c('cell type'='cell_type'),species)
  
  uniq_cells <- unique(embs_reduced$`cell type`)
  id_colors <- colors$col
  if (length(uniq_cells)>length(id_colors)){
    rand_colors <- randomcoloR::randomColor(1000,luminosity ='dark')
    rand_colors <- rand_colors[which(!(rand_colors %in% id_colors))]
    id_colors <- c(id_colors,rand_colors[1:(length(uniq_cells)-length(id_colors))])
  }
  
  if (dim==2){
    tsne_plot <- ggplot(embs_reduced,aes(`tSNE-1`,`tSNE-2`)) +geom_point(aes(col=`cell type`,shape=species))+
      scale_color_manual(values = id_colors)+
      ggtitle(paste0('t-SNE plot of single cell data in the ',space_annotation,' space')) + 
      xlab(paste0('tSNE-1'))+ ylab(paste0('tSNE-2'))+theme_minimal()+
      theme(text = element_text(size=14),plot.title = element_text(hjust = 0.5),
            legend.text=element_text(size=14))
  }else{
    tsne_plot <- ggplot(embs_reduced,aes(x=`tSNE-1`,y=`tSNE-2`,z=`tSNE-3`,col=`cell type`,shape=species))+
      ggtitle(paste0('t-SNE plot of single cell data in the ',space_annotation,' space')) +
      scale_color_manual(values = id_colors)+
      theme_void() +
      labs_3D(labs=c("tSNE-1", "tSNE-2", "tSNE-3"),
              angle=c(0,0,0),
              hjust=c(0,2,2),
              vjust=c(2,2,-1))+
      axes_3D(phi=30) +
      stat_3D()+
      theme(text = element_text(size=14),plot.title = element_text(hjust = 0.5),
            legend.text=element_text(size=14))
  }
  if (show_plot==T){
    print(tsne_plot)
  }
  return(tsne_plot)
}


umap_visualize <- function(embbedings,dim=2,show_plot=T,n_neighbors=15,metric='cosine',knn_repeats=5,
                                   space_annotation = c('composed latent','basal latent'),
                                   colors=data.frame(col=c("#125b80","#cc8110",
                                                           "#0b7545",
                                                           "#a81443",
                                                           "#78db0f",
                                                           "#f96d93",
                                                           "#bc1836", 
                                                           "#d3bb32", 
                                                           "#f461e1", 
                                                           "#5fbee8",
                                                           "#a54a29"))){
  
  embs <- embbedings %>% column_to_rownames('id') %>%
    select(-species,-cell_type,-diagnosis)
  sample_info <- embbedings %>% select(id,species,cell_type,diagnosis) %>% mutate(conditionId = paste0(cell_type,'_',diagnosis))
  
  library(umap)
  set.seed(42)
  umap_all <- umap(embs, n_components = dim,n_neighbors=n_neighbors,metric=metric, random_state = 42)
  df_umap <- data.frame(umap_all$layout)
  colnames(df_umap) <- paste0('umap-',seq(1,ncol(df_umap)))
  rownames(df_umap) <- rownames(embs)
  col_names <- colnames(df_umap)
  embs_reduced <- df_umap %>% rownames_to_column('id')
  embs_reduced <- left_join(embs_reduced,
                            sample_info %>% select(id,species,cell_type,diagnosis) %>% 
                              mutate(conditionId = paste0(cell_type,'_',diagnosis)) %>% 
                              unique())
  embs_reduced <- embs_reduced %>% select(all_of(col_names),c('cell type'='cell_type'),species)
  
  uniq_cells <- unique(embs_reduced$`cell type`)
  id_colors <- colors$col
  if (length(uniq_cells)>length(id_colors)){
    rand_colors <- randomcoloR::randomColor(1000,luminosity ='dark')
    rand_colors <- rand_colors[which(!(rand_colors %in% id_colors))]
    id_colors <- c(id_colors,rand_colors[1:(length(uniq_cells)-length(id_colors))])
  }
  
  if (dim==2){
    umap_plot <- ggplot(embs_reduced,aes(`umap-1`,`umap-2`)) +geom_point(aes(col=`cell type`,shape=species))+
      scale_color_manual(values = id_colors)+
      ggtitle(paste0('uMAP of single cell data in the ',space_annotation,' space')) + 
      xlab(paste0('umap-1'))+ ylab(paste0('umap-2'))+theme_minimal()+
      theme(text = element_text(size=14),plot.title = element_text(hjust = 0.5),
            legend.text=element_text(size=14))
  }else{
    umap_plot <- ggplot(embs_reduced,aes(x=`umap-1`,y=`umap-2`,z=`umap-3`,col=`cell type`,shape=species))+
      ggtitle(paste0('uMAP of single cell data in the ',space_annotation,' space')) +
      scale_color_manual(values = id_colors)+
      theme_void() +
      labs_3D(labs=c("umap-1", "umap-2", "umap-3"),
              angle=c(0,0,0),
              hjust=c(0,2,2),
              vjust=c(2,2,-1))+
      axes_3D() +
      stat_3D()+
      theme(text = element_text(size=14),plot.title = element_text(hjust = 0.5),
            legend.text=element_text(size=14))
  }
  if (show_plot==T){
    print(umap_plot)
  }
  return(umap_plot)
}

plotList <- NULL
distrList <- NULL
df_effsize <- data.frame()
#df_effsize_train <- data.frame()
for (i in 0:9){
  message(paste0('Start split ',i))
  # Load embeddings of pre-trained
  # embs_train_mi <- rbind(data.table::fread(paste0('results/embs/train/trainEmbs_',i,'_human.csv'),header = T) %>% mutate(species='human') %>%
  #                        select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('human_',id)),
  #                        data.table::fread(paste0('results/embs/train/trainEmbs_',i,'_mouse.csv'),header = T) %>% mutate(species='mouse') %>%
  #                        select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('mouse_',id))) %>% unique()
  
  embs_test_mi <- rbind(data.table::fread(paste0('results/embs/validation/valEmbs_',i,'_human.csv'),header = T) %>% mutate(species='human') %>%
                           select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('human_',id)),
                         data.table::fread(paste0('results/embs/validation/valEmbs_',i,'_mouse.csv'),header = T) %>% mutate(species='mouse') %>%
                           select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('mouse_',id))) %>% unique()
  
  # embs_train <- rbind(data.table::fread(paste0('results/embs/train/trainEmbs_base_',i,'_human.csv'),header = T) %>% mutate(species='human') %>%
  #                     select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('human_',id)),
  #                     data.table::fread(paste0('results/embs/train/trainEmbs_base_',i,'_mouse.csv'),header = T) %>% mutate(species='mouse') %>% 
  #                     select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('mouse_',id))) %>% unique()
  embs_test <- rbind(data.table::fread(paste0('results/embs/validation/valEmbs_base_',i,'_human.csv'),header = T) %>% mutate(species='human')%>%
                     select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('human_',id)),
                     data.table::fread(paste0('results/embs/validation/valEmbs_base_',i,'_mouse.csv'),header = T) %>% mutate(species='mouse')%>%
                     select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('mouse_',id))) %>% unique()
  
  # embs_train_mi <- embs_train_mi %>% mutate(cell_type=ifelse(cell_type==1,'immune',
  #                                                            ifelse(cell_type==2,'mesenchymal',
  #                                                                   ifelse(cell_type==3,'epithelial',
  #                                                                          ifelse(cell_type==4,'endothelial','stem cell')))))
  # embs_train <- embs_train %>% mutate(cell_type=ifelse(cell_type==1,'immune',
  #                                                             ifelse(cell_type==2,'mesenchymal',
  #                                                                    ifelse(cell_type==3,'epithelial',
  #                                                                           ifelse(cell_type==4,'endothelial','stem cell')))))
  # embs_train <- sample_n(embs_train, ceiling(0.1*nrow(embs_train)))
  # embs_train <- embs_train %>% arrange(id)
  # embs_train_mi <- embs_train_mi %>% filter(id %in% embs_train$id)
  # embs_train_mi <- embs_train_mi %>% arrange(id)
  embs_test_mi <- embs_test_mi %>% mutate(cell_type=ifelse(cell_type==1,'immune',
                                                              ifelse(cell_type==2,'mesenchymal',
                                                                     ifelse(cell_type==3,'epithelial',
                                                                            ifelse(cell_type==4,'endothelial','stem cell')))))
  embs_test <- embs_test %>% mutate(cell_type=ifelse(cell_type==1,'immune',
                                                              ifelse(cell_type==2,'mesenchymal',
                                                                     ifelse(cell_type==3,'epithelial',
                                                                            ifelse(cell_type==4,'endothelial','stem cell')))))
  
  
  # Check distributions in the latent space----
  # dist_train_mi <- samples_separation(embs_train_mi,
  #                                    compare_level='species',
  #                                    metric = 'cosine',
  #                                    show_plot = F)
  # dist_train_mi <- dist_train_mi %>% mutate(space='latent')
  # dist_train_mi <- dist_train_mi %>% mutate(set='Train')
  dist_test_mi <- samples_separation(embs_test_mi,
                                     compare_level='species',
                                     metric = 'cosine',
                                     show_plot = F)
  dist_test_mi <- dist_test_mi %>% mutate(space='latent')
  dist_test_mi <- dist_test_mi %>% mutate(set='Validation')
  
  # dist_train <- samples_separation(embs_train,
  #                                 compare_level='species',
  #                                 metric = 'cosine',
  #                                 show_plot = F)
  # dist_train <- dist_train %>% mutate(space='Basal latent')
  # dist_train <- dist_train %>% mutate(set='Train')
  
  dist_test <- samples_separation(embs_test,
                                  compare_level='species',
                                  metric = 'cosine',
                                  show_plot = F)
  dist_test <- dist_test %>% mutate(space='Basal latent')
  dist_test <- dist_test %>% mutate(set='Validation')
  
  all_dists <- bind_rows(dist_test_mi,dist_test)
  # all_dists <- bind_rows(dist_test_mi,dist_test,dist_train,dist_train_mi)
  
  d_val_basal = effectsize::cohens_d(as.matrix(all_dists %>% filter(space=='Basal latent') %>% filter(set!='Train') %>% 
                                              filter(is_same=='Same species') %>% select(value)),
                                  as.matrix(all_dists %>% filter(space=='Basal latent')%>% filter(set!='Train') %>%
                                              filter(is_same!='Same species')%>% select(value)),
                                  ci=0.95)
  d_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(space!='Basal latent')%>% filter(set!='Train') %>%
                                              filter(is_same=='Same species') %>% select(value)),
                                  as.matrix(all_dists %>% filter(space!='Basal latent') %>% filter(set!='Train')%>%
                                              filter(is_same!='Same species')%>% select(value)),
                                  ci=0.95)
  # d_train_basal = effectsize::cohens_d(as.matrix(all_dists %>% filter(space=='Basal latent') %>% filter(set=='Train') %>% 
  #                                           filter(is_same=='Same species') %>% select(value)),
  #                               as.matrix(all_dists %>% filter(space=='Basal latent') %>% filter(set=='Train')%>%
  #                                           filter(is_same!='Same species')%>% select(value)),
  #                               ci=0.95)
  # d_train = effectsize::cohens_d(as.matrix(all_dists %>% filter(space!='Basal latent')%>% filter(set=='Train') %>%
  #                                           filter(is_same=='Same species') %>% select(value)),
  #                               as.matrix(all_dists %>% filter(space!='Basal latent') %>% filter(set=='Train')%>% 
  #                                           filter(is_same!='Same species')%>% select(value)),
  #                               ci=0.95)
  
  all_dists_val_basal <- all_dists %>% filter(set!='Train') %>%  filter(space!='latent') %>%
    mutate(effsize = abs(d_val_basal$Cohens_d))
  all_dists_val <- all_dists %>% filter(set!='Train') %>%  filter(space=='latent') %>%
    mutate(effsize = abs(d_val$Cohens_d))
  # all_dists_train_basal <- all_dists %>% filter(set=='Train') %>%  filter(space!='latent') %>%
  #   mutate(effsize = abs(d_train_basal$Cohens_d))
  # all_dists_train <- all_dists %>% filter(set=='Train') %>%  filter(space=='latent') %>%
  #   mutate(effsize = abs(d_train$Cohens_d))
  all_dists <- rbind(all_dists_val,all_dists_val_basal)
  # all_dists <- rbind(all_dists_train,all_dists_train_basal,all_dists_val,all_dists_val_basal)
  all_dists  <- all_dists %>% mutate(effsize = paste0('Cohen`s d: ',round(effsize,3)))
  cohen_df <- distinct(all_dists %>% select(effsize,set,space))
  cohen_df <- cohen_df %>% mutate(effsize = str_split_fixed(effsize,' ',2)[,2]) #added 
  df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
                        filter(effsize!='d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% 
                        select(split,set,space,'Cohen`s d'))
  # df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
  #                       filter(effsize!='Cohen`s d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% 
  #                       select(split,set,space,'Cohen`s d'))
  # gc()
  
  violin_separation <- ggplot(all_dists, aes(x=space, y=value, fill = is_same)) + 
    geom_violin(position = position_dodge(width = 1),width = 1)+geom_boxplot(position = position_dodge(width = 1),width = 0.05,
                                                                             outlier.shape = NA)+
    scale_fill_discrete(name="Embedding distance distribution in the latent space",
                        labels=c("Random Signatures","Same species"))+
    ylim(0,2)+
    xlab("")+ylab("Cosine Distance")+ 
    theme(axis.ticks.x=element_blank(),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(),text = element_text(family = "Arial",size = 20),legend.position = "bottom")+
    theme_minimal(base_family = "Arial",base_size = 20) +
    geom_text(aes(x=space,y=max(all_dists  %>% select(value))+0.1, label=effsize),
              data=cohen_df ,inherit.aes = FALSE,size=5)#+facet_wrap(~ space)
  #stat_compare_means(aes(group=is_same), method = "t.test")
  violin_separation <- violin_separation + theme(legend.position = "bottom")
  #print(violin_separation)
  plotList[[i+1]] <- violin_separation
  
  saveRDS(plotList,'results/plotList_cell_checkpoint.rds')
  saveRDS(df_effsize,'results/df_effsize_cell_checkpoint.rds')
  
  # png(paste0('results/compare_pca/pca_cells_mi_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # pca_test_mi <- pca_visualize(embs_test_mi,scale=T,dim=3,show_plot = F)
  # print(pca_test_mi)
  # dev.off()
  # 
  # png(paste0('results/compare_pca/pca_cells_basal_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # pca_test <- pca_visualize(embs_test,scale=T,dim=3,space_annotation='basal latent',show_plot = F)
  # print(pca_test)
  # dev.off()
  # 
  # png(paste0('results/compare_pca/pca_train_cells_mi_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # pca_train_mi <- pca_visualize(embs_train_mi,scale=T,dim=3,show_plot = F)
  # print(pca_train_mi)
  # dev.off()
  # 
  # png(paste0('results/compare_pca/pca_train_cells_basal_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # pca_train <- pca_visualize(embs_train,scale=T,dim=3,space_annotation='basal latent',show_plot = F)
  # print(pca_train)
  # dev.off()
  # 
  # png(paste0('results/compare_tsne/tsne_cells_mi_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # tsne_test_mi <- tsne_visualize(embs_test_mi,dim=3,scale=T,normalize=F,show_plot = F)
  # print(tsne_test_mi)
  # dev.off()
  # 
  # png(paste0('results/compare_tsne/tsne_cells_basal_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # tsne_test <- tsne_visualize(embs_test,dim=3,scale=T,normalize=F,space_annotation='basal latent',show_plot = F)
  # print(tsne_test)
  # dev.off()
  # 
  # png(paste0('results/compare_tsne/tsne_train_cells_mi_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # tsne_train_mi <- tsne_visualize(embs_train_mi,dim=3,scale=T,normalize=F,show_plot = F)
  # print(tsne_train_mi)
  # dev.off()
  # 
  # png(paste0('results/compare_tsne/tsne_train_cells_basal_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # tsne_train <- tsne_visualize(embs_train,dim=3,scale=T,normalize=F,space_annotation='basal latent',show_plot = F)
  # print(tsne_train)
  # dev.off()
  # 
  # 
  # png(paste0('results/compare_umap/umap_cells_mi_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # umap_test_mi <- umap_visualize(embs_test_mi,dim=2,n_neighbors=15,show_plot = F)
  # print(umap_test_mi)
  # dev.off()
  # 
  # png(paste0('results/compare_umap/umap_cells_basal_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # umap_test <- umap_visualize(embs_test,dim=2,n_neighbors=15,space_annotation='basal latent',show_plot = F)
  # print(umap_test)
  # dev.off()
  # 
  # png(paste0('results/compare_umap/umap_train_cells_mi_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # umap_train_mi <- umap_visualize(embs_train_mi,dim=2,n_neighbors=15,show_plot = F)
  # print(umap_train_mi)
  # dev.off()
  # 
  # png(paste0('results/compare_umap/umap_train_cells_basal_split',i,'.png'),width=10,height = 10,units = "in",res=300)
  # umap_train <- umap_visualize(embs_train,dim=2,n_neighbors=15,space_annotation='basal latent',show_plot = F)
  # print(umap_train)
  # dev.off()
  
  
  message(paste0('Done split ',i))
}

plotList <- readRDS('results/plotList_checkpoint.rds')
df_effsize <- readRDS('results/df_effsize_checkpoint.rds')

colnames(df_effsize)[4] <- 'value'
df_effsize <- df_effsize %>% mutate(space=ifelse(space=='Basal latent','basal latent',space))
df_effsize$space <- factor(df_effsize$space,levels = c("latent","basal latent"))
p  <- ggplot(df_effsize ,aes(x=space,y=value,fill=space)) + 
  geom_boxplot() + ggtitle('Species embeddings separation')+
  ylab('Cohen`s d')+
  stat_compare_means(size=10) + #,label = "p.signif"
  theme(panel.background = element_rect(fill = "white",
                                        colour = "white",
                                        linewidth = 0.5, linetype = "solid"),
        panel.grid.major = element_line(linewidth = 1, linetype = 'solid',
                                        colour = "#EEEDEF"), 
        panel.grid.minor = element_line(linewidth = 1, linetype = 'solid',
                                        colour = "#EEEDEF"),
        axis.text = element_text(family='Arial',face='bold'),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),#plot.title = element_text(hjust=1),
        text = element_text(family = "Arial",size = 34),legend.position = "bottom")+
  facet_wrap(~ set)
print(p)
ggsave(
  'results/cpa_compare_cohensd_basal_and_composed_space_species.eps',
  plot = p,
  device = cairo_ps,
  scale = 1,
  width = 9,
  height = 9,
  units = "in",
  dpi = 600,
)
setEPS()
postscript('results/cpa_compare_cohensd_basal_and_composed_space_species.eps',width = 9,height = 9)
print(p)
dev.off()
# png(file="results/samespecies_vs_random_cosine_embs_comparison_latent_celltypes.png",width=10,height=16,units = "in",res=600)
# p <- ggarrange(plotlist=plotList,ncol=2,nrow=5,common.legend = TRUE,legend = 'bottom',
#                labels = paste0(rep('Split ',10),seq(1,10)),
#                font.label = list(size = 10, color = "black", face = "plain", family = NULL),
#                hjust=-0.15,vjust = 0.7)
# 
# annotate_figure(p, top = text_grob("Embedding distance distribution in the composed latent space for cell-type effect", 
#                                    color = "black",face = 'plain', size = 14))
# dev.off()

### Plot distance distributions for two extreme cases of observed difference
diff <- NULL
colnames(df_effsize) <- c("split","set","space","value")
for (i in 1:10){
  tmp <- df_effsize %>% filter(split==i)
  diff[i] <- as.matrix(tmp %>% filter(set!='Train') %>% filter(space=='latent') %>% 
                                select(value))[,1]-as.matrix(tmp %>% filter(set!='Train') %>% filter(space!='latent') %>% 
                                                               select(value))
}
ind_max <- which.max(diff)
ind_min <- which.min(diff)

## For only those splits visualize distance distributions and PCA or/and tsne------------
projectionPlot <- NULL
list_visualize <- NULL
df_effsize <- data.frame()
j <- 1
for (i in c(ind_min,ind_max)){
  message(paste0('Start split ',i))
  embs_test_mi <- rbind(data.table::fread(paste0('results/embs/validation/valEmbs_',i-1,'_human.csv'),header = T) %>% mutate(species='human') %>%
                          select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('human_',id)),
                        data.table::fread(paste0('results/embs/validation/valEmbs_',i-1,'_mouse.csv'),header = T) %>% mutate(species='mouse') %>%
                          select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('mouse_',id))) %>% unique()
  embs_test <- rbind(data.table::fread(paste0('results/embs/validation/valEmbs_base_',i-1,'_human.csv'),header = T) %>% mutate(species='human')%>%
                       select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('human_',id)),
                     data.table::fread(paste0('results/embs/validation/valEmbs_base_',i-1,'_mouse.csv'),header = T) %>% mutate(species='mouse')%>%
                       select(-V1) %>% rownames_to_column('id') %>% mutate(id=paste0('mouse_',id))) %>% unique()
  embs_test_mi <- embs_test_mi %>% mutate(cell_type=ifelse(cell_type==1,'immune',
                                                           ifelse(cell_type==2,'mesenchymal',
                                                                  ifelse(cell_type==3,'epithelial',
                                                                         ifelse(cell_type==4,'endothelial','stem cell')))))
  embs_test <- embs_test %>% mutate(cell_type=ifelse(cell_type==1,'immune',
                                                     ifelse(cell_type==2,'mesenchymal',
                                                            ifelse(cell_type==3,'epithelial',
                                                                   ifelse(cell_type==4,'endothelial','stem cell')))))
  dist_test_mi <- samples_separation(embs_test_mi,
                                     compare_level='species',
                                     metric = 'cosine',
                                     show_plot = F)
  dist_test_mi <- dist_test_mi %>% mutate(space='latent')
  dist_test_mi <- dist_test_mi %>% mutate(set='Validation')
  
  
  dist_test <- samples_separation(embs_test,
                                  compare_level='species',
                                  metric = 'cosine',
                                  show_plot = F)
  dist_test <- dist_test %>% mutate(space='Basal latent')
  dist_test <- dist_test %>% mutate(set='Validation')
  
  all_dists <- bind_rows(dist_test_mi,dist_test)

  d_val_basal = effectsize::cohens_d(as.matrix(all_dists %>% filter(space=='Basal latent') %>% filter(set!='Train') %>% 
                                                 filter(is_same=='Same species') %>% select(value)),
                                     as.matrix(all_dists %>% filter(space=='Basal latent')%>% filter(set!='Train') %>%
                                                 filter(is_same!='Same species')%>% select(value)),
                                     ci=0.95)
  d_val = effectsize::cohens_d(as.matrix(all_dists %>% filter(space!='Basal latent')%>% filter(set!='Train') %>%
                                           filter(is_same=='Same species') %>% select(value)),
                               as.matrix(all_dists %>% filter(space!='Basal latent') %>% filter(set!='Train')%>%
                                           filter(is_same!='Same species')%>% select(value)),
                               ci=0.95)
  
  all_dists_val_basal <- all_dists %>% filter(set!='Train') %>%  filter(space!='latent') %>%
    mutate(effsize = abs(d_val_basal$Cohens_d))
  all_dists_val <- all_dists %>% filter(set!='Train') %>%  filter(space=='latent') %>%
    mutate(effsize = abs(d_val$Cohens_d))
  all_dists <- rbind(all_dists_val,all_dists_val_basal)
  all_dists  <- all_dists %>% mutate(effsize = paste0('Cohen`s d: ',round(effsize,3)))
  cohen_df <- distinct(all_dists %>% select(effsize,set,space))
  cohen_df <- cohen_df %>% mutate(effsize = str_split_fixed(effsize,' ',2)[,2]) #added 
  df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
                        filter(effsize!='d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% 
                        select(split,set,space,'Cohen`s d'))
  # df_effsize <- rbind(df_effsize,cohen_df %>% mutate(split=i+1) %>% mutate(effsize=strsplit(effsize,': ')) %>% unnest(effsize) %>%
  #                       filter(effsize!='Cohen`s d') %>% mutate('Cohen`s d'=as.numeric(effsize)) %>% 
  #                       select(split,set,space,'Cohen`s d'))
  all_dists <- all_dists %>% mutate()
  
  violin_separation <- ggplot(all_dists, aes(x=space, y=value, fill = is_same)) + 
    geom_violin(position = position_dodge(width = 1),width = 1)+
    geom_boxplot(position = position_dodge(width = 1),width = 0.05,
                 outlier.shape = NA)+
    scale_fill_discrete(name="Embeddings` distance distributions",
                        labels=c("Random Signatures","Same cell-type"))+
    ylim(0,2)+
    xlab("")+ylab("Cosine Distance")+ 
    theme_minimal(base_family = "Arial",base_size = 34) +
    theme(text = element_text(family = "Arial",size = 34),
          axis.ticks.x=element_blank(),
          axis.text = element_text(family = "Arial",face = 'bold',size=36),
          axis.title.y = element_text(family = "Arial",face = 'bold',size=32),
          legend.spacing.x = unit(5,'mm'),
          legend.title = element_text(family = "Arial",size = 26) ,
          legend.position = "bottom")+
    geom_text(aes(x=space,y=max(all_dists  %>% select(value))+0.1, label=effsize),
              data=cohen_df ,inherit.aes = FALSE,size=12,fontface='bold')
  violin_separation <- violin_separation + theme(legend.position = "bottom")
  #print(violin_separation)
  list_visualize[[j]] <- violin_separation
  # pca_test_mi <- pca_visualize(embs_test_mi,scale=T,dim=3,show_plot = F)
  # print(pca_test_mi)
  # 
  # pca_test <- pca_visualize(embs_test,scale=T,dim=3,space_annotation='basal latent',show_plot = F)
  # print(pca_test)
  # 
  # tsne_test_mi <- tsne_visualize(embs_test_mi,dim=3,scale=T,normalize=F,show_plot = F)
  # print(tsne_test_mi)
  # 
  # tsne_test <- tsne_visualize(embs_test,dim=3,scale=T,normalize=F,space_annotation='basal latent',show_plot = F)
  # print(tsne_test)
  
  j <- j +1
  message(paste0('Done split ',i))
}
#list_visualize <- plotList[c(ind_min,ind_max)]
p <- ggarrange(plotlist=list_visualize,ncol=2,nrow=1,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',2),c(ind_min,ind_max)),
               font.label = list(size = 30, color = "black", face = "plain", family = 'Arial'),
               hjust=-0.15,vjust = 1.1)

annotate_figure(p, top = text_grob("Cosine distance distribution in the latent spaces", 
                                   color = "black",face = 'plain', size = 32))
ggsave(
  'results/cpa_compare_distance_basal_and_composed_space_celltypes.eps', 
  device = cairo_ps,
  scale = 1,
  width = 14,
  height = 9,
  units = "in",
  dpi = 600,
)
