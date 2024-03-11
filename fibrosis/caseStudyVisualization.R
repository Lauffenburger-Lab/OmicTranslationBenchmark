library(tidyverse)
library(readxl)
library(ggplot2)
library(ggsignif)
library(ggpubr)
library(factoextra)
library(ggridges)
library(Rtsne)
library(gg3D)
set.seed(42)

### get a PCA and a tsne for every fold
folds <- 10
plotList <- NULL
for (i in 1:folds){
  #Load embeddings
  z_human <- data.table::fread(paste0('../../../Fibrosis Species Translation/human lung fibrosis/embs/embeddings_case_study/latent_human4translation_',i-1,'.csv')) %>% column_to_rownames('V1')
  latent_dim <- ncol(z_human)-3
  z_human <- z_human %>% mutate(species = 'human')
  colnames(z_human)[c(514,515)] <- c("cell_type","specific_cell")
  z_mouse <- data.table::fread(paste0('../../../Fibrosis Species Translation/human lung fibrosis/embs/embeddings_case_study/latent_mouse4translation_',i-1,'.csv')) %>% column_to_rownames('V1')
  z_mouse <- z_mouse %>% mutate(species = 'mouse')
  z_mouse <- z_mouse %>% select(-predicted_diagnosis)
  colnames(z_mouse)[c(514,515)] <- c("cell_type","specific_cell")
  z_latent <- rbind(z_human,z_mouse)
  
  ### Dimensionality reduction and visualization
  pca <- prcomp(z_latent[,1:latent_dim],center = T)
  fviz_eig(pca, addlabels = TRUE,ncp = 15)
  df_pca<- pca$x[,1:3]
  df_pca <- as.data.frame(df_pca)
  colnames(df_pca) <- c('PC1','PC2','PC3')
  df_pca <- df_pca %>% mutate(species = z_latent$species)
  df_pca$species <- factor(df_pca$species)
  df_pca <- df_pca %>% mutate(fibrosis = z_latent$original_diagnosis)
  df_pca$fibrosis <- factor(df_pca$fibrosis)
  df_pca <- df_pca %>% mutate(cell = z_latent$specific_cell)
  df_pca$cell <- factor(df_pca$cell)
  df_pca$cell_type <- factor(z_latent$cell_type)
  pca_plot <- ggplot(df_pca,aes(PC1,PC2)) +
    geom_point(aes(color=cell),size=2)+
    ggtitle("PCA visualization of the latent space") +
    xlab(paste0('PC1'))+ ylab(paste0('PC2'))+theme_minimal()+
    theme(text = element_text("Arial",size = 20),
          legend.position = 'right',
          plot.title = element_text(hjust=0.5,vjust=2,size=20))
  # pca_plot <- ggplot(df_pca,aes(x=PC1,y=PC2,z=PC3,col=cell))+
  #   ggtitle("PCA visualization of the latent space") +
  #   theme_void() +
  #   labs_3D(labs=c("PC1", "PC2", "PC3"),
  #           angle=c(0,0,0),
  #           hjust=c(0,2,2),
  #           vjust=c(2,2,-1))+
  #   axes_3D(phi=30) +
  #   stat_3D()+
  #   theme(text = element_text(size=20,family = 'Arial'),plot.title = element_text(hjust = 0.5),
  #         legend.text=element_text(size=20))
  print(pca_plot)
  # plotList[[i]] <- pca_plot
  
  ## perform tsne
  # perpl = DescTools::RoundTo(sqrt(nrow(z_latent)), multiple = 5, FUN = round)
  perpl <- 50
  emb_size = ncol(z_latent) -4
  set.seed(42)
  tsne_all <- Rtsne(z_latent[,1:latent_dim], 
                    dims = 3, perplexity=perpl, 
                    verbose=T, max_iter = 1000,
                    initial_dims = 10,
                    num_threads = 15)
  df_tsne <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2], V3 =tsne_all$Y[,3])
  colnames(df_tsne) <- c('tSNE-1','tSNE-2','tSNE-3')
  df_tsne <- df_tsne %>% mutate(species = z_latent$species)
  df_tsne$species <- factor(df_tsne$species)
  df_tsne <- df_tsne %>% mutate(fibrosis = z_latent$original_diagnosis)
  df_tsne$fibrosis <- factor(df_tsne$fibrosis)
  df_tsne <- df_tsne %>% mutate(cell = z_latent$specific_cell)
  df_tsne$cell <- factor(df_tsne$cell)
  tsne_plot <- ggplot(df_tsne,aes(`tSNE-1`,`tSNE-2`)) +
    geom_point(aes(color=cell),size=2)+
    ggtitle("t-SNE visualization of the latent space") +
    xlab(paste0('PC1'))+ ylab(paste0('PC2'))+theme_minimal()+
    theme(text = element_text("Arial",size = 20),
          legend.position = 'right',
          plot.title = element_text(hjust=0.5,vjust=2,size=20))
  # tsne_plot <- ggplot(df_tsne,aes(x=`tSNE-1`,y=`tSNE-2`,z=`tSNE-3`,col=cell))+
  #   ggtitle("t-SNE visualization of the latent space") +
  #   theme_void() +
  #   labs_3D(labs=c("tSNE-1", "tSNE-2", "tSNE-3"),
  #           angle=c(0,0,0),
  #           hjust=c(0,2,2),
  #           vjust=c(2,2,-1))+
  #   axes_3D(phi=30) +
  #   stat_3D()+
  #   theme(text = element_text(size=20,family = 'Arial'),plot.title = element_text(hjust = 0.5),
  #         legend.text=element_text(size=20))
  print(tsne_plot)
  # plotList[[i]] <- tsne_plot

}
