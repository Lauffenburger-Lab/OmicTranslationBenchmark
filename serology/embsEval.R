library(tidyverse)
library(readxl)
library(ggplot2)
library(ggsignif)
library(ggpubr)
library(factoextra)
library(ggridges)

### Load embeddings and data and combine them
#Load data
human_expr <- data.table::fread('data/human_exprs.csv') %>% column_to_rownames('V1')
human_hiv <- data.table::fread('data/human_metadata.csv') %>% column_to_rownames('V1')
primates_expr <- data.table::fread('data/primates_exprs.csv') %>% column_to_rownames('V1')
primates_nhp <- data.table::fread('data/primates_metadata.csv') %>% column_to_rownames('V1')

plotList <- NULL
min_effect <- 2
max_effect <- 0
for (i in 1:10){
  #Load embeddings
  z_human <- data.table::fread(paste0('results/embs/10fold/z_human_train_',i-1,'.csv')) %>% column_to_rownames('V1')
  latent_dim <- ncol(z_human)
  # z_human <- z_human %>% mutate(protected = human_hiv$protected)
  # z_human <- z_human %>% mutate(vaccinated = human_hiv$trt)
  z_human <- z_human %>% mutate(species = 'human')
  z_human_base <- data.table::fread(paste0('results/embs/10fold/z_human_base_train_',i-1,'.csv')) %>% column_to_rownames('V1')
  # z_human_base <- z_human_base %>% mutate(protected = human_hiv$protected)
  # z_human_base <- z_human_base %>% mutate(vaccinated = human_hiv$trt)
  z_human_base <- z_human_base %>% mutate(species = 'human')
  z_primates <- data.table::fread(paste0('results/embs/10fold/z_primates_train_',i-1,'.csv')) %>% column_to_rownames('V1')
  # z_primates <- z_primates %>% mutate(protected = primates_nhp$ProtectBinary)
  # z_primates <- z_primates %>% mutate(vaccinated = z_humanprimates_nhp$Vaccine)
  z_primates <- z_primates %>% mutate(species = 'primates')
  z_primates_base <- data.table::fread(paste0('results/embs/10fold/z_primates_base_train_',i-1,'.csv')) %>% column_to_rownames('V1')
  # z_primates_base <- z_primates_base %>% mutate(protected = primates_nhp$ProtectBinary)
  # z_primates_base <- z_primates_base %>% mutate(vaccinated = primates_nhp$Vaccine)
  z_primates_base <- z_primates_base %>% mutate(species = 'primates')
  
  z_latent <- rbind(z_human,z_primates)
  z_latent_base <- rbind(z_human_base,z_primates_base)
  
  data.table::fwrite(z_latent_base,paste0('results/embs/combined_10fold/latent_embeddings_global_',i,'.csv'))
  data.table::fwrite(z_latent,paste0('results/embs/combined_10fold/latent_embeddings_',i,'.csv'))
  
  ### Dimensionality reduction and visualization
  pca <- prcomp(z_latent_base[,1:latent_dim],center = T)
  #fviz_eig(pca, addlabels = TRUE,ncp = 15)
  df_pca<- pca$x[,1:3]
  df_pca <- as.data.frame(df_pca)
  colnames(df_pca) <- c('PC1','PC2','PC3')
  df_pca <- df_pca %>% mutate(protected = z_latent_base$protected)
  df_pca$protected <- factor(df_pca$protected)
  df_pca <- df_pca %>% mutate(vaccinated = z_latent_base$vaccinated)
  df_pca$vaccinated <- factor(df_pca$vaccinated)
  df_pca <- df_pca %>% mutate(species = z_latent_base$species)
  df_pca$species <- factor(df_pca$species)
  pca_plot <- ggplot(df_pca,aes(PC1,PC2)) +
    geom_point(aes(color=protected,shape=species,size=vaccinated),aplha=0.85)+
    scale_shape_manual(values = c(20,15))+
    scale_color_manual(values = c('#4878CF','#D65F5F'))+
    scale_size_manual(values = c(1.5,4))+
    #scale_alpha_manual(values = c(0.5,1))+
    ggtitle("PCA visualization of the global latent space") +
    xlab(paste0('PC1'))+ ylab(paste0('PC2'))+theme_minimal()+
    theme(text = element_text("Arial",size = 34),
          legend.position = 'right',
          plot.title = element_text(hjust=0.5,vjust=2,size=34))
  #print(pca_plot)
  plotList[[i]] <- pca_plot
  
  z_latent_base_gathered <- z_latent_base %>% gather('latent_variable','value',-species,-vaccinated,-protected)
  p_base <- ggplot(z_latent_base_gathered, aes(x = value, y = as.factor(latent_variable))) +
    geom_density_ridges(stat = "binline",bins = 100,alpha = 0.8,
                        fill = '#125b80',color='black') +
    ggtitle('Distribution of latent variables in global space')+
    xlab('value') + ylab('latent variable')+ theme(base_family = "Arial") +
    theme_pubr(base_family = "Arial",base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5))
  # p_base <- ggplot(z_latent_base,aes(x=species)) + geom_histogram(stat="count")
  #print(p_base)
  # print(sum(z_latent_base$species!='human')/sum(z_latent_base$species=='human'))
  
  mat <- as.matrix(df_pca[1:3])
  sim_matrix <- tcrossprod(mat) / (sqrt(rowSums(mat^2)) %*% t(sqrt(rowSums(mat^2))))
  dist <- 1 - sim_matrix
  dist <- dist[which(df_pca$protected==1),which(df_pca$protected!=1)]
  mu_dist_protect <- mean(dist)
  dist <- 1 - sim_matrix
  dist <- dist[which(df_pca$vaccinated==1),which(df_pca$vaccinated!=1)]
  mu_dist_vacc <- mean(dist)
  dist <- 1 - sim_matrix
  dist <- dist[which(df_pca$species=='human'),which(df_pca$species!='human')]
  mu_dist_spec <- mean(dist)
  effect <- (mu_dist_protect+mu_dist_vacc+mu_dist_spec)/3
  if (effect<=min_effect){
    min_effect <- effect
    min_effect_id <- i
  }
  if (effect>=max_effect){
    max_effect <- effect
    max_effect_id <- i
  }
}
list_visualize <- plotList[c(min_effect_id,max_effect_id)]
# # Visualize all subplots
# p <- ggarrange(plotlist=list_visualize,ncol=1,nrow=2,common.legend = TRUE,legend = 'bottom',
#                labels = paste0(rep('Split ',2),seq(1,10)),
#                font.label = list(size = 24, color = "black", face = "plain", family = 'Arial'),
#                hjust=-0.15)
# annotate_figure(p, top = text_grob("PCA visualization of the global latent space", 
#                                    color = "black",face = 'plain',family = 'Arial', size = 22))

print(plotList[10])
ggsave(
  'results/pca_2d_global_train_2000ep_subset.eps', 
  device = cairo_ps,
  scale = 1,
  width = 12,
  height = 6,
  units = "in",
  dpi = 600,
)
# png(file="results/pca_2d_global_train_2000ep_subset.png",width=18,height=12,units = "in",res=600)
# p <- ggarrange(plotlist=plotList,ncol=2,nrow=5,common.legend = TRUE,legend = 'bottom',
#                labels = paste0(rep('Split ',2),seq(1,10)),
#                font.label = list(size = 24, color = "black", face = "plain", family = 'Arial'),
#                hjust=-0.15)
# annotate_figure(p, top = text_grob("PCA visualization of the global latent space", 
#                                    color = "black",face = 'plain',family = 'Arial', size = 22))
# dev.off()

### Reapeat for the average of all models-------------
# #Load embeddings
# z_human <- data.table::fread('results/embs/z_human.csv') %>% column_to_rownames('V1')
# latent_dim <- ncol(z_human)
# z_human <- z_human %>% mutate(protected = human_hiv$protected)
# z_human <- z_human %>% mutate(vaccinated = human_hiv$trt)
# z_human <- z_human %>% mutate(species = 'human')
# z_human_base <- data.table::fread('results/embs/z_human_base.csv') %>% column_to_rownames('V1')
# z_human_base <- z_human_base %>% mutate(protected = human_hiv$protected)
# z_human_base <- z_human_base %>% mutate(vaccinated = human_hiv$trt)
# z_human_base <- z_human_base %>% mutate(species = 'human')
# z_primates <- data.table::fread('results/embs/z_primates.csv') %>% column_to_rownames('V1')
# z_primates <- z_primates %>% mutate(protected = primates_nhp$ProtectBinary)
# z_primates <- z_primates %>% mutate(vaccinated = primates_nhp$Vaccine)
# z_primates <- z_primates %>% mutate(species = 'primates')
# z_primates_base <- data.table::fread('results/embs/z_primates_base.csv') %>% column_to_rownames('V1')
# z_primates_base <- z_primates_base %>% mutate(protected = primates_nhp$ProtectBinary)
# z_primates_base <- z_primates_base %>% mutate(vaccinated = primates_nhp$Vaccine)
# z_primates_base <- z_primates_base %>% mutate(species = 'primates')
# 
# z_latent <- rbind(z_human,z_primates)
# z_latent_base <- rbind(z_human_base,z_primates_base)
# data.table::fwrite(z_latent_base,paste0('results/embs/latent_embeddings_global_',i,'.csv'))
# data.table::fwrite(z_latent,paste0('results/embs/latent_embeddings_',i,'.csv'))
# 
# ### Dimensionality reduction and visualization
# pca <- prcomp(z_latent_base[,1:latent_dim],center = T)
# fviz_eig(pca, addlabels = TRUE,ncp = 15)
# df_pca<- pca$x[,1:3]
# df_pca <- as.data.frame(df_pca)
# colnames(df_pca) <- c('PC1','PC2','PC3')
# df_pca <- df_pca %>% mutate(protected = z_latent_base$protected)
# df_pca$protected <- factor(df_pca$protected)
# df_pca <- df_pca %>% mutate(vaccinated = z_latent_base$vaccinated)
# df_pca$vaccinated <- factor(df_pca$vaccinated)
# df_pca <- df_pca %>% mutate(species = z_latent_base$species)
# df_pca$species <- factor(df_pca$species)
# pca_plot <- ggplot(df_pca,aes(PC1,PC2)) +geom_point(aes(col=protected,shape=species,alpha=vaccinated))+
#   scale_color_manual(values = c('#4878CF','#D65F5F'))+
#   scale_alpha_manual(values = c(0.5,1))+
#   ggtitle('') +
#   xlab(paste0('PC1'))+ ylab(paste0('PC2'))+theme_minimal()+
#   theme(text = element_text(size=16),plot.title = element_text(hjust = 0.5),
#         legend.text=element_text(size=16))
# print(pca_plot)





# plot_ly(df_pca) %>% add_trace(x = ~PC1, y = ~PC2, z = ~PC3,color = ~species,
#                               type = "scatter3d",
#                               mode = "markers",
#                               colors = c('#4878CF','#D65F5F'),
#                               showlegend = TRUE,
#                               size = 2
# )
# ## Perform tsne
# perpl = DescTools::RoundTo(sqrt(nrow(z_latent_base)), multiple = 5, FUN = round)
# #perpl= 50
# init_dim = 10
# iter = 1000
# emb_size = latent_dim
# set.seed(42)
# tsne_all <- Rtsne(z_latent_base[,1:latent_dim], 
#                   dims = 2, perplexity=perpl, 
#                   verbose=TRUE, 
#                   max_iter = iter,
#                   initial_dims = init_dim,
#                   check_duplicates = T,
#                   normalize = T,pca_scale = F,
#                   num_threads = 15)
# df_all <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
# df_all <- df_all %>% mutate(protected = z_latent_base$protected)
# df_all$protected <- factor(df_all$protected)
# df_all <- df_all %>% mutate(vaccinated = z_latent_base$vaccinated)
# df_all$vaccinated <- factor(df_all$vaccinated)
# df_all <- df_all %>% mutate(species = z_latent_base$species)
# df_all$species <- factor(df_all$species)
# 
# 
# gtsne <- ggplot(df_all, aes(V1, V2),alpha=0.2)+
#   geom_point(aes(col =protected,shape=species)) + labs(title="t-SNE plot of global latent space") + 
#   scale_color_manual(values = c('#4878CF','#D65F5F'))+
#   xlab('t-SNE 1') + ylab('t-SNE 2')+theme_minimal()+
#   theme(text = element_text(size=16),plot.title = element_text(hjust = 0.5),
#        legend.text=element_text(size=16))
# print(gtsne)
