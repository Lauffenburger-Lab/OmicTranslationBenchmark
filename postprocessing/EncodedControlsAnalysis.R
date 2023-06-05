library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpubr)
library(factoextra)
library(Rtsne)

### Load data and controls-----
control_latent_1 <- data.table::fread('../results/trained_embs_all/ControlsEmbs_landmarks_CPA_a375.csv') %>% unique() %>% 
  column_to_rownames('V1') %>% mutate(label='A375') %>% mutate(type='L1000 control')
control_latent_2 <- data.table::fread('../results/trained_embs_all/ControlsEmbs_landmarks_CPA_ht29.csv') %>% unique() %>%
  column_to_rownames('V1') %>% mutate(label='HT29') %>% mutate(type='L1000 control')
covariate <- data.table::fread('../results/trained_embs_all/CellCovariate_landmarks_CPA_a375_ht29.csv') %>% column_to_rownames('V1')
rownames(covariate) <- c('A375 trained weight','HT29 trained weight')
covariate$label <- c('A375','HT29')
covariate$type <- 'trained covariate'

### Load CCLE and combine-----
ccle_latent <- rbind(data.table::fread('../results/trained_embs_all/ControlsCCLE_landmarks_CPA_a375.csv') %>% column_to_rownames('V1'),
                     data.table::fread('../results/trained_embs_all/ControlsCCLE_landmarks_CPA_ht29.csv') %>% column_to_rownames('V1'))
ccle_latent$label <- c('A375','HT29')
ccle_latent <- ccle_latent %>% mutate(type='CCLE sequence')
  
### Visualize--------
controls <- rbind(control_latent_1,control_latent_2,covariate,ccle_latent)

## PCA
pca <- prcomp(controls[,1:292],scale=F)
fviz_eig(pca,ncp=30)
df_pca<- pca$x[,1:2]
df_pca <- as.data.frame(df_pca)
df_pca$label <- controls$label
df_pca$type <- controls$type

p <- ggplot(df_pca,aes(x=PC1,y=PC2,color=label,shape=type)) + geom_point(size=4.5) +
  ggtitle('Scatterplot of controls in the PCA space') +
  theme_gray(base_family = "Arial",base_size = 28)+
  theme(plot.title = element_text(hjust = 0.5,family = "Arial",size=28),
        legend.title = element_text(family = "Arial",size=18),
        legend.text = element_text(family = "Arial",size=18))
png('../figures/pca_controlsa375_ht29_landmarks.png',width=9,height=6,units = "in",res = 600)
print(p)
dev.off()

print(p)
ggsave(
  '../figures/pca_controlsa375_ht29_landmarks.eps', 
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)

#### Repeat for pc3 and ha1e-------
control_latent_1 <- data.table::fread('../results/trained_embs_all/ControlsEmbs_landmarks_CPA_pc3.csv') %>% unique() %>% 
  column_to_rownames('V1') %>% mutate(label='PC3') %>% mutate(type='L1000 control')
control_latent_2 <- data.table::fread('../results/trained_embs_all/ControlsEmbs_landmarks_CPA_ha1e.csv') %>% unique() %>%
  column_to_rownames('V1') %>% mutate(label='HA1E') %>% mutate(type='L1000 control')
covariate <- data.table::fread('../results/trained_embs_all/CellCovariate_landmarks_CPA_pc3_ha1e.csv') %>% column_to_rownames('V1')
rownames(covariate) <- c('PC3 trained weight','HA1E trained weight')
covariate$label <- c('PC3','HA1E')
covariate$type <- 'trained covariate'

### Load CCLE and combine
ccle_latent_2 <- rbind(data.table::fread('../results/trained_embs_all/ControlsCCLE_landmarks_CPA_pc3.csv') %>% column_to_rownames('V1'),
                     data.table::fread('../results/trained_embs_all/ControlsCCLE_landmarks_CPA_ha1e.csv') %>% column_to_rownames('V1'))
ccle_latent_2$label <- c('PC3','HA1E')
ccle_latent_2 <- ccle_latent_2 %>% mutate(type='CCLE sequence')

### Visualize
controls_2 <- rbind(control_latent_1,control_latent_2,covariate,ccle_latent_2)

## PCA
pca <- prcomp(controls_2[,1:292],scale=F)
fviz_eig(pca,ncp=30)
df_pca<- pca$x[,1:2]
df_pca <- as.data.frame(df_pca)
df_pca$label <- controls_2$label
df_pca$type <- controls_2$type

p <- ggplot(df_pca,aes(x=PC1,y=PC2,color=label,shape=type)) + geom_point(size=4.5) +
  ggtitle('Scatterplot of controls in the PCA space') +
  theme_gray(base_family = "Arial",base_size = 28)+
  theme(plot.title = element_text(hjust = 0.5,family = "Arial",size=28),
        legend.title = element_text(family = "Arial",size=18),
        legend.text = element_text(family = "Arial",size=18))
png('../figures/pca_controlspc3_ha1e_landmarks.png',width=9,height=6,units = "in",res = 600)
print(p)
dev.off()
print(p)
ggsave(
  '../figures/pca_controlspc3_ha1e_landmarks.eps', 
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)

### Combine both ----------------------
all_controls <- rbind(controls,controls_2)
## PCA
pca <- prcomp(all_controls[,1:292],scale=F)
fviz_eig(pca,ncp=30)
df_pca<- pca$x[,1:2]
df_pca <- as.data.frame(df_pca)
df_pca$label <- all_controls$label
df_pca$type <- all_controls$type
p <- ggplot(df_pca,aes(x=PC1,y=PC2,color=label,shape=type)) + geom_point(size=4.5) +
  ggtitle('Scatterplot of controls in the PCA space') +
  theme_gray(base_family = "Arial",base_size = 28)+
  theme(plot.title = element_text(hjust = 0.5,family = "Arial",size=28),
        legend.title = element_text(family = "Arial",size=18),
        legend.text = element_text(family = "Arial",size=18))
png('../figures/pca_controlsall_cells_landmarks.png',width=9,height=6,units = "in",res = 600)
print(p)
dev.off()
print(p)
ggsave(
  '../figures/pca_controlsall_cells_landmarks.eps', 
  device = cairo_ps,
  scale = 1,
  width = 9,
  height = 6,
  units = "in",
  dpi = 600,
)

## t-SNE
perpl = DescTools::RoundTo(sqrt(nrow(all_controls)), multiple = 5, FUN = round)
# perpl=5
init_dim = 5
iter = 1000
emb_size = ncol(all_controls)-2
set.seed(42)
tsne <- Rtsne(all_controls[1:emb_size],
              dims = 2, perplexity=perpl, normalize_input = F,
              verbose=TRUE, max_iter = iter,initial_dims = init_dim,
              check_duplicates = F)
df_tsne <- data.frame(V1 = tsne$Y[,1], V2 =tsne$Y[,2])
rownames(df_tsne) <- rownames(all_controls)
df_tsne <- as.data.frame(df_tsne)
df_tsne$label <- all_controls$label
df_tsne$type <- all_controls$type

png('../figures/tsne_controlsall_cells_landmarks.png',width=9,height=6,units = "in",res = 600)
ggplot(df_tsne,aes(x=V1,y=V2,color=label,shape=type)) + geom_point(size=4.5) +
  ggtitle('Scatterplot of controls in the t-SNE space') +
  xlab('Dim 1') + ylab('Dim 2') +
  theme_gray(base_family = "Arial",base_size = 28)+
  theme(plot.title = element_text(hjust = 0.5,family = "Arial",size=28),
        legend.title = element_text(family = "Arial",size=18),
        legend.text = element_text(family = "Arial",size=18))
dev.off()
ggplot(df_tsne,aes(x=V1,y=V2,color=label,shape=type)) + geom_point(size=4.5) +
  ggtitle('Scatterplot of controls in the t-SNE space') +
  xlab('Dim 1') + ylab('Dim 2') +
  theme_gray(base_family = "Arial",base_size = 28)+
  theme(plot.title = element_text(hjust = 0.5,family = "Arial",size=28),
        legend.title = element_text(family = "Arial",size=18),
        legend.text = element_text(family = "Arial",size=18))
ggsave(
  '../figures/tsne_controlsall_cells_landmarks.eps', 
  device = cairo_ps,
  scale = 1,
  width = 9,
  height = 6,
  units = "in",
  dpi = 600,
)
