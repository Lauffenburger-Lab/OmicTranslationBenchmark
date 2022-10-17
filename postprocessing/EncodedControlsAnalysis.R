library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpubr)
library(factoextra)
library(Rtsne)

### Load data and controls-----
control_latent_1 <- data.table::fread('../results/trained_embs_all/ControlsEmbs_CPA_a375.csv') %>% unique() %>% 
  column_to_rownames('V1') %>% mutate(label='A375') %>% mutate(type='L1000 control')
control_latent_2 <- data.table::fread('../results/trained_embs_all/ControlsEmbs_CPA_ht29.csv') %>% unique() %>%
  column_to_rownames('V1') %>% mutate(label='HT29') %>% mutate(type='L1000 control')
covariate <- data.table::fread('../results/trained_embs_all/CellCovariate_CPA_a375_ht29.csv') %>% column_to_rownames('V1')
rownames(covariate) <- c('A375 trained weight','HT29 trained weight')
covariate$label <- c('A375','HT29')
covariate$type <- 'trained covariate'

### Load CCLE and combine-----
ccle_latent <- rbind(data.table::fread('../results/trained_embs_all/ControlsCCLE_CPA_a375.csv') %>% column_to_rownames('V1'),
                     data.table::fread('../results/trained_embs_all/ControlsCCLE_CPA_ht29.csv') %>% column_to_rownames('V1'))
ccle_latent$label <- c('A375','HT29')
ccle_latent <- ccle_latent %>% mutate(type='CCLE sequence')
  
### Visualize--------
controls <- rbind(control_latent_1,control_latent_2,covariate,ccle_latent)

## PCA
pca <- prcomp(controls[,1:1024],scale=F)
fviz_eig(pca,ncp=30)
df_pca<- pca$x[,1:2]
df_pca <- as.data.frame(df_pca)
df_pca$label <- controls$label
df_pca$type <- controls$type

png('../figures/pca_controls_ccle_a375_ht29.png',width=9,height=8,units = "in",res = 600)
ggplot(df_pca,aes(x=PC1,y=PC2,color=label,shape=type)) + geom_point(size=3.5) +
  ggtitle('Scatterplot of controls in the PCA space') +
  theme_gray(base_family = "serif",base_size = 20)+
  theme(plot.title = element_text(hjust = 0.5,size=20))
dev.off()

## t-SNE
#perpl = DescTools::RoundTo(sqrt(nrow(controls)), multiple = 5, FUN = round)
perpl=2
init_dim = 5
iter = 1000
emb_size = ncol(controls)-2
set.seed(42)
tsne <- Rtsne(controls[1:emb_size], 
              dims = 2, perplexity=perpl, normalize_input = F,
              verbose=TRUE, max_iter = iter,initial_dims = init_dim,
              check_duplicates = F)
df_tsne <- data.frame(V1 = tsne$Y[,1], V2 =tsne$Y[,2])
rownames(df_tsne) <- rownames(controls)
df_tsne <- as.data.frame(df_tsne)
df_tsne$label <- controls$label
df_tsne$type <- controls$type

png('../figures/tsne_controls_ccle_a375_ht29.png',width=9,height=8,units = "in",res = 600)
ggplot(df_tsne,aes(x=V1,y=V2,color=label,shape=type)) + geom_point(size=2) +
  ggtitle('Scatterplot of controls in the t-SNE space') + 
  xlab('Dim 1') + ylab('Dim 2') +
  theme_gray(base_family = "serif",base_size = 13)+
  theme(plot.title = element_text(hjust = 0.5,size=15))
dev.off()

