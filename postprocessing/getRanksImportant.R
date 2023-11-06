library(tidyverse)
library(caret)
library(Rtsne)
library(factoextra)
library(ggplot2)
library(ggpubr)
library(ggpattern)
library(ggridges)
library(ggVennDiagram)
library(umap)
library(corrplot)
library(reshape2)
library(EGSEAdata)
library(fgsea)
library(topGO)
library(AnnotationDbi)
library(org.Hs.eg.db)
egsea.data(species = "human",returnInfo = TRUE)
human_keggs <- kegg.pathways$human$kg.sets
# setEPS(paper='a3',pointsize=3)

geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
lands <- geneInfo %>% filter(feature_space=='landmark')
lands <- as.character(lands$gene_id)
## Translation load------------
df1 <- data.table::fread('../results/Importance_results/important_scores_ha1e_to_pc3_allgenes_withclass_noabs.csv',header=T) %>% 
  column_to_rownames('V1')
df2 <- data.table::fread('../results/Importance_results/important_scores_pc3_to_ha1e_allgenes_withclass_noabs.csv',header=T) %>% 
  column_to_rownames('V1')
df <- 0.5*(df1+df2)
df <- abs(df)
rownames(df) <- rownames(df1)
# df <- read.csv('../results/Importance_results/important_scores_pc3_to_ha1e_per_sample_allgenes.csv')
df <- distinct(df)
df_ranked <- apply(-abs(df),1,rank)
df_ranked <- df_ranked/nrow(df)
pr1 <- 20
pr_sum <- round(length(which(diag(df_ranked*100)>=pr1))/nrow(df),4)*100
pr2 <- diag(df_ranked*100)[which(cumsum(diag(df_ranked*100))>=0.5)[1]]
df_aggragated <- apply(df_ranked,1,median)
top1000 <- names(df_aggragated[order(df_aggragated)])[1:1000]
#top1000 <- str_remove_all(top1000,'X')
df_lands <- df[which(rownames(df) %in% lands),which(colnames(df) %in% lands)]
df_ranked_lands <- apply(-abs(df_lands),1,rank)
df_ranked_lands <- df_ranked_lands/nrow(df_lands)
pr1 <- 20
pr_sum_lands <- round(length(which(diag(df_ranked_lands*100)>=pr1))/nrow(df_lands),4)*100
pr2_lands <- diag(df_ranked_lands*100)[which(cumsum(diag(df_ranked_lands*100))>=0.5)[1]]
# png('../figures/ranks_of_self_translate_pc3_ha1e_allgenes.png',width=12,height=8,units = "in",res=300)
# postscript('../figures/ranks_of_self_translate_pc3_ha1e_allgenes.eps',width=16,height=6,paper='letter')
# hist(diag(df_ranked*100),breaks = 40,
#      main= 'Distribution of self-gene ranks in ~10k genes',xlab='Percentage rank (%)',ylab='Counts',
#      cex.axis=3,cex.lab=3,cex.main = 3)
# abline(v=pr1,col="red",lwd=2,lty='dashed')
# text(pr1+20, 1000, paste0('~',pr_sum,'% of genes'),cex = 3)
# dev.off()
p <- ggplot(data.frame(ranks=diag(df_ranked*100)),aes(x=ranks)) + 
  geom_histogram(fill='#d3d3d3',color='black',bins = 40,lwd=1)+
  xlab('Percentage rank (%)') + ylab('Counts') + 
  ggtitle('Distribution of self-gene ranks in ~10k genes')+
  geom_vline(xintercept =pr1,color='red',linewidth = 1, lty = 'dashed')+
  annotate('text',x =pr1+30, y=1000,label= paste0('~',pr_sum,'% of genes'),size=15)+
  theme_minimal(base_family = "Arial",base_size = 36)+
  theme(plot.title = element_text(size=33,hjust = 0.5),
        axis.text = element_text(family = "Arial",size = 38))
print(p)
ggsave(
  '../figures/ranks_of_self_translate_pc3_ha1e_allgenes.png',
  plot = p,
  scale = 1,
  width = 12,
  height = 6,
  units = "in",
  dpi = 600,
)
postscript('../figures/ranks_of_self_translate_pc3_ha1e_allgenes.eps',
           width = 12,
           height = 6)
print(p)
dev.off()
#png('../figures/ranks_of_self_translate_pc3_ha1e_landmarks.png',width=12,height=8,units = "in",res=300)
# postscript('../figures/ranks_of_self_translate_pc3_ha1e_landmarks.eps',width=13,height=6,paper='letter')
# hist(diag(df_ranked_lands*100),breaks = 40,
#      main= 'Subset distribution of self-gene ranks in landmarks',xlab='Percentage rank (%)',ylab='Counts',
#      cex.axis=3,cex.lab=3,cex.main = 3)
# abline(v=pr1,col="red",lwd=2,lty='dashed')
# text(pr1+20, 130, paste0('~',pr_sum_lands,'% of genes'),cex = 3)
# dev.off()
#plot(ecdf(diag(df_ranked*100)),main='Cumulative probability distribution',xlab='Percentage rank (%)')
p <- ggplot(data.frame(ranks=diag(df_ranked_lands*100)),aes(x=ranks)) + 
  geom_histogram(fill='#d3d3d3',color='black',bins = 40,lwd=1)+
  xlab('Percentage rank (%)') + ylab('Counts') + 
  ggtitle('Subset distribution of self-gene ranks in landmarks')+
  geom_vline(xintercept =pr1,color='red',linewidth = 1, lty = 'dashed')+
  annotate('text',x =pr1+30, y=130,label= paste0('~',pr_sum_lands,'% of genes'),size=15)+
  theme_minimal(base_family = "Arial",base_size = 36)+
  theme(plot.title = element_text(size=33,hjust = 0.5),
        axis.text = element_text(family = "Arial",size = 38))
print(p)
ggsave(
  '../figures/ranks_of_self_translate_pc3_ha1e_landmarks.png',
  plot = p,
  scale = 1,
  width = 12,
  height = 6,
  units = "in",
  dpi = 600,
)
postscript('../figures/ranks_of_self_translate_pc3_ha1e_landmarks.eps',
           width = 12,
           height = 6)
print(p)
dev.off()
#df_aggragated <- apply(df_ranked,1,median)
#top1000 <- names(df_aggragated[order(df_aggragated)])[1:1000]

### Expression vs Importance-------------
cmap <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header=T) %>% column_to_rownames('V1')
cmap_median <- apply(cmap,2,median)
cmap_median <- as.data.frame(cmap_median)
cmap_median <- cmap_median %>% rownames_to_column('gene')

#plot cmap median expression of a gene vs importance score
df <- 0.5*(df1+df2)
df_self <- diag(as.matrix(df))
rownames(df) <- rownames(df1)
df <- apply(df,1,median)
df <- as.data.frame(df)
df <- df %>% rownames_to_column('gene')
df <- left_join(df,cmap_median)
#%>% mutate(cmap_median=abs(cmap_median)) %>% mutate(df=abs(df))
p <- ggscatter(df ,#%>% mutate(cmap_median=abs(cmap_median)) %>% mutate(abs(df)),
               x='cmap_median',y='df',rug = TRUE,
               alpha = 0.5,size=1,color = '#1f77b4',
               cor.coef=T,cor.coef.size = 5) + 
  geom_hline(yintercept = 0,color='black',lty=2,size=1) + geom_vline(xintercept = 0,color='black',lty=2,size=1)+
  #geom_smooth(color='black',lty=2)+
  xlab('median differential gene expression') + ylab('median gene importance score') + 
  ggtitle('Relationship between diffrerential gene expression and importance score')+
  theme_minimal(base_family = "Arial",base_size = 20)+
  theme(plot.title = element_text(size=20,hjust = 0.5))
print(p)
ggsave(
  '../figures/gex_vs_importance.eps',
  plot = p,
  device = cairo_ps,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)

## Load linear weights for linear regression model
# weights1 <- data.table::fread('../results/Importance_results/linear_gene_importance_pc3_to_ha1e.csv',header = T) %>% column_to_rownames('V1')
# weights2 <- data.table::fread('../results/Importance_results/linear_gene_importance_ha1e_to_pc3.csv',header = T)%>% column_to_rownames('V1')
# weights <- 0.5*(weights1+weights2)
# weights <- weights[df$gene,]
# weights <- scale(weights)
df <- 0.5*(abs(df1)+abs(df2))
rownames(df) <- rownames(df1)
df <- apply(df,1,median)
df <- as.data.frame(df)
df <- df %>% rownames_to_column('gene')
df <- left_join(df,cmap_median)

### Load reconstruction importance scores
# imp_enc_1 <- data.table::fread('../results/Importance_results/important_scores_pc3_to_encode.csv') %>%
#   dplyr::select(c('Gene_1'='V1'),all_of(var_1)) %>% column_to_rownames('Gene_1')
# imp_enc_1 <- imp_enc_1 %>% mutate(mean_imp=rowMeans(imp_enc_1))
# imp_enc_2 <- data.table::fread('../results/Importance_results/important_scores_ha1e_to_encode.csv')%>%
#   dplyr::select(c('Gene_2'='V1'),all_of(var_2)) %>% column_to_rownames('Gene_2')
# imp_enc_2 <- imp_enc_2 %>% mutate(mean_imp=rowMeans(imp_enc_2))
# ordered_1 <- rownames(imp_enc_1)[order(-imp_enc_1$mean_imp)]
# ordered_2 <- rownames(imp_enc_2)[order(imp_enc_2$mean_imp)]
# reconstruction_importance_pc3 <- data.table::fread('../results/Importance_results/important_reconstruction_scores_pc3.csv',header = T) %>%
#   column_to_rownames('V1')
# reconstruction_importance_pc3 <- reconstruction_importance_pc3 %>% dplyr::select(all_of(ordered_1[1:50]))
# reconstruction_importance_pc3 <- rowMeans(abs(reconstruction_importance_pc3))
# reconstruction_importance_pc3 <- as.data.frame(reconstruction_importance_pc3)
# reconstruction_importance_ha1e <- data.table::fread('../results/Importance_results/important_reconstruction_scores_ha1e.csv',header = T) %>%
#   column_to_rownames('V1')
# reconstruction_importance_ha1e <- reconstruction_importance_ha1e %>% dplyr::select(all_of(ordered_2[1:50]))
# reconstruction_importance_ha1e <- rowMeans(abs(reconstruction_importance_ha1e))
# reconstruction_importance_ha1e <- as.data.frame(reconstruction_importance_ha1e)
# print(all(rownames(reconstruction_importance_ha1e)==rownames(reconstruction_importance_pc3)))
# reconstruction_importance <- 0.5*(reconstruction_importance_ha1e+reconstruction_importance_pc3)
# colnames(reconstruction_importance) <- 'recon_score'

### PCA important genes
pca_cmap <- prcomp(cmap,center = T)
# fviz_screeplot(pca_cmap,ncp=20)
pca_cmap_results <- summary(pca_cmap)$importance
#pca_cmap_paired_results[3,5]
loadings_cmap <- pca_cmap$rotation
loadings_cmap <- loadings_cmap[,1:5]
pca_importance <- rowMeans(abs(loadings_cmap))
### Correlation calculation 
genes_inferred <- geneInfo %>% filter(feature_space=='best inferred') %>% filter(gene_type=='protein-coding')
cmap_corr_mat <- cor(cmap,method='spearman')
cmap_corr_mat <- cmap_corr_mat[lands,]
cmap_corr_mat <- cmap_corr_mat[,which(!(colnames(cmap_corr_mat) %in% lands))]
cmap_corr_mat <- rowMeans(cmap_corr_mat)
cmap_corr_mat_abs <- cor(abs(cmap),method='spearman')
cmap_corr_mat_abs <- cmap_corr_mat_abs[lands,]
cmap_corr_mat_abs <- cmap_corr_mat_abs[,which(!(colnames(cmap_corr_mat_abs) %in% lands))]
cmap_corr_mat_abs <- rowMeans(cmap_corr_mat_abs)
###
self_gene_corr_abs <- NULL
self_gene_corr <- NULL
spearman_corr <- NULL
spearman_corr_abs <- NULL
spearman_corr_rand <- NULL
spearman_corr_abs_rand <- NULL
# spearman_corr_linear <- NULL
# spearman_corr_abs_linear <- NULL
spearman_corr_abs_rf <- NULL
spearman_corr_rf <- NULL
spearman_corr_abs_pca <- NULL
spearman_corr_pca <- NULL
spearman_corr_abs_recon <- NULL
spearman_corr_recon <- NULL
for (i in 1:nrow(cmap)){
  gex <- cmap[i,]
  gex <- t(gex)
  gex <- gex[df$gene,]
  gex <- as.data.frame(gex)
  spearman_corr[i] <- cor(gex$gex, df$df, method = "spearman")
  spearman_corr_abs[i] <- cor(abs(gex$gex), abs(df$df), method = "spearman")
  spearman_corr_rand[i] <- cor(gex$gex[sample.int(n=nrow(gex),size = nrow(gex),replace = F)], df$df, method = "spearman")
  spearman_corr_abs_rand[i] <- cor(abs(gex$gex[sample.int(n=nrow(gex),size = nrow(gex),replace = F)]), abs(df$df), method = "spearman")
  # spearman_corr_linear[i] <- cor(gex$gex, weights, method = "spearman")
  # spearman_corr_abs_linear[i] <- cor(abs(gex$gex), abs(weights), method = "spearman")
  # spearman_corr_rf[i] <- cor(gex$gex, importance_rf, method = "spearman")
  # spearman_corr_abs_rf[i] <- cor(abs(gex$gex), abs(importance_rf), method = "spearman")
  self_gene_corr_abs[i] <- cor(abs(gex$gex), abs(df_self), method = "spearman")
  self_gene_corr[i] <- cor(gex$gex, df_self, method = "spearman")
  spearman_corr_abs_pca[i] <- cor(abs(gex$gex),rowMeans(abs(loadings_cmap)),method='spearman')
  spearman_corr_pca[i] <- cor(gex$gex,rowMeans(loadings_cmap),method='spearman')
  # spearman_corr_abs_recon[i] <- cor(abs(gex$gex),abs(reconstruction_importance$recon_score),method='spearman')
  # spearman_corr_recon[i] <- cor(gex$gex,reconstruction_importance$recon_score,method='spearman')
  if (i %% 100 == 0 | i==1){
    print(paste0('Finished sample ',i))
  }
}
df_corr <- data.frame(abs_spear = spearman_corr_abs,spear = spearman_corr)
# df_corr_linear <- data.frame(abs_spear = spearman_corr_abs_linear,spear = spearman_corr_linear)
df_corr_rand <- data.frame(abs_spear = spearman_corr_abs_rand,spear = spearman_corr_rand)
# df_corr_rf <- data.frame(abs_spear = spearman_corr_abs_rf,spear = spearman_corr_rf)
df_corr_gex_lands <- data.frame(abs_spear=cmap_corr_mat_abs,spear=cmap_corr_mat)
df_corr_self <- data.frame(abs_spear=self_gene_corr_abs,spear=self_gene_corr)
df_corr_pca <- data.frame(abs_spear=spearman_corr_abs_pca,spear=spearman_corr_pca)
# df_corr_recon <- data.frame(abs_spear=spearman_corr_abs_recon,spear=spearman_corr_recon)
df_corr_all <- rbind(df_corr %>% mutate(type = 'model'),
                     #df_corr_self %>% mutate(type = 'same gene-to-gene'),
                     df_corr_pca %>% mutate(type='PCA importance'),
                     #df_corr_recon %>% mutate(type='reconstruction importance'),
                     # df_corr_gex_lands %>% mutate(type = 'landmarks Gene Exprs.'),
                     #df_corr_rf %>% mutate(type = 'random forest'),
                     #df_corr_linear %>% mutate(type = 'linear'),
                     df_corr_rand %>% mutate(type = 'shuffled'))
#'#d3d3d3'
# p <- ggplot(df_corr_all,aes(x=abs_spear,fill=type)) + geom_density(aes(y=..scaled..),color='black',lwd=1,adjust=0.85) +
#   xlab('per sample Spearman`s correlation') + ylab('scaled density') + #ylab('Counts') +
#   ggtitle('Correlation between gene importance and expression')+
#   theme_minimal(base_family = "Arial",base_size = 36)+
#   theme(plot.title = element_text(size=33,hjust = 1),
#         axis.text = element_text(family = "Arial",size = 38))
model_vs_rand <- effectsize::cohens_d(df_corr$abs_spear,
                                      df_corr_rand$abs_spear,
                                      paired = T,
                                      ci=0.95)$Cohens_d
model_vs_pca <- effectsize::cohens_d(df_corr$abs_spear,
                                      df_corr_pca$abs_spear,
                                      paired = T,
                                      ci=0.95)$Cohens_d
df_corr_all$type <- factor(df_corr_all$type,levels = c('PCA importance','model','shuffled')) # ,'reconstruction importance'
p <- ggplot(df_corr_all,aes(x=abs_spear,fill=type)) + geom_histogram(color='black',bins = 50,lwd=1,alpha = 0.5,position="identity") +
  xlab('per sample Spearman`s correlation') + ylab('Counts') + 
  ggtitle('Correlation between gene importance and expression')+
  # geom_text(aes(x=abs_spear,y=count, label=effect_size),
  #           data=cof_results ,inherit.aes = FALSE,size=5, hjust = 0)+
  theme_minimal(base_family = "Arial",base_size = 36)+
  theme(plot.title = element_text(size=33,hjust = 1),
        axis.text = element_text(family = "Arial",size = 38),
        legend.position = 'top')
print(p)
ggsave(
  '../figures/gex_vs_importance_spearman.png',
  plot = p,
  scale = 1,
  width = 12,
  height = 6,
  units = "in",
  dpi = 600,
)
postscript('../figures/gex_vs_importance_spearman.eps',
           width=12,
           height=6)
ggplot(df_corr_all,aes(x=abs_spear,fill=type)) + geom_histogram(color='black',bins = 50,lwd=1,position="identity") +
  xlab('per sample Spearman`s correlation') + ylab('Counts') + 
  ggtitle('Correlation between gene importance and expression')+
  # geom_text(aes(x=abs_spear,y=count, label=effect_size),
  #           data=cof_results ,inherit.aes = FALSE,size=5, hjust = 0)+
  theme_minimal(base_family = "Arial",base_size = 36)+
  theme(plot.title = element_text(size=33,hjust = 1),
        axis.text = element_text(family = "Arial",size = 38),
        legend.position = 'top')
dev.off()
# From the important genes how many are also highly regulated in the cell line we translate from
FindPercentageIntersection <- function(grdRanks,DeXs,no_top=1000){
  #df_gene <- grdRanks
  tops <- names(grdRanks[order(grdRanks)])[1:no_top]
  percentage <- length(which(tops %in% DeXs))/length(tops)
  return(percentage)
}
cmap <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header=T) %>% column_to_rownames('V1')
sample_paired <- data.table::fread('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/paired_pc3_ha1e.csv',header=T) %>% column_to_rownames('V1')
cmap <- cmap[sample_paired$sig_id.x,]
cmap <- as.matrix(abs(cmap))
genes_regulated <- NULL
mean_perc <- NULL
sd_perc <- NULL
gc()

FindPercentageIntersection <- function(grdRanks,DeXs,no_top=1000){
  num_genes <- nrow(DeXs)
  grdRanked <- 1 * (grdRanks <= (top_num/num_genes))
  DeXs <- DeXs[names(grdRanked),]
  grdRanked <- replicate(ncol(DeXs),grdRanked)
  percentage <- grdRanked + DeXs
  percentage <- percentage == 2
  percentage <- 1 * percentage
  percentage <- apply(percentage,2,sum)
  percentage <- percentage/no_top
  return(percentage)
}
#### Use per sample analysis
top_nums  <- c(10,20,50,100,250,500,1000,2000)
df_results <- data.frame()
for (top_num in top_nums){
  gex <- apply(-t(cmap),2,rank)
  gex <- 1* (gex <= top_num)
  percentage_of_importants_in_top_regulated <- apply(df_ranked,2,FindPercentageIntersection,DeXs=gex,no_top=top_num)
  mean_perc <- apply(percentage_of_importants_in_top_regulated,1,mean)
  sd_perc <- apply(percentage_of_importants_in_top_regulated,1,sd)
  # combine in dataframe
  tmp <- data.frame(mu = mean_perc,
                   sd = sd_perc)
  tmp <- tmp %>% mutate(top_number = top_num)
  df_results <- rbind(df_results,tmp)
  print(paste0('Finished for top ',top_num,' genes'))
}
# saveRDS(df_results,'../results/Importance_results/overlap_important_regulated.rds')

df_results <- df_results %>% mutate(mu = 100* mu) %>% mutate(sd = 100* sd) 
p_imp_gex_suppl <- ggplot(df_results, aes(x = mu, y = as.factor(top_number))) +
  geom_density_ridges(stat = "binline",bins = 50,alpha = 0.8,fill = '#125b80',color='black') +
  xlab('average overlap (%)') + ylab('# top genes')+
  theme_pubr(base_family = "Arial",base_size = 24) +
  theme(text = element_text(family = 'Arial'))
print(p_imp_gex_suppl)
ggsave('../figures/figure4C_suppl.png',
       plot = p_imp_gex_suppl,
       height = 9,
       width = 9,
       units = 'in',
       dpi = 600)
postscript('../figures/figure4C_suppl.eps',width = 12,height = 8)
ggplot(df_results, aes(x = mu, y = as.factor(top_number))) +
  geom_density_ridges(stat = "binline",bins = 50,fill = '#125b80',color='black') +
  xlab('average overlap (%)') + ylab('# top genes')+
  theme_pubr(base_family = "Arial",base_size = 24) +
  theme(text = element_text(family = 'Arial'))
dev.off()
p_imp_gex <- ggplot(rbind(df_results %>% group_by(top_number) %>% mutate(avg = mean(mu)) %>% mutate(sigma = sd(mu)) %>% 
                    ungroup() %>% dplyr::select(avg,sigma,top_number) %>% unique() %>% mutate(metric = 'mean'),
                    df_results %>% group_by(top_number) %>% mutate(avg = mean(sd)) %>% mutate(sigma = sd(sd)) %>% 
                      ungroup() %>% dplyr::select(avg,sigma,top_number) %>% unique() %>% mutate(metric = 'standard deviation')) , 
                    aes(x = top_number, y = avg,color = metric)) +
  geom_point(size=1.5) +
  geom_line(linewidth = 0.5)+
  geom_errorbar(aes(ymin = avg - 2.576*sigma/sqrt(nrow(cmap)) , ymax = avg + 2.576*sigma/sqrt(nrow(cmap))), width = 20,linewidth = 0.75)+
  annotate('text',x=1200, y =11, label='Less than ~10% for looking up to ~100 top genes',size=6) +
  geom_hline(yintercept = 10,linewidth=1,color='black',linetype='dashed')+
  xlab('# top genes') + ylab('mean overlap (%)')+ 
  theme_pubr(base_family = "Arial",base_size = 24) + 
  theme(text = element_text(family = 'Arial'),
        legend.position = 'top')
print(p_imp_gex)
ggsave('../figures/figure4C.png',
       plot = p_imp_gex,
       height = 9,
       width = 9,
       units = 'in',
       dpi = 600)
postscript('../figures/figure4C.eps',height = 9,width = 9)
print(p_imp_gex)
dev.off()

### Importance analysis----
## Per sample analysis in all
percentage_of_lands_in_important <- NULL
for (j in 1:ncol(df_ranked)){
  df_gene <- df_ranked[,j]
  top1000 <- names(df_gene[order(df_gene)])[1:1000]
  percentage_of_lands_in_important[j] <- length(which(lands %in% top1000))/length(lands)
}

print(length(which(lands %in% top1000))/length(lands))

png('../figures/percentage_of_lands_in_1000important_pc3_to_ha1e.png',width=16,height=8,units = "in",res=300)
hist(percentage_of_lands_in_important*100,breaks = 40,
     main= 'Percentages of landmark genes present in top 1000 important genes',xlab='Percentage (%)')
dev.off()

library(ggVennDiagram)
library(ggplot2)

# List of items
# From per sample ranking genes
x <- list('Top 1000 important genes' = top1000, 'Landmark genes' = lands)

# 2D Venn diagram
png('../figures/venn_lands_top1000_to_translate_pc3_to_ha1e.png',width=16,height=8,units = "in",res=300)
ggVennDiagram(x, color = 1, lwd = 0.7) + 
  scale_fill_gradient(low = "#F4FAFE", high = "#4981BF")+
  theme(legend.position = "none")
dev.off()

# From the important genes how many are also highly regulated in A375
FindPercentageIntersection <- function(grdRanks,DeXs,no_top=1000){
  #df_gene <- grdRanks
  top1000 <- names(grdRanks[order(grdRanks)])[1:no_top]
  percentage <- length(which(top1000 %in% DeXs))/length(top1000)
  return(percentage)
}
cmap <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header=T) %>% column_to_rownames('V1')
sample_paired <- data.table::fread('../preprocessing/preprocessed_data/10fold_validation_spit/alldata/paired_pc3_ha1e.csv',header=T) %>% column_to_rownames('V1')
cmap <- cmap[sample_paired$sig_id.x,]
cmap <- as.matrix(abs(cmap))
genes_regulated <- NULL
mean_perc <- NULL
sd_perc <- NULL

## Check correlation between genes and see if genes not in important
# are correlated with some important
# Here I use the the genes genes importance summed accross all samples
gex <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header = T) %>% column_to_rownames('V1')
GeneCorr <- cor(gex)
GeneCorr <- GeneCorr[top1000,lands] #lands[which(!(lands %in% top1000))]

png('corr_of_importants_with_lands_pc3_to_ha1e.png',width=12,height=8,units = "in",res=300)
hist(GeneCorr,breaks = 50, freq = T,
     main = 'Pearson correlation between model-important genes and landmark genes',xlab='Pearson`s r')
dev.off()

library("RColorBrewer")
col <- colorRampPalette(brewer.pal(10, "RdBu"))(256)
png('corrHeat_of_importants_with_nonimp_lands_pc3_to_ha1e.png',width=12,height=8,units = "in",res=300)
heatmap(GeneCorr,scale = "none",col=col)
dev.off()

## Check correlation of important landamarks to non-important
GeneCorr <- GeneCorr[lands[which((lands %in% top1000))],lands[which(!(lands %in% top1000))]]

png('../figures/corr_of_lands_to_lands_pc3_to_ha1e.png',width=12,height=8,units = "in",res=300)
hist(GeneCorr,breaks = 50, freq = T,
     main = 'Pearson correlation between model-important and non-important landmark genes',xlab='Pearson`s r')
dev.off()

col <- colorRampPalette(brewer.pal(10, "RdBu"))(256)
png('../figures/corrHeat_of_lands_to_lands_pc3_to_ha1e.png',width=12,height=8,units = "in",res=300)
heatmap(GeneCorr,scale = "none",col=col)
dev.off()

### Importance scores to encode-----
library(tidyverse)
library(ggVennDiagram)
library(ggplot2)
gex <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header = T) %>% column_to_rownames('V1')
scores_1 <- data.table::fread('../results/Importance_results/important_scores_pc3_to_encode.csv',header = T) 
scores_1 <- distinct(scores_1)
rownames( scores_1 ) <- NULL
scores_1 <- scores_1 %>% column_to_rownames('V1')
scores_2 <- data.table::fread('../results/Importance_results/important_scores_ha1e_to_encode.csv',header = T) 
scores_2 <- distinct(scores_2)
rownames( scores_2 ) <- NULL
scores_2 <- scores_2 %>% column_to_rownames('V1')

### Find unimportant for encoding but good to explain variance of the data in the PC space
gene_size <- 10086
library(factoextra)
pca <- prcomp(gex,scale =F)
fviz_eig(pca,ncp=20)
tt <- summary(pca)
loadings <- pca$rotation[,1:400] # all of them if I want all the data
load_importance <- apply(abs(loadings),1,mean)
load_rankings <- rank(-load_importance) # wanting high contribution in explaining variance of all samples
#load_rankings[order(load_rankings)][1:3]
#load_rankings[which(names(load_rankings)=='5997')]/gene_size

score_importance_1 <- apply(scores_1,2,mean)
score_rankings_1 <- rank(score_importance_1) # wanting low importance to encode (filter out)
ranking_1 <- 0.5*(score_rankings_1+load_rankings)
ranking_1 <- ranking_1/gene_size
ranking_1[order(ranking_1)][1:10]


score_importance_2 <- apply(scores_2,2,mean)
score_rankings_2 <- rank(score_importance_2) # wanting low importance to encode (filter out)
ranking_2 <- 0.5*(score_rankings_2+load_rankings)
ranking_2 <- ranking_2/gene_size
ranking_2[order(ranking_2)][1:10]

# Find bellow 20% and exclude common
important1 <- ranking_1[which(ranking_1<=0.05)]
important2 <- ranking_2[which(ranking_2<=0.05)]
common <- Reduce(intersect,list(names(important1),names(important2)))
important1 <- important1[which(!(names(important1) %in% common))]
important2 <- important2[which(!(names(important2) %in% common))]

### t-SNE with all features and only the good ones
gex <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header = T) %>% column_to_rownames('V1')
library(Rtsne)
perpl = DescTools::RoundTo(sqrt(nrow(gex)), multiple = 5, FUN = round)
#Use the above formula to calculate perplexity (perpl). But if perplexity is too large for the number of data you have define manually
#perpl=2
init_dim = 20
iter = 1000
emb_size = ncol(gex)
set.seed(42)
tsne_all <- Rtsne(gex, 
                  dims = 2, perplexity=perpl, 
                  verbose=TRUE, max_iter = iter,
                  initial_dims = init_dim,check_duplicates = F,
                  normalize = F,pca_scale = F,
                  num_threads = 15)
df_tsne_allgenes <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
rownames(df_tsne_allgenes) <- rownames(gex)
colnames(df_tsne_allgenes) <- c('Dim1','Dim2')
df_tsne_allgenes  <- df_tsne_allgenes %>% mutate(cell='None')
ind1 <- grep('PC3',rownames(df_tsne_allgenes))
ind2 <- grep('HA1E',rownames(df_tsne_allgenes))
df_tsne_allgenes$cell[ind1] <- 'PC3'
df_tsne_allgenes$cell[ind2] <- 'HA1E'
tsne_plot_allgenes <- ggplot(df_tsne_allgenes,aes(Dim1,Dim2)) +geom_point(aes(col=cell))+
  ggtitle('t-SNE plot of transcriptomic data for 2 cell-lines') + xlab('Dim 1')+ ylab('Dim 2')+
  theme(text = element_text(size=13))
print(tsne_plot_allgenes)
png(paste0('../figures/tsne_pc3_ha1e_gex.png'),width=10,height = 10,units = "in",res=300)
print(tsne_plot_allgenes)
dev.off()

df_pca<- pca$x[,1:2]
df_pca <- as.data.frame(df_pca)
df_pca  <- df_pca %>% mutate(cell='None')
ind1 <- grep('PC3',rownames(df_pca))
ind2 <- grep('HA1E',rownames(df_pca))
df_pca$cell[ind1] <- 'PC3'
df_pca$cell[ind2] <- 'HA1E'
pca_plot <- ggplot(df_pca,aes(PC1,PC2)) +geom_point(aes(col=cell))+
  ggtitle('PCA plot of transcriptomic data for 2 cell-lines') + xlab('PC1')+ ylab('PC2')+
  theme(text = element_text(size=13))
print(pca_plot)
png(paste0('../figures/pca_pc3_ha1e_gex.png'),width=10,height = 10,units = "in",res=300)
print(pca_plot)
dev.off()

gex <- gex[,c(names(important1),names(important2))]
perpl = DescTools::RoundTo(sqrt(nrow(gex)), multiple = 5, FUN = round)
#Use the above formula to calculate perplexity (perpl). But if perplexity is too large for the number of data you have define manually
#perpl=2
init_dim = 5
iter = 1000
emb_size = ncol(gex)
set.seed(42)
tsne_all <- Rtsne(gex, 
                  dims = 2, perplexity=perpl, 
                  verbose=TRUE, max_iter = iter,
                  check_duplicates = F,normalize = F,
                  pca_scale = F,pca=T,init_dim=init_dim,
                  num_threads = 15)
df_tsne <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
rownames(df_tsne) <- rownames(gex)
colnames(df_tsne) <- c('Dim1','Dim2')
df_tsne  <- df_tsne %>% mutate(cell='None')
ind1 <- grep('PC3',rownames(df_tsne))
ind2 <- grep('HA1E',rownames(df_tsne))
df_tsne$cell[ind1] <- 'PC3'
df_tsne$cell[ind2] <- 'HA1E'
tsne_plot <- ggplot(df_tsne,aes(Dim1,Dim2)) +geom_point(aes(col=cell))+
  ggtitle('t-SNE plot of transcriptomic data for 2 cell-lines') + xlab('Dim 1')+ ylab('Dim 2')+
  theme(text = element_text(size=13))
print(tsne_plot)


## Build classifier only using those genes
gex <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header = T) %>% column_to_rownames('V1')
gex <- gex[,c(names(important1),names(important2))]
ind1 <- grep('PC3',rownames(gex))
ind2 <- grep('HA1E',rownames(gex))

gex <- gex %>% mutate(cell='None')
gex$cell[ind1] <- 'PC3'
gex$cell[ind2] <- 'HA1E'
#gex$cell <- factor(gex$cell,levels = c('HA1E', 'PC3'))

library(caret)
library(doRNG)
ctrl <- trainControl(method = "cv", number = 10)
train_gex <- sample_n(gex,1742)
test_gex <- gex[which(!(rownames(gex) %in% rownames(train_gex))),]
mdl <- train(cell ~ ., data = train_gex, method = "rf", trControl = ctrl,trace=T)
#mdl$results$Accuracy
mean(mdl[["resample"]][["Accuracy"]])
#summary(mdl)
# Evaluate precision and accuracy in test set
y <- predict(mdl,newdata = test_gex[,1:(ncol(test_gex)-1)])
confusionMatrix(factor(test_gex$cell,levels = c('HA1E', 'PC3')),y)
feature_imp <- varImp(mdl,scale=T)
feature_imp <- feature_imp[["importance"]]
p <- ggplot(gex,aes(x=`6194`,y=`1000`,col=cell)) + geom_point()+
  ggtitle('Scatter plot for 2 cell-lines')+
  theme(text = element_text(size=13))
print(p)


## Build classifier only using latent space embeddings
emb1 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_CPA_pc3.csv',header = T)) %>% column_to_rownames('V1')
emb1 <- emb1 %>% mutate(cell='PC3')
emb2 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_CPA_ha1e.csv',header = T)) %>% column_to_rownames('V1')
emb2 <- emb2 %>% mutate(cell='HA1E')

all_embs <- rbind(emb1,emb2)
all_embs <- all_embs[sample(1:nrow(all_embs)),]
all_embs <- all_embs %>% mutate(label=ifelse(cell=='PC3',1,0))
all_embs$label <- factor(all_embs$label,levels=c(0,1))
all_embs <- all_embs %>% dplyr::select(-cell)

library(caret)
ctrl <- trainControl(method = "cv", number = 10)
train_embs <- sample_n(all_embs,1991)
test_embs <- all_embs[which(!(rownames(all_embs) %in% rownames(train_embs))),]
mdl <- train(label ~ ., data = train_embs, method = "rf", trControl = ctrl,trace=T)
#mdl$results$Accuracy
mean(mdl[["resample"]][["Accuracy"]])
#summary(mdl)
# Evaluate precision and accuracy in test set
y <- predict(mdl,newdata = test_embs[,1:(ncol(test_embs)-1)])
#confusionMatrix(factor(test_embs$cell,levels = c('HA1E', 'PC3')),y)
confusionMatrix(test_embs$label,y)
feature_imp <- varImp(mdl,scale=T)
feature_imp <- feature_imp[["importance"]]
p <- ggplot(all_embs,aes(x=`z307`,y=`z992`,col=label)) + geom_point()+
  ggtitle('Scatter plot for 2 cell-lines')+
  theme(text = element_text(size=13))
print(p)

### Find important genes to encode into each latent variable important for classification------------------
importance_class_1 <- data.table::fread('../results/Importance_results/important_scores_to_classify_as_pc3.csv',header=T) %>% column_to_rownames('V1')
importance_class_2 <- data.table::fread('../results/Importance_results/important_scores_to_classify_as_ha1e.csv',header=T) %>% column_to_rownames('V1')
importance_class_1 <- apply(importance_class_1,2,mean)
importance_class_2 <- apply(importance_class_2,2,mean)

kmeans_class1 <-  kmeans(as.matrix(importance_class_1),centers = 3,iter.max = 100, nstart = 50)
df_class_1 <-  data.frame(latent_variable=names(importance_class_1),score = importance_class_1,cluster =kmeans_class1$cluster)
df_class_1_summary <- df_class_1 %>% group_by(cluster) %>% summarise(counts = n()) %>% arrange(counts)
cl1 <- df_class_1_summary$cluster[1]
cl2 <- df_class_1_summary$cluster[2]
hist(df_class_1$score[df_class_1$cluster==cl1])
hist(df_class_1$score[df_class_1$cluster==cl2])
if (max(df_class_1$score[df_class_1$cluster==cl1])<=min(df_class_1$score[df_class_1$cluster==cl2])){
  th <- mean(max(df_class_1$score[df_class_1$cluster==cl1]),min(df_class_1$score[df_class_1$cluster==cl2]))
} else{
  th <- mean(min(df_class_1$score[df_class_1$cluster==cl1]),max(df_class_1$score[df_class_1$cluster==cl2]))
}
important_1 <- df_class_1 %>% filter(score>=th) %>% dplyr::select(latent_variable)
important_1 <- important_1$latent_variable

kmeans_class2 <-  kmeans(as.matrix(importance_class_2),centers = 3,iter.max = 100, nstart = 50)
df_class_2 <-  data.frame(latent_variable=names(importance_class_2),score = importance_class_2,cluster =kmeans_class2$cluster)
df_class_2_summary <- df_class_2 %>% group_by(cluster) %>% summarise(counts = n()) %>% arrange(counts)
cl1 <- df_class_2_summary$cluster[1]
cl2 <- df_class_2_summary$cluster[2]
hist(df_class_2$score[df_class_2$cluster==cl1])
hist(df_class_2$score[df_class_2$cluster==cl2])
if (max(df_class_2$score[df_class_2$cluster==cl1])<=min(df_class_2$score[df_class_2$cluster==cl2])){
  th <- mean(max(df_class_2$score[df_class_2$cluster==cl1]),min(df_class_2$score[df_class_2$cluster==cl2]))
} else{
  th <- mean(min(df_class_2$score[df_class_2$cluster==cl1]),max(df_class_2$score[df_class_2$cluster==cl2]))
}
important_2 <- df_class_2 %>% filter(score<=th) %>% dplyr::select(latent_variable)
important_2 <- important_2$latent_variable
  
# top10_1 <- order(-abs(importance_class_1))[1:10]
# top10_2 <- order(-abs(importance_class_2))[1:10]

top10_1 <- important_1
top10_2 <- important_2

df1 <- data.frame(scores_1=importance_class_1[top10_1]) %>% rownames_to_column('Genes1')
df2 <- data.frame(scores_2=importance_class_2[top10_2]) %>% rownames_to_column('Genes2')

p <- ggplot(df1,aes(x=reorder(Genes1, -scores_1),y=scores_1)) + geom_bar(stat='identity',fill='#0077b3') + xlab('Latent variable') + 
  ylab('Importance score for classifying into PC3')+
  ggtitle('Important latent variables for cell classification') + 
  scale_x_discrete(expand = c(0, 0))+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(plot.title = element_text(family='Arial',size=32,hjust = 0.5),
        axis.text = element_text(family='Arial',size=27))
print(p)
ggsave(
  '../figures/top_important_latent_variables_to_classify_pc3.png',
  plot = p,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
postscript('../figures/top_important_latent_variables_to_classify_pc3.eps',width = 12,height = 9)
print(p)
dev.off()
ggplot(df2,aes(x=Genes2,y=scores_2)) + geom_bar(stat='identity')

# var_1 <- names(importance_class_1)[top10_1[1:10]]
# var_2 <- names(importance_class_2)[top10_2[1:10]]

var_1 <- top10_1
var_2 <- top10_1

emb1 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_MI_pc3_withclass.csv',header = T)) %>% column_to_rownames('V1')
emb1 <- emb1 %>% mutate(cell='PC3')
emb2 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_MI_ha1e_withclass.csv',header = T)) %>% column_to_rownames('V1')
emb2 <- emb2 %>% mutate(cell='HA1E')
all_embs <- rbind(emb1,emb2)

# x=`z1009`,y=`z263` for AE with classifier only and MI not CPA
p <- ggplot(all_embs,aes(x=`z1009`,y=`z263`)) + geom_point(aes(color=cell)) +
  ggtitle('Scatter plot using only 2 latent variable') +
  theme_minimal(base_family = "Arial",base_size = 36)+
  theme(plot.title = element_text(size=32,hjust = 0.5))
print(p)
ggsave(
  '../figures/AE_with_class_pc3_ha1e_cellspecific_latent_vars.png',
  plot = p,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
postscript('../figures/AE_with_class_pc3_ha1e_cellspecific_latent_vars.eps',width = 12,height = 9)
print(p)
dev.off()
# Load importance to encode
#important_scores_pc3_encode_allgenes_withclass_noabs.csv and var_x[1:2]

### For only top 2 ###
# var_1 <- c('z1009','z263')
# var_2 <- c('z1009','z263')
####

imp_enc_1 <- data.table::fread('../results/Importance_results/important_scores_pc3_to_encode.csv') %>%
  dplyr::select(c('Gene_1'='V1'),all_of(var_1)) %>% column_to_rownames('Gene_1')
imp_enc_1 <- imp_enc_1 %>% mutate(mean_imp=rowMeans(imp_enc_1))
imp_enc_1 <- imp_enc_1 %>% mutate(gene_score = z1009*importance_class_1['z1009'] + z263*importance_class_1['z263'])
imp_enc_2 <- data.table::fread('../results/Importance_results/important_scores_ha1e_to_encode.csv')%>%
  dplyr::select(c('Gene_2'='V1'),all_of(var_2)) %>% column_to_rownames('Gene_2')
imp_enc_2 <- imp_enc_2 %>% mutate(mean_imp=rowMeans(imp_enc_2))
imp_enc_2 <- imp_enc_2 %>% mutate(gene_score = z1009*importance_class_1['z1009'] + z263*importance_class_1['z263'])
df_corr_encode <- data.frame(cell1_corr=imp_enc_1$mean_imp,cell2_corr=imp_enc_2$mean_imp)
spear <- cor(df_corr_encode$cell1_corr,df_corr_encode$cell2_corr,method='spearman')
p <- ggscatter(df_corr_encode,
               x='cell1_corr',y='cell2_corr',rug = TRUE,
               alpha = 0.5,size=1,color = '#1f77b4') + 
  geom_hline(yintercept = 0,color='black',lty=2,linewidth=1) + geom_vline(xintercept = 0,color='black',lty=2,size=1)+
  #geom_smooth(color='black',lty=2)+
  xlab('average importance score for PC3') + ylab('average importance score for HA1E') + 
  ggtitle('Gene importance score for each cell-line according to the model')+
  annotate("text",x=-2e-04,y=4e-04,label=paste0('Spearman`s correlation ',round(spear,4)),size=10)+
  theme_minimal(base_family = "Arial",base_size = 28)+
  theme(plot.title = element_text(size=28,hjust = 1))
print(p)
ggsave(
  '../figures/scorePC3_vs_score_HA1E.png',
  plot = p,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
postscript('../figures/scorePC3_vs_score_HA1E.eps',width = 12,height = 9)
ggscatter(df_corr_encode,
          x='cell1_corr',y='cell2_corr',rug = TRUE,
          size=1,color = '#1f77b4') + 
  geom_hline(yintercept = 0,color='black',lty=2,linewidth=1) + geom_vline(xintercept = 0,color='black',lty=2,size=1)+
  #geom_smooth(color='black',lty=2)+
  xlab('average importance score for PC3') + ylab('average importance score for HA1E') + 
  ggtitle('Gene importance score for each cell-line according to the model')+
  annotate("text",x=-2e-04,y=4e-04,label=paste0('Spearman`s correlation ',round(spear,4)),size=10)+
  theme_minimal(base_family = "Arial",base_size = 28)+
  theme(plot.title = element_text(size=28,hjust = 1))
dev.off()

## See for every z-latent
df_corr_encode <- data.frame(cell1_corr=imp_enc_1$z1009,cell2_corr=imp_enc_2$z1009)
spear <- cor(df_corr_encode$cell1_corr,df_corr_encode$cell2_corr,method='spearman')
p <- ggscatter(df_corr_encode,
               x='cell1_corr',y='cell2_corr',rug = TRUE,
               alpha = 0.5,size=1,color = '#1f77b4') + 
  geom_hline(yintercept = 0,color='black',lty=2,linewidth=1) + geom_vline(xintercept = 0,color='black',lty=2,size=1)+
  #geom_smooth(color='black',lty=2)+
  xlab('z1009 importance score for PC3') + ylab('z1009 importance score for HA1E') + 
  ggtitle('Gene importance score for each cell-line according to the model')+
  annotate("text",x=-1e-03,y=2e-03,label=paste0('Spearman`s correlation ',round(spear,4)),size=5)+
  theme_minimal(base_family = "Arial",base_size = 20)+
  theme(plot.title = element_text(size=20,hjust = 0.5))
print(p)
ggsave(
  '../figures/z1009_scorePC3_vs_score_HA1E.png',
  plot = p,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
postscript('../figures/z1009_scorePC3_vs_score_HA1E.eps',width = 12,height = 9)
ggscatter(df_corr_encode,
          x='cell1_corr',y='cell2_corr',rug = TRUE,
          size=1,color = '#1f77b4') + 
  geom_hline(yintercept = 0,color='black',lty=2,linewidth=1) + geom_vline(xintercept = 0,color='black',lty=2,size=1)+
  #geom_smooth(color='black',lty=2)+
  xlab('z1009 importance score for PC3') + ylab('z1009 importance score for HA1E') + 
  ggtitle('Gene importance score for each cell-line according to the model')+
  annotate("text",x=-1e-03,y=2e-03,label=paste0('Spearman`s correlation ',round(spear,4)),size=5)+
  theme_minimal(base_family = "Arial",base_size = 20)+
  theme(plot.title = element_text(size=20,hjust = 0.5))
dev.off()

df_corr_encode <- data.frame(cell1_corr=imp_enc_1$z263,cell2_corr=imp_enc_2$z263)
spear <- cor(df_corr_encode$cell1_corr,df_corr_encode$cell2_corr,method='spearman')
p <- ggscatter(df_corr_encode,
               x='cell1_corr',y='cell2_corr',rug = TRUE,
               alpha = 0.5,size=1,color = '#1f77b4') + 
  geom_hline(yintercept = 0,color='black',lty=2,linewidth=1) + geom_vline(xintercept = 0,color='black',lty=2,size=1)+
  #geom_smooth(color='black',lty=2)+
  xlab('z263 importance score for PC3') + ylab('z263 importance score for HA1E') + 
  ggtitle('Gene importance score for each cell-line according to the model')+
  annotate("text",x=-1.3e-03,y=2e-03,label=paste0('Spearman`s correlation ',round(spear,4)),size=5)+
  theme_minimal(base_family = "Arial",base_size = 20)+
  theme(plot.title = element_text(size=20,hjust = 0.5))
print(p)
ggsave(
  '../figures/z263_scorePC3_vs_score_HA1E.png',
  plot = p,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
postscript('../figures/z263_scorePC3_vs_score_HA1E.eps',width = 12,height = 9)
ggscatter(df_corr_encode,
          x='cell1_corr',y='cell2_corr',rug = TRUE,
          size=1,color = '#1f77b4') + 
  geom_hline(yintercept = 0,color='black',lty=2,linewidth=1) + geom_vline(xintercept = 0,color='black',lty=2,size=1)+
  #geom_smooth(color='black',lty=2)+
  xlab('z263 importance score for PC3') + ylab('z263 importance score for HA1E') + 
  ggtitle('Gene importance score for each cell-line according to the model')+
  annotate("text",x=-1.3e-03,y=2e-03,label=paste0('Spearman`s correlation ',round(spear,4)),size=5)+
  theme_minimal(base_family = "Arial",base_size = 20)+
  theme(plot.title = element_text(size=20,hjust = 0.5))
dev.off()

### Perform GSEA for the important cell line specific genes
df_corr_encode <- data.frame(PC3=imp_enc_1$mean_imp,HA1E=imp_enc_2$mean_imp)
print(all(rownames(df_corr_encode)==geneInfo$gene_id))
rownames(df_corr_encode) <- geneInfo$gene_symbol
# Order the genes based on the importance
ordered_1 <- rownames(df_corr_encode)[order(-df_corr_encode$PC3)]
ordered_1 <- ordered_1[1:50]
ordered_2 <- rownames(df_corr_encode)[order(df_corr_encode$HA1E)]
ordered_2 <- ordered_2[1:50]



### Perform GSEA in using the scores of latent z1-z3 (or all latents and plot average score)--------------
library(fgsea)
library(EGSEAdata)
# library(topGO)
# library(org.Hs.eg.db)
# library(GO.db)
# GOs_description <- data.frame(GOTERM)
# GOs_description <- GOs_description %>% filter(Ontology=="BP") %>% dplyr::select(go_id,Term) %>% unique()
egsea.data(species = "human",returnInfo = TRUE)
kegg_list <-  kegg.pathways$human$kg.sets
# genes <- factor(x = rep(1,nrow(imp_enc_1)),levels = c(0,1))
# names(genes) <- rownames(imp_enc_1)
# GOobject <- new("topGOdata",ontology = "BP", allGenes = genes, annot=annFUN.org, mapping="org.Hs.eg.db", 
#                 ID = "entrez", nodeSize = 10)
# term.genes <- genesInTerm(GOobject, GOobject@graph@nodes)

print("running fgsea for enrichment space for cell 1")
n_permutations <- 20000
genesets_list <-apply(imp_enc_1 %>% dplyr::select(-gene_score,-mean_imp),
                      MARGIN = 2,fgsea,
                      pathways = kegg_list,
                      minSize=5,
                      maxSize=500,
                      nperm = n_permutations)
print("fgsea finished, preparing outputs for cell 1")
# Prepare output
NES_cell1 <- genesets_list[[1]]$NES
padj_cell1 <- genesets_list[[1]]$padj
pval_cell1 <- genesets_list[[1]]$pval
# ###only for when I use only mean score
# NES_cell1 <- as.matrix(NES_cell1)
# padj_cell1 <- as.matrix(padj_cell1)
# pval_cell1 <- as.matrix(pval_cell1)
# ###
for (i in 2:length(genesets_list)) {
  NES_cell1 <- cbind(NES_cell1,genesets_list[[i]]$NES)
  padj_cell1 <- cbind(padj_cell1,genesets_list[[i]]$padj)
  pval_cell1 <- cbind(pval_cell1,genesets_list[[i]]$pval)
}
colnames(NES_cell1) <- names(genesets_list)
rownames(NES_cell1) <- genesets_list[[1]]$pathway
colnames(pval_cell1) <- names(genesets_list)
rownames(pval_cell1) <- genesets_list[[1]]$pathway
colnames(padj_cell1) <- names(genesets_list)
rownames(padj_cell1) <- genesets_list[[1]]$pathway

print("running fgsea for enrichment space for cell 2")
genesets_list <-apply(imp_enc_2 %>% dplyr::select(-gene_score,-mean_imp),
                      MARGIN = 2,fgsea,
                      pathways = kegg_list,
                      minSize=5,
                      maxSize=500,
                      nperm = n_permutations)
print("fgsea finished, preparing outputs for cell 2")
# Prepare output
NES_cell2 <- genesets_list[[1]]$NES
padj_cell2 <- genesets_list[[1]]$padj
pval_cell2 <- genesets_list[[1]]$pval
# ###only for when I use only mean score
# NES_cell2 <- as.matrix(NES_cell2)
# padj_cell2 <- as.matrix(padj_cell2)
# pval_cell2 <- as.matrix(pval_cell2)
# ###
for (i in 2:length(genesets_list)) {
  NES_cell2 <- cbind(NES_cell2,genesets_list[[i]]$NES)
  padj_cell2 <- cbind(padj_cell2,genesets_list[[i]]$padj)
  pval_cell2 <- cbind(pval_cell2,genesets_list[[i]]$pval)
}
colnames(NES_cell2) <- names(genesets_list)
rownames(NES_cell2) <- genesets_list[[1]]$pathway
colnames(pval_cell2) <- names(genesets_list)
rownames(pval_cell2) <- genesets_list[[1]]$pathway
colnames(padj_cell2) <- names(genesets_list)
rownames(padj_cell2) <- genesets_list[[1]]$pathway

### Process and visualize outcome
# gather cell-line 1 results
padj_cell1 <- as.data.frame(padj_cell1) %>% rownames_to_column('kegg_path') %>% 
  gather('latent_var','p.adj',-kegg_path)
NES_cell1 <- as.data.frame(NES_cell1) %>% rownames_to_column('kegg_path') %>% 
  gather('latent_var','NES',-kegg_path)
df_kegg_cell_1 <- left_join(NES_cell1,padj_cell1)
colnames(df_kegg_cell_1)[3:4] <- c('NES.1','p.adj.1')
# gather cell-line 2 results
padj_cell2 <- as.data.frame(padj_cell2) %>% rownames_to_column('kegg_path') %>% 
  gather('latent_var','p.adj',-kegg_path)
NES_cell2 <- as.data.frame(NES_cell2) %>% rownames_to_column('kegg_path') %>% 
  gather('latent_var','NES',-kegg_path)
df_kegg_cell_2 <- left_join(NES_cell2,padj_cell2)
colnames(df_kegg_cell_2)[3:4] <- c('NES.2','p.adj.2')
# combine
df_kegg <- left_join(df_kegg_cell_1,df_kegg_cell_2)
df_kegg <- df_kegg %>% mutate(significant = ifelse(p.adj.1<0.05,ifelse(p.adj.2<0.05,'both','PC3'),
                                                   ifelse(p.adj.2<0.05,'HA1E','not-significant')))
df_kegg <- rbind(df_kegg %>% dplyr::select(kegg_path,latent_var,c('NES'='NES.1'),significant) %>% unique() %>% mutate(cell='PC3'),
                 df_kegg %>% dplyr::select(kegg_path,latent_var,c('NES'='NES.2'),significant) %>% unique() %>% mutate(cell='HA1E'))
df_kegg <- df_kegg %>% unique()
saveRDS(df_kegg,'../results/Importance_results/kegg_enrichment_imp_genes.rds')
df_kegg <- df_kegg %>% filter(significant!='not-significant')
# df_kegg <- left_join(df_kegg,GOs_description,by=c("kegg_path"="go_id"))
# p <- ggplot(df_kegg,aes(x=NES,y=kegg_path,fill=cell,pattern=significant)) + ylab('KEGG Pathways')+xlab('Normalized Enrichment Score')+
#   geom_bar_pattern(stat='identity',position = "dodge")+
#   theme_classic()+
#   theme(text = element_text(size=25),
#         plot.title = element_text(hjust = 0.5),
#         legend.position="top")+
#   facet_wrap(~latent_var)

p <- ggplot(df_kegg,aes(x=latent_var,y=kegg_path,fill=NES)) + ylab('KEGG Pathways')+xlab('Latent variable')+
  geom_tile()+
  scale_fill_gradient2()+
  theme_classic()+
  theme(text = element_text(size=25),
        plot.title = element_text(hjust = 0.5),
        legend.position="top")+
  facet_wrap(~cell)
print(p)
### Build classifier for fewer and fewer genes-------------------------------
gex <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header = T) %>% column_to_rownames('V1')
gex <-  gex[rownames(all_embs),]
gex <- gex %>% mutate(cell=all_embs$cell)
gex <- gex %>% mutate(cell=ifelse(cell=='PC3',1,0))
gex$cell <- factor(gex$cell,levels = c(0,1))
gc()

# Order the genes based on the importance
ordered_1 <- rownames(imp_enc_1)[order(-imp_enc_1$mean_imp)]
ordered_2 <- rownames(imp_enc_2)[order(imp_enc_2$mean_imp)]

# Train test split
train_indices <- createDataPartition(gex$cell, p = 0.75, list = FALSE)
# Subset your data into training and testing sets based on the indices
train_data <- gex[train_indices, ]
print(sum(train_data $cell==1)/nrow(train_data ))
test_data <- gex[-train_indices, ]
print(sum(test_data$cell==1)/nrow(test_data))
gc()
# train_data <- readRDS('../../train_split_small_cell_glm.rds')
# test_data <- readRDS('../../test_split_small_cell_glm.rds')
# saveRDS(train_data,'../../train_split_small_cell_glm.rds')
# saveRDS(test_data,'../../test_split_small_cell_glm.rds')
genes_to_keep <- c(1,3,5,10,15,20,25,30,35,40,45,50,70,100,150,200)
F1 <- NULL
ACC <- NULL
for (i in 1:length(genes_to_keep)){
  df <- train_data %>% 
    dplyr::select(all_of(unique(c(ordered_1[1:genes_to_keep[i]],ordered_2[1:genes_to_keep[i]],'cell'))))
  # Define training control method
  ctrl <- trainControl(method = "cv", number = 10)
  mdl <- train(cell ~ ., data = df, method = "glm", trControl = ctrl,trace=F,family='binomial')
  y <- predict(mdl,newdata =test_data %>%
                 dplyr::select(all_of(unique(c(ordered_1[1:genes_to_keep[i]],ordered_2[1:genes_to_keep[i]])))))
  conf <- confusionMatrix(reference=test_data$cell,data=y,positive = '1')
  F1[i] <- conf$table[2,2]/(conf$table[2,2]+0.5*(conf$table[2,1]+conf$table[1,2]))
  ACC[i] <- conf$overall['Accuracy']
  message(paste0('Done top ',genes_to_keep[i],' genes'))
}

gene_results <- data.frame(genes_number=genes_to_keep,F1=F1,accuracy = ACC)
ggplot(gene_results %>% filter(!is.na(F1)),aes(x=genes_number,y=F1*100)) + 
  geom_point(color='black',size=3)+
  geom_smooth(se=T,color='#4878CF') + ylim(c(0,100)) +
  scale_y_continuous(breaks=seq(0,100,20),limits = c(0,100))+
  geom_hline(yintercept = 50,color='red',lty='dashed',linewidth=1) + 
  annotate('text',x=50,y=47,label = "50% random F1 threshold",size=10)+
  xlab(paste0('number of important genes used from each cell-line'))+ ylab(paste0('F1 score (%)'))+theme_minimal()+
  ggtitle('GLM performance for classifying cell-line')+
  theme(text = element_text(size=32),plot.title = element_text(hjust = 0.5),
        legend.text=element_text(size=32))

## Random classifier with the same number of genes
random_iter <- 100
genes_to_keep <- c(1,3,5,10,15,20,25,30,35,40,45,50,70,100,150,200)
F1 <- NULL
ACC <- NULL
radom_f1s <- matrix(0,nrow = length(genes_to_keep),random_iter)
radom_accs <- matrix(0,nrow = length(genes_to_keep),random_iter)
for (i in 1:length(genes_to_keep)){
  #radom_f1s <- NULL
  tt <- NULL
  qq <- NULL
  for (j in 1:random_iter){
    random_genes <- sample(colnames(gex)[1:ncol(gex)-1],genes_to_keep[i])
    df <- train_data %>% 
      dplyr::select(all_of(unique(c(random_genes,'cell'))))
    # Define training control method
    ctrl <- trainControl(method = "cv", number = 10)
    mdl <- train(cell ~ ., data = df, method = "glm", trControl = ctrl,trace=F,family='binomial')
    y <- predict(mdl,newdata =test_data %>%
                   dplyr::select(all_of(unique(random_genes))))
    conf <- confusionMatrix(reference=test_data$cell,data=y,positive = '1')
    radom_f1s[i,j] <-  conf$table[2,2]/(conf$table[2,2]+0.5*(conf$table[2,1]+conf$table[1,2]))
    radom_accs[i,j] <- conf$overall['Accuracy']
    #tt[j] <- conf$overall['Accuracy']
    #qq[j] <- conf$table[2,2]/(conf$table[2,2]+0.5*(conf$table[2,1]+conf$table[1,2]))
  }
  # hist(tt)
  # hist(qq)
  #F1[i] <- mean(random_f1s)
  message(paste0('Done top ',genes_to_keep[i],' genes'))
}
# saveRDS(radom_f1s,'../results/Importance_results/glm_radom_f1s.rds')
# saveRDS(radom_accs,'../results/Importance_results/glm_radom_accs.rds')
F1 <- apply(radom_f1s, 1, mean,na.rm=T)
F1_sds <- apply(radom_f1s, 1, sd,na.rm=T)
ACC <- apply(radom_accs, 1, mean,na.rm=T)
ACC_sds <- apply(radom_accs, 1, sd,na.rm=T)
gene_random_results <- data.frame(genes_number=genes_to_keep,F1=F1,F1_sds=F1_sds,accuracy=ACC,accuracy_sd=ACC_sds)
ggplot(gene_random_results %>% filter(!is.na(F1)),aes(x=genes_number,y=F1*100)) + 
  geom_point(color='black',size=3)+
  geom_smooth(se=T,color='#4878CF') + ylim(c(0,100)) +
  scale_y_continuous(breaks=seq(0,100,20),limits = c(0,100))+
  geom_hline(yintercept = 50,color='red',lty='dashed',linewidth=1) + 
  annotate('text',x=50,y=47,label = "50% random F1 threshold",size=10)+
  xlab(paste0('number of important genes used from each cell-line'))+ ylab(paste0('F1 score (%)'))+theme_minimal()+
  ggtitle('GLM random genes performance for classifying cell-line')+
  theme(text = element_text(size=32),plot.title = element_text(hjust = 0.5),
        legend.text=element_text(size=32))

#plot both together
gene_random_results_combined <- rbind(gene_random_results %>% mutate(selected='random genes'),
                                      gene_results %>% mutate(F1_sds=NA) %>% mutate(accuracy_sd=NA) %>% mutate(selected='important genes'))
ggplot(gene_random_results_combined,aes(x=genes_number,y=accuracy*100,color=selected)) + 
  geom_point(size=3)+
  geom_errorbar(data=gene_random_results_combined %>% filter(!is.na(F1_sds)),
                aes(ymin = 100*(accuracy-accuracy_sd), ymax = 100*(accuracy+accuracy_sd)), 
                width = 2,linewidth=1)+
  geom_smooth(se=T) + ylim(c(0,100)) +
  scale_color_manual(values = c('#4878CF','#d97b38'))+
  scale_y_continuous(breaks=seq(0,100,20),limits = c(0,100))+
  geom_hline(yintercept = 50,color='red',lty='dashed',linewidth=1) + 
  annotate('text',x=105,y=46,label = "random cell classification threshold: accuracy=50%",size=10)+
  xlab(paste0('number of genes used from each cell-line'))+ ylab(paste0('accuracy (%)'))+theme_minimal()+
  ggtitle('GLM performance for classifying cell-line')+
  theme(text = element_text(size=32),plot.title = element_text(hjust = 0.5),
        legend.text=element_text(size=30),legend.position = 'bottom')
ggsave(
  '../figures/glm_performance_using_important_genes.png',
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
#saveRDS(gene_results,'../results/Importance_results/glm_genenumber_vs_f1.rds')
pdf('../figures/glm_performance_using_important_genes.pdf',width = 12,height = 9)
ggplot(gene_random_results_combined,aes(x=genes_number,y=accuracy*100,color=selected)) + 
  geom_point(size=3)+
  geom_errorbar(data=gene_random_results_combined %>% filter(!is.na(F1_sds)),
                aes(ymin = 100*(accuracy-accuracy_sd), ymax = 100*(accuracy+accuracy_sd)), 
                width = 2,linewidth=1)+
  geom_smooth(se=T) + ylim(c(0,100)) +
  scale_color_manual(values = c('#4878CF','#d97b38'))+
  scale_y_continuous(breaks=seq(0,100,20),limits = c(0,100))+
  geom_hline(yintercept = 50,color='red',lty='dashed',linewidth=1) + 
  annotate('text',x=105,y=46,label = "random cell classification threshold: accuracy=50%",size=10)+
  xlab(paste0('number of genes used from each cell-line'))+ ylab(paste0('accuracy (%)'))+theme_minimal()+
  ggtitle('GLM performance for classifying cell-line')+
  theme(text = element_text(size=32),plot.title = element_text(hjust = 0.5),
        legend.text=element_text(size=30),legend.position = 'bottom')
dev.off()

### Seems like 25 to 40 genes from each cell-line are enough
### Get these and put them to gProfiler
saveRDS(imp_enc_1,'../results/Importance_results/imp_enc_1.rds')
saveRDS(imp_enc_2,'../results/Importance_results/imp_enc_2.rds')
genes_1 <- data.frame(gene_id = ordered_1[1:25])
genes_1 <- left_join(genes_1,geneInfo %>% dplyr::select(gene_id,gene_symbol))
fileConn<-file("output1.txt")
writeLines(genes_1$gene_symbol, fileConn)
close(fileConn)
saveRDS(genes_1,'../results/Importance_results/top25_genes_pc3.rds')
genes_2 <- data.frame(gene_id = ordered_2[1:25])
genes_2 <- left_join(genes_2,geneInfo %>% dplyr::select(gene_id,gene_symbol))
fileConn<-file("output2.txt")
writeLines(genes_2$gene_symbol, fileConn)
close(fileConn)
saveRDS(genes_2,'../results/Importance_results/top25_genes_ha1e.rds')

# cmap_filtered <- cmap[,unique(c(genes_2$gene_id,genes_1$gene_id))]
# pca_filt <- prcomp(cmap_filtered,scale = F)
# fviz_screeplot(pca_filt)
# df_pca<- pca_filt$x[,1:2]
# df_pca <- as.data.frame(df_pca)
# df_pca  <- df_pca %>% mutate(cell=rownames(cmap_filtered))
# df_pca <- df_pca %>% mutate(cell = ifelse(grepl('HA1E',cell),'HA1E','PC3'))
# pca_plot <- ggplot(df_pca,aes(PC1,PC2)) +geom_point(aes(col=cell))+
#   ggtitle('PCA plot of transcriptomic data for 2 cell-lines') + xlab('PC1')+ ylab('PC2')+
#   theme(text = element_text(size=13))
# print(pca_plot)

### Find relationship between the 2 latent variables and the classification out-come
#var_1 <- c('z1009','z263')
#var_2 <- c('z1009','z263')
# var_1 <- top10_1[c(1,2,4,5,6,9,11)]
# var_2 <- top10_1[c(1,2,4,5,6,9,11)]
var_1 <- top10_1
var_2 <- top10_1
ctrl <- trainControl(method="CV", number=10)
train_data <- sample_n(all_embs[,which(colnames(all_embs) %in% c(var_1,'cell'))],1742)
train_data$cell <- factor(train_data$cell,levels = c('HA1E', 'PC3'))
test_data <- all_embs[which(!(rownames(all_embs) %in% rownames(train_data))),which(colnames(all_embs) %in% c(var_2,'cell'))]
test_data$cell <- factor(test_data$cell,levels = c('HA1E', 'PC3'))
mdl <- train(cell ~ ., data = train_data, method = "bayesglm", trControl = ctrl,trace=T)
mean(mdl[["resample"]][["Accuracy"]])
# Evaluate precision and accuracy in test set
y <- predict(mdl,newdata = test_data[,1:(ncol(test_data)-1)])
results <- confusionMatrix(factor(test_data$cell,levels = c('HA1E', 'PC3')),y)
results$byClass
params <- summary(mdl)
print(params)
params <- params[["coefficients"]]
a_z263 <- params[2,1]
b_z1009 <-  params[3,1]

df_1 <- imp_enc_1 %>% mutate(mean_score=a_z263*z263+b_z1009*z1009)
df_1 <- df_1 %>% rownames_to_column('gene')
df_2 <- imp_enc_2 %>% mutate(mean_score=a_z263*z263+b_z1009*z1009)
df_2 <- df_2 %>% rownames_to_column('gene')

### kmeans approach where I use all vars not just the top 2
kmeans_1 <- kmeans(as.matrix(imp_enc_1[,1:ncol(imp_enc_1)-1]),centers = 3,iter.max = 1000, nstart = 200)
df_1 <-  data.frame(gene=rownames(imp_enc_1),mean_score = imp_enc_1$mean_imp,cluster =kmeans_1$cluster)
df_1_summary <- df_1 %>% group_by(cluster) %>% summarise(counts = n()) %>% arrange(counts)
cl1 <- df_1_summary$cluster[1]
cl2 <- df_1_summary$cluster[2]
hist(df_1$mean_score[df_1$cluster==cl1])
hist(df_1$mean_score[df_1$cluster==cl2])
if (mean(df_class_2$score[df_class_2$cluster==cl1])>mean(df_class_2$score[df_class_2$cluster==cl2])){
  df_1 <- df_1 %>% filter(cluster==cl1)
} else if (mean(df_class_2$score[df_class_2$cluster==cl1])<mean(df_class_2$score[df_class_2$cluster==cl2])){
  df_1 <- df_1 %>% filter(cluster==cl2)
}else{
  print('Cannot discern two cell-lines')
}
kmeans_2 <- kmeans(as.matrix(imp_enc_2[,1:ncol(imp_enc_2)-1]),centers = 3,iter.max = 1000, nstart = 200)
df_2 <-  data.frame(gene=rownames(imp_enc_2),mean_score = imp_enc_2$mean_imp,cluster =kmeans_2$cluster)
df_2_summary <- df_2 %>% group_by(cluster) %>% summarise(counts = n()) %>% arrange(counts)
cl1 <- df_2_summary$cluster[1]
cl2 <- df_2_summary$cluster[2]
hist(df_2$mean_score[df_2$cluster==cl1])
hist(df_2$mean_score[df_2$cluster==cl2])
if (mean(df_class_2$score[df_class_2$cluster==cl1])<mean(df_class_2$score[df_class_2$cluster==cl2])){
  df_2 <- df_2 %>% filter(cluster==cl1)
} else if (mean(df_class_2$score[df_class_2$cluster==cl1])>mean(df_class_2$score[df_class_2$cluster==cl2])){
  df_2 <- df_2 %>% filter(cluster==cl2)
}else{
  print('Cannot discern two cell-lines')
}

#df_1 <- imp_enc_1 %>% rownames_to_column('gene')
#df_2 <- imp_enc_2 %>% rownames_to_column('gene')
geneInfo$gene_id <- as.character(geneInfo$gene_id)
df1 <- left_join(df_1,geneInfo,by=c("gene"="gene_id"))
g1 <- df1$gene
df2 <- left_join(df_2,geneInfo,by=c("gene"="gene_id"))
#df1 <- df1 %>% filter(!(gene %in% df2$gene))
#df2 <- df2 %>% filter(!(gene %in% g1))
df1 <- df1 %>% filter(mean_score>0)
df1 <- df1 %>% arrange(-mean_score)
df2 <- df2 %>% filter(mean_score<0)
df2 <- df2 %>% arrange(mean_score)
####

gex <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header = T) %>% column_to_rownames('V1')
ind1 <- grep('PC3',rownames(gex))
ind2 <- grep('HA1E',rownames(gex))
gex <- gex %>% mutate(cell='None')
gex$cell[ind1] <- 'PC3'
gex$cell[ind2] <- 'HA1E'

### ONLY FOR KMEANS APPROACH
gex_filtered <- gex %>% select(all_of(unique(c(df1$gene,df2$gene))),cell)
####

# p <- ggplot(gex,aes(x=`23636`,y=`1465`)) + geom_point(aes(color=cell))+
#   ggtitle('Scatter plot of transcriptomic data for 2 cell-lines') +
#   theme(text = element_text(size=13))
# print(p)
# png('../figures/2dscatterplot_pc3_ha1e_gex_withclass.png',width=10,height = 10,units = "in",res=300)
# print(p)
# dev.off()

# library(plotly)
# fig <- plot_ly(gex, x = ~`5997`, y = ~`3162`, z = ~`84617`, color = ~cell, colors = c('#BF382A', '#0C4B8E'))
# fig <- fig %>% add_markers(size=1)
# fig

## Get top 100 (50 for every cell) genes important to encode into cell-associated latent space
#imp_enc_1 <- imp_enc_1 %>% column_to_rownames('Gene_1')
#imp_enc_2 <- imp_enc_2 %>% column_to_rownames('Gene_2')

# to eixa order by absolute value kai sta 2
top50_1 <- rownames(imp_enc_1)[order(-imp_enc_1$mean_imp)][1:50]
top50_2 <- rownames(imp_enc_2)[order(imp_enc_2$mean_imp)][1:50]

df1 <- as.data.frame(top50_1)
df2 <- as.data.frame(top50_2)
geneInfo$gene_id <- as.character(geneInfo$gene_id) 
df1 <- left_join(df1,geneInfo,by=c("top50_1"="gene_id"))
df2 <- left_join(df2,geneInfo,by=c("top50_2"="gene_id"))

gex_filtered <- gex %>% select(all_of(unique(c(top50_1,top50_2))),cell)

pca_filt <- prcomp(gex_filtered[,1:(ncol(gex_filtered)-1)],scale = F)
fviz_screeplot(pca_filt)
df_pca<- pca_filt$x[,1:2]
df_pca <- as.data.frame(df_pca)
df_pca  <- df_pca %>% mutate(cell=gex_filtered$cell)
pca_plot <- ggplot(df_pca,aes(PC1,PC2)) +geom_point(aes(col=cell))+
  ggtitle('PCA plot of transcriptomic data for 2 cell-lines') + xlab('PC1')+ ylab('PC2')+
  theme(text = element_text(size=13))
print(pca_plot)


#png(paste0('../figures/pca_pc3_ha1e_gex_filtered.png'),width=10,height = 10,units = "in",res=300)
#print(tsne_plot_allgenes)
#dev.off()


perpl = DescTools::RoundTo(sqrt(nrow(gex)), multiple = 5, FUN = round)
#perpl=2
init_dim = 10
iter = 1000
emb_size = ncol(gex_filtered)-1
set.seed(42)
tsne_all <- Rtsne(gex_filtered[,1:(ncol(gex_filtered)-1)], 
                  dims = 2, perplexity=perpl, 
                  verbose=TRUE, max_iter = iter,
                  check_duplicates = F,normalize = F,
                  pca_scale = F,pca=T,init_dim=init_dim,
                  num_threads = 15)
df_tsne <- data.frame(V1 = tsne_all$Y[,1], V2 =tsne_all$Y[,2])
rownames(df_tsne) <- rownames(gex)
colnames(df_tsne) <- c('Dim1','Dim2')
df_tsne  <- df_tsne %>% mutate(cell=gex_filtered$cell)
tsne_plot <- ggplot(df_tsne,aes(Dim1,Dim2)) +geom_point(aes(col=cell))+
  ggtitle('t-SNE plot of transcriptomic data for 2 cell-lines') + xlab('Dim 1')+ ylab('Dim 2')+
  theme(text = element_text(size=13))
print(tsne_plot)

library(caret)
ctrl <- trainControl(method = "cv", number = 10)
train_gex <- sample_n(gex_filtered,1742)
test_gex <- gex_filtered[which(!(rownames(gex_filtered) %in% rownames(train_gex))),]
mdl <- train(cell ~ ., data = train_gex, method = "rf", trControl = ctrl,trace=T)
#saveRDS(mdl,'../../rf_model_with_importants.rds')
#mdl$results$Accuracy
mean(mdl[["resample"]][["Accuracy"]])
#summary(mdl)
# Evaluate precision and accuracy in test set
y <- predict(mdl,newdata = test_gex[,1:(ncol(test_gex)-1)])
confusionMatrix(factor(test_gex$cell,levels = c('HA1E', 'PC3')),y)
feature_imp <- varImp(mdl,scale=T)
feature_imp <- feature_imp[["importance"]]
p <- ggplot(gex,aes(x=`4638`,y=`7846`,col=cell)) + geom_point()+
  ggtitle('Scatter plot for 2 cell-lines')+
  theme(text = element_text(size=13))
print(p)

library(plotly)
fig <- plot_ly(gex_filtered, x = ~`4638`, y = ~`7846`, z = ~`10398`, color = ~cell, colors = c('#BF382A', '#0C4B8E'))
fig <- fig %>% add_markers(size=1)
fig

ggplot(feature_imp %>% rownames_to_column('gene_id') %>% arrange(desc(Overall)) %>% mutate(id=seq(1,nrow(feature_imp))),
       aes(x=id,y=Overall)) + geom_bar(stat = 'identity') + xlim(c(0,50))

selected <- feature_imp %>% rownames_to_column('gene_id') %>% arrange(desc(Overall))
selected <- selected %>% mutate(gene_id=gsub("[^[:alnum:][:blank:]+?&/\\-]", "", gene_id))
selected <- selected$gene_id[1:10]
### Run random forest again for selected genes
train_gex <- sample_n(gex_filtered[,which(colnames(gex_filtered) %in% c(selected,'cell'))],1742)
test_gex <- gex_filtered[which(!(rownames(gex_filtered) %in% rownames(train_gex))),which(colnames(gex_filtered) %in% c(selected,'cell'))]
mdl <- train(cell ~ ., data = train_gex, method = "rf", trControl = ctrl,trace=T)
mean(mdl[["resample"]][["Accuracy"]])
# Evaluate precision and accuracy in test set
y <- predict(mdl,newdata = test_gex[,1:(ncol(test_gex)-1)])
confusionMatrix(factor(test_gex$cell,levels = c('HA1E', 'PC3')),y)
feature_imp <- varImp(mdl,scale=T)
feature_imp <- feature_imp[["importance"]]
p <- ggplot(gex,aes(x=`4638`,y=`7846`,col=cell)) + geom_point()+
  ggtitle('Scatter plot for 2 cell-lines')+
  theme(text = element_text(size=13))
print(p)
fig <- plot_ly(gex_filtered, x = ~`4638`, y = ~`7846`, z = ~`8942`, color = ~cell, colors = c('#BF382A', '#0C4B8E'))
fig <- fig %>% add_markers(size=1)
fig
###

### Perform genesets functional analysis in important----------------------------------------------------------------------------
importance_translation <- read.csv('../results/Importance_results/important_scores_pc3_to_ha1e_allgenes_withclass_noabs.csv')
importance_translation <- importance_translation %>% column_to_rownames('X')
colnames(importance_translation) <- rownames(importance_translation)
cmap <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header=T) %>% column_to_rownames('V1')
gc()


### Perform the averaging analysis with absolute values
importance_translation_mean <- as.matrix(rowMeans(abs(importance_translation)))
# importance_translation_mean <- scale(importance_translation_mean,center = T,scale=T)
hist(importance_translation_mean,40)
colnames(importance_translation_mean) <- 'HA1E_2_PC3'
avg_keggs <- fastenrichment(colnames(importance_translation_mean),
                            colnames(cmap),
                            importance_translation_mean,
                            enrichment_space = "kegg",
                            order_columns = F,
                            pval_adjustment=T,
                            n_permutations=10000)
avg_keggs_nes <- as.data.frame(as.matrix(avg_keggs[["NES"]]$`NES KEGG`)) %>% rownames_to_column('KEGG pathway')
colnames(avg_keggs_nes)[2] <- 'NES'
avg_keggs_pval <- as.data.frame(as.matrix(avg_keggs[["Pval"]]$`Pval KEGG`)) %>% rownames_to_column('KEGG pathway')
colnames(avg_keggs_pval)[2] <- 'p.adj'
df_avg_keggs <- left_join(avg_keggs_nes,avg_keggs_pval)
df_avg_keggs <- df_avg_keggs %>% mutate(`KEGG pathway`=strsplit(`KEGG pathway`,"_"))
df_avg_keggs <- df_avg_keggs %>% unnest(`KEGG pathway`) %>% filter(!(`KEGG pathway` %in% c("KEGG","FL1000")))
df_avg_keggs <- df_avg_keggs %>% filter(p.adj<=0.05)
df_avg_keggs <- df_avg_keggs %>% mutate(`KEGG pathway`=as.character(`KEGG pathway`))
df_avg_keggs <- df_avg_keggs %>% mutate(`KEGG pathway`=substr(`KEGG pathway`, 9, nchar(`KEGG pathway`)))
# saveRDS(df_avg_keggs,'../results/Importance_results/keggs_enrich_HA1E_to_PC3.rds')
df_avg_keggs <- readRDS('../results/Importance_results/keggs_enrich_PC3_to_HA1E.rds')
top_keggs <-df_avg_keggs$`KEGG pathway`[order(df_avg_keggs$p.adj)]
top_keggs <- top_keggs[1:15]
df_avg_keggs <- df_avg_keggs %>% filter(`KEGG pathway` %in% top_keggs)
df_avg_keggs$`KEGG pathway` <- factor(df_avg_keggs$`KEGG pathway`,levels = df_avg_keggs$`KEGG pathway`[order(df_avg_keggs$NES)])
ggplot(df_avg_keggs,aes(x=NES,y=`KEGG pathway`,fill=p.adj)) + geom_bar(stat = 'identity',color='black',size=1.5) +
  scale_fill_gradient(low = "red",high = "white",limits = c(min(df_avg_keggs$p.adj),0.05)) +
  xlab('importance enrichment score') +
  ggtitle('Top 15 significant KEGG pathways for translating PC3 to HA1E')+
  theme_pubr(base_family = 'Arial',base_size = 22)+
  theme(plot.title = element_text(hjust = 0.8),
        legend.key.size = unit(1.5, "lines"),
        legend.position = 'right',
        legend.justification = "center")
postscript('../figures/significant_KEGG_PC3_to_HA1E.eps',width=12,height=9)
ggplot(df_avg_keggs,aes(x=NES,y=`KEGG pathway`,fill=p.adj)) + geom_bar(stat = 'identity',color='black',size=1.5) +
  scale_fill_gradient(low = "red",high = "white",limits = c(min(df_avg_keggs$p.adj),0.05)) +
  xlab('importance enrichment score') +
  ggtitle('Top 15 significant KEGG pathways for translating PC3 to HA1E')+
  theme_pubr(base_family = 'Arial',base_size = 22)+
  theme(plot.title = element_text(hjust = 0.8),
        legend.key.size = unit(1.5, "lines"),
        legend.position = 'right',
        legend.justification = "center")
dev.off()
ggsave('../figures/significant_KEGG_PC3_to_HA1E.png',
       height = 9,
       width = 12,
       units = 'in',
       dpi=600)

# ### Repeat the same for GO Terms
# avg_gos <- fastenrichment(colnames(importance_translation_mean),
#                             colnames(cmap),
#                             importance_translation_mean,
#                             enrichment_space = "go_bp",
#                             order_columns = F,
#                             pval_adjustment=T,
#                             n_permutations=10000)
# avg_gos_nes <- as.data.frame(as.matrix(avg_gos[["NES"]]$`NES GO BP`)) %>% rownames_to_column('GO Terms')
# colnames(avg_gos_nes)[2] <- 'NES'
# avg_gos_pval <- as.data.frame(as.matrix(avg_gos[["Pval"]]$`Pval GO BP`)) %>% rownames_to_column('GO Terms')
# colnames(avg_gos_pval)[2] <- 'p.adj'
# df_avg_gos <- left_join(avg_gos_nes,avg_gos_pval)
# df_avg_gos <- df_avg_gos %>% mutate(`GO Terms`=strsplit(`GO Terms`,"_"))
# df_avg_gos <- df_avg_gos %>% unnest(`GO Terms`) %>% filter(!(`GO Terms` %in% c("GO","BP","FL1000")))
# go_annotations_list <- as.list(GOTERM)
# go_annotations <- data.frame(GOs = Term(GOTERM),
#                              'GO Terms' = GOID(GOTERM),
#                              definition = Definition(GOTERM),
#                              ontology = Ontology(GOTERM))
# colnames(go_annotations) <- c('GO','GO Terms','definition','ontology')
# df_avg_gos <- left_join(df_avg_gos,go_annotations)
# df_avg_gos <- df_avg_gos %>% dplyr::select(GO,NES,p.adj) %>% unique()
# colnames(df_avg_gos)[1] <- 'GO Terms'
# df_avg_gos <- df_avg_gos %>% filter(p.adj<0.05)
# df_avg_gos <- df_avg_gos %>% mutate(`GO Terms`=as.character(`GO Terms`))
# #df_avg_gos <- df_avg_gos %>% mutate(`GO Terms`=substr(`GO Terms`, 9, nchar(`GO Terms`)))
# df_avg_gos$`GO Terms` <- factor(df_avg_gos$`GO Terms`,levels = df_avg_gos$`GO Terms`[order(df_avg_gos$NES)])
# ggplot(df_avg_gos %>% filter(NES>=1.887),aes(x=NES,y=`GO Terms`,fill=p.adj)) + geom_bar(stat = 'identity',color='black',size=1) +
#   scale_fill_gradient(low = "red",high = "white") +
#   ggtitle('Top 50 significantly enriched GO Terms for translating PC3 to HA1E')+
#   theme_pubr(base_family = 'Arial',base_size = 18)+
#   theme(plot.title = element_text(hjust = 0.5),
#         legend.key.size = unit(1.5, "lines"),
#         legend.position = 'right',
#         legend.justification = "center")
# ggsave('../figures/significant_GOs_PC3_to_HA1E.eps',
#        device = cairo_ps,
#        height = 12,
#        width = 24,
#        units = 'in',
#        dpi=600) 
# ggsave('../figures/significant_GOs_PC3_to_HA1E.png',
#        height = 12,
#        width = 24,
#        units = 'in',
#        dpi=600)
### Perform GSEA with TFs
geneInfo <- geneInfo %>% filter(feature_space!='inferred') %>% filter(gene_type=='protein-coding')
x <- importance_translation_mean
print(all(rownames(x)==geneInfo$gene_id))
rownames(x) <- geneInfo$gene_symbol
avg_tfs <- fastenrichment(colnames(x),
                          rownames(x),
                          x,
                          enrichment_space = "tf_dorothea",
                          order_columns = F,
                          pval_adjustment=T,
                          tf_path='../../../Artificial-Signaling-Network/TF activities/annotation/dorothea.tsv',
                          n_permutations=10000)
avg_tfs_nes <- as.data.frame(as.matrix(avg_tfs[["NES"]]$`NES TF`)) %>% rownames_to_column('TF')
colnames(avg_tfs_nes)[2] <- 'NES'
avg_tfs_pval <- as.data.frame(as.matrix(avg_tfs[["Pval"]]$`Pval TF`)) %>% rownames_to_column('TF')
colnames(avg_tfs_pval)[2] <- 'p.adj'
df_avg_tfs <- left_join(avg_tfs_nes,avg_tfs_pval)
df_avg_tfs <- df_avg_tfs %>% mutate(`TF`=strsplit(`TF`,"_"))
df_avg_tfs <- df_avg_tfs %>% unnest(`TF`) %>% filter(!(`TF` %in% c("TF","DOROTHEA","FL1000")))
df_avg_tfs <- df_avg_tfs %>% filter(p.adj<0.05)
df_avg_tfs <- df_avg_tfs %>% mutate(`TF`=as.character(`TF`))
# saveRDS(df_avg_tfs,'../results/Importance_results/tfs_enrich_PC3_to_HA1E.rds')
df_avg_tfs <- readRDS('../results/Importance_results/tfs_enrich_PC3_to_HA1E.rds')
#df_avg_tfs <- df_avg_tfs %>% mutate(`TF`=substr(`TF`, 9, nchar(`TF`)))
top_tfs <-df_avg_tfs$`TF`[order(df_avg_tfs$p.adj)]
top_tfs <- top_tfs[1:16]
df_avg_tfs <- df_avg_tfs %>% filter(`TF` %in% top_tfs)
df_avg_tfs$`TF` <- factor(df_avg_tfs$`TF`,levels = df_avg_tfs$`TF`[order(df_avg_tfs$NES)])
ggplot(df_avg_tfs,aes(x=NES,y=`TF`,fill=p.adj)) + geom_bar(stat = 'identity',color='black',size=1.2) +
  scale_fill_gradient(low = "red",high = "white",limits = c(min(df_avg_tfs$p.adj),0.05)) +
  xlab('importance enrichment score') + ylab('transcription factor') +
  ggtitle('Top 16 Significant TFs for translating PC3 to HA1E')+
  theme_pubr(base_family = 'Arial',base_size = 24)+
  theme(plot.title = element_text(hjust = 0.8),
        axis.text.y = element_text(size=14),
        legend.key.size = unit(1.5, "lines"),
        legend.position = 'right',
        legend.justification = "center")
postscript('../figures/significant_TF_PC3_to_HA1E.eps',width=12,height=9)
ggplot(df_avg_tfs,aes(x=NES,y=`TF`,fill=p.adj)) + geom_bar(stat = 'identity',color='black',size=1.2) +
  scale_fill_gradient(low = "red",high = "white",limits = c(min(df_avg_tfs$p.adj),0.05)) +
  xlab('importance enrichment score') + ylab('transcription factor') +
  ggtitle('Top 16 Significant TFs for translating PC3 to HA1E')+
  theme_pubr(base_family = 'Arial',base_size = 24)+
  theme(plot.title = element_text(hjust = 0.8),
        axis.text.y = element_text(size=14),
        legend.key.size = unit(1.5, "lines"),
        legend.position = 'right',
        legend.justification = "center")
dev.off()
ggsave('../figures/significant_TF_PC3_to_HA1E.png',
       height = 9,
       width = 12,
       units = 'in',
       dpi=600)

# ## Repeat per sample for KEGGS and tfs
# avg_keggs <- fastenrichment(colnames(importance_translation),
#                             colnames(cmap),
#                             importance_translation,
#                             enrichment_space = "kegg",
#                             order_columns = T,
#                             pval_adjustment=T,
#                             n_permutations=1000)
# avg_keggs <- readRDS('../../../../Downloads/avg_keggs_per_sample.rds')
# avg_keggs_nes <- as.data.frame(as.matrix(avg_keggs[["NES"]]$`NES KEGG`)) %>% rownames_to_column('KEGG pathway')
# # avg_keggs_nes <-  avg_keggs_nes %>% mutate(meanNES=rowMeans(avg_keggs_nes[,2:10087]))
# # avg_keggs_nes <-  avg_keggs_nes %>% mutate(sdNES=apply(avg_keggs_nes[,2:10087],1,sd))
# avg_keggs_nes <- avg_keggs_nes %>% gather('gene','NES',-`KEGG pathway`)
# avg_keggs_pval <- as.data.frame(as.matrix(avg_keggs[["Pval"]]$`Pval KEGG`)) %>% rownames_to_column('KEGG pathway')
# avg_keggs_pval <- avg_keggs_pval %>% gather('gene','p.adj',-`KEGG pathway`)
# df_avg_keggs <- left_join(avg_keggs_nes,avg_keggs_pval)
# df_avg_keggs <- df_avg_keggs %>% mutate(`KEGG pathway`=strsplit(`KEGG pathway`,"_"))
# df_avg_keggs <- df_avg_keggs %>% unnest(`KEGG pathway`) %>% filter(!(`KEGG pathway` %in% c("KEGG","FL1000")))
# df_avg_keggs <- df_avg_keggs %>% group_by(`KEGG pathway`) %>% mutate(significant_counts = sum(p.adj<0.05)) %>% ungroup()
# df_avg_keggs <- df_avg_keggs %>% filter(p.adj<0.05) %>% group_by(`KEGG pathway`) %>%
#   mutate(meanNES = mean(NES)) %>% mutate(sdNES=sd(NES)) %>% ungroup()
# #df_avg_keggs <- df_avg_keggs %>% filter(p.adj<=0.05)
# df_avg_keggs <- df_avg_keggs %>% mutate(`KEGG pathway`=as.character(`KEGG pathway`))
# df_avg_keggs <- df_avg_keggs %>% mutate(`KEGG pathway`=substr(`KEGG pathway`, 9, nchar(`KEGG pathway`)))
# df_avg_keggs$`KEGG pathway` <- factor(df_avg_keggs$`KEGG pathway`,levels = unique(df_avg_keggs$`KEGG pathway`[order(df_avg_keggs$meanNES)]))
# df_avg_keggs <- df_avg_keggs %>% filter(significant_counts>=1)
# ggplot(df_avg_keggs %>% dplyr::select(`KEGG pathway`,meanNES,sdNES,significant_counts) %>% 
#          filter(significant_counts>=1000)%>% unique(),
#        aes(x=meanNES,y=`KEGG pathway`,fill=significant_counts)) + geom_bar(stat = 'identity',color='black',size=1.5) +
#   scale_fill_gradient(low = "#fae1e1",high = "red") +
#   ggtitle('Significantly enriched KEGG pathways for translating PC3 to HA1E')+
#   theme_pubr(base_family = 'Arial',base_size = 10)+
#   theme(plot.title = element_text(hjust = 0.5),
#         legend.key.size = unit(1.5, "lines"),
#         legend.position = 'right',
#         legend.justification = "center")
# ggsave('../figures/significant_KEGG_PC3_to_HA1E_persample.eps',
#        device = cairo_ps,
#        height = 9,
#        width = 12,
#        units = 'in',
#        dpi=600) 
# ggsave('../figures/significant_KEGG_PC3_to_HA1E_persample.png',
#        height = 9,
#        width = 12,
#        units = 'in',
#        dpi=600)
# ## Run for TFS
# x <- importance_translation
# print(all(rownames(x)==geneInfo$gene_id))
# rownames(x) <- geneInfo$gene_symbol
# avg_tfs <- fastenrichment(colnames(x),
#                           rownames(x),
#                           x,
#                           enrichment_space = "tf_dorothea",
#                           order_columns = F,
#                           pval_adjustment=T,
#                           tf_path='../../../Artificial-Signaling-Network/TF activities/annotation/dorothea.tsv',
#                           n_permutations=10000)
# avg_tfs <- readRDS('../../../../Downloads/avg_tfs_per_sample.rds')
# avg_tfs_nes <- as.data.frame(as.matrix(avg_tfs[["NES"]]$`NES TF`)) %>% rownames_to_column('TF')
# avg_tfs_pval <- as.data.frame(as.matrix(avg_tfs[["Pval"]]$`Pval TF`)) %>% rownames_to_column('TF')
# avg_tfs_nes <- avg_tfs_nes %>% gather('gene','NES',-`TF`)
# avg_tfs_pval <- avg_tfs_pval %>% gather('gene','p.adj',-`TF`)
# df_avg_tfs <- left_join(avg_tfs_nes,avg_tfs_pval)
# df_avg_tfs <- df_avg_tfs %>% mutate(`TF`=strsplit(`TF`,"_"))
# df_avg_tfs <- df_avg_tfs %>% unnest(`TF`) %>% filter(!(`TF` %in% c("TF","DOROTHEA","FL1000")))
# df_avg_tfs <- df_avg_tfs %>% group_by(`TF`) %>% mutate(significant_counts = sum(p.adj<0.05)) %>% ungroup()
# df_avg_tfs <- df_avg_tfs %>% filter(p.adj<0.05)
# df_avg_tfs <- df_avg_tfs %>% filter(p.adj<0.05) %>% group_by(`TF`) %>%
#   mutate(meanNES = mean(NES)) %>% mutate(sdNES=sd(NES)) %>% ungroup()
# df_avg_tfs <- df_avg_tfs %>% mutate(`TF`=as.character(`TF`))
# #df_avg_tfs <- df_avg_tfs %>% mutate(`TF`=substr(`TF`, 9, nchar(`TF`)))
# df_avg_tfs$`TF` <- factor(df_avg_tfs$`TF`,levels = unique(df_avg_tfs$`TF`[order(df_avg_tfs$meanNES)]))
# df_avg_tfs <- df_avg_tfs %>% filter(significant_counts>=1)
# ggplot(df_avg_tfs %>% dplyr::select(`TF`,meanNES,sdNES,significant_counts) %>% 
#          filter(significant_counts>=5000)%>% unique(),
#        aes(x=meanNES,y=`TF`,fill=significant_counts)) + geom_bar(stat = 'identity',color='black',size=1.5) +
#   scale_fill_gradient(high = "red",low = "#fae1e1") +
#   ggtitle('Top 50% significantly enriched TFs for translating PC3 to HA1E')+
#   theme_pubr(base_family = 'Arial',base_size = 24)+
#   theme(plot.title = element_text(hjust = 0.5),
#         axis.text.y = element_text(size=14),
#         legend.key.size = unit(1.5, "lines"),
#         legend.position = 'right',
#         legend.justification = "center")
# ggsave('../figures/significant_TF_PC3_to_HA1E_perSample.eps',
#        device = cairo_ps,
#        height = 9,
#        width = 12,
#        units = 'in',
#        dpi=600) 
# ggsave('../figures/significant_TF_PC3_to_HA1E_perSample.png',
#        height = 9,
#        width = 12,
#        units = 'in',
#        dpi=600)

### Investigate the overlap between the 2 directions of translation
dorothera_regulon <- read.delim('../../../Artificial-Signaling-Network/TF activities/annotation/dorothea.tsv')
confLevel <- c('A','B')
dorothera_regulon <- dorothera_regulon %>% filter(confidence %in% confLevel)
dorothera_regulon <- dorothera_regulon %>% dplyr::select(tf,target)
dorothera_regulon <- dorothera_regulon %>% filter(target %in% rownames(x))
dorothera_regulon <- distinct(dorothera_regulon)
print(length(unique(dorothera_regulon$tf)))
dorothera_regulon <- dorothera_regulon %>% dplyr::select(c('TF'='tf')) %>% mutate(set='in_regulon') %>% unique()
tfs_pc3_2_ha1e <- readRDS('../results/Importance_results/tfs_enrich_PC3_to_HA1E.rds')
tfs_pc3_2_ha1e <- tfs_pc3_2_ha1e %>% gather('metric','value',-TF) %>% mutate(set='PC3_2_HA1E')
tfs_ha1e_2_pc3 <- readRDS('../results/Importance_results/tfs_enrich_HA1E_to_PC3.rds')
tfs_ha1e_2_pc3 <- tfs_ha1e_2_pc3 %>% gather('metric','value',-TF) %>% mutate(set='HA1E_2_PC3')
tf_overlap <- rbind(tfs_ha1e_2_pc3,tfs_pc3_2_ha1e)
ggVennDiagram(list("PC3 to HA1E" = tfs_pc3_2_ha1e$TF,
                   "HA1E to PC3" = tfs_ha1e_2_pc3$TF),
              label_alpha = 0) + 
  scale_fill_gradient(low="white",high = "red",limits = c(0,42)) + 
  scale_color_manual(values = c('black','black'))+
  ggtitle('Overlap of enriched TFs') +
  theme(text = element_text(family = 'Arial',size = 20),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'none')
postscript('../figures/significant_tfs_overlap_HA1E_PC3.eps',width = 9,height = 9) 
ggVennDiagram(list("PC3 to HA1E" = tfs_pc3_2_ha1e$TF,
                   "HA1E to PC3" = tfs_ha1e_2_pc3$TF),
              label_alpha = 0) + 
  scale_fill_gradient(low="white",high = "red",limits = c(0,42)) + 
  scale_color_manual(values = c('black','black'))+
  ggtitle('Overlap of enriched TFs') +
  theme(text = element_text(family = 'Arial',size = 20),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'none')
dev.off()
ggsave('../figures/significant_tfs_overlap_HA1E_PC3.png',
       height = 9,
       width = 9,
       units = 'in',
       dpi=600)
## Same for KEGGs
keggs_pc3_2_ha1e <- readRDS('../results/Importance_results/keggs_enrich_PC3_to_HA1E.rds')
keggs_pc3_2_ha1e <- keggs_pc3_2_ha1e %>% gather('metric','value',-`KEGG pathway`) %>% mutate(set='PC3_2_HA1E')
keggs_ha1e_2_pc3 <- readRDS('../results/Importance_results/keggs_enrich_HA1E_to_PC3.rds')
keggs_ha1e_2_pc3 <- keggs_ha1e_2_pc3 %>% gather('metric','value',-`KEGG pathway`) %>% mutate(set='HA1E_2_PC3')
kegg_overlap <- rbind(keggs_ha1e_2_pc3,keggs_pc3_2_ha1e)
ggVennDiagram(list("PC3 to HA1E" = keggs_pc3_2_ha1e$`KEGG pathway`,
                   "HA1E to PC3" = keggs_ha1e_2_pc3$`KEGG pathway`),
              label_alpha = 0) + 
  scale_fill_gradient(low="white",high = "red",limits = c(0,15)) + 
  scale_color_manual(values = c('black','black'))+
  ggtitle('Overlap of enriched KEGG pathways') +
  theme(text = element_text(family = 'Arial',size = 20),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'none')
postscript('../figures/significant_KEGG_overlap_HA1E_PC3.eps',width = 9,height = 9)
ggVennDiagram(list("PC3 to HA1E" = keggs_pc3_2_ha1e$`KEGG pathway`,
                   "HA1E to PC3" = keggs_ha1e_2_pc3$`KEGG pathway`),
              label_alpha = 0) + 
  scale_fill_gradient(low="white",high = "red",limits = c(0,15)) + 
  scale_color_manual(values = c('black','black'))+
  ggtitle('Overlap of enriched KEGG pathways') +
  theme(text = element_text(family = 'Arial',size = 20),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'none')
dev.off()
ggsave('../figures/significant_KEGG_overlap_HA1E_PC3.png',
       height = 9,
       width = 9,
       units = 'in',
       dpi=600)
### Per sample analysis GSEA--------------------------------------------------------------------------------------------
df_per_sample <- read.csv('../results/Importance_results/important_scores_ha1e_to_pc3_per_sample_allgenes.csv')
df_per_sample <- df_per_sample %>% unique()
rownames( df_per_sample ) <- NULL
df_per_sample <- df_per_sample %>% column_to_rownames('X')
colnames(df_per_sample) <- str_remove_all(colnames(df_per_sample),'X')
df_ranked_per_sample <- apply(-abs(df_per_sample),1,rank)
df_ranked_per_sample <- df_ranked_per_sample/nrow(df_per_sample)
#df_aggragated <- apply(df_ranked_per_sample,1,median)
findTop <- function(namedRank,top=1000){
  return(names(namedRank[order(namedRank)])[1:top])
}

# data : data frame with rows of genes entrezid and columns the samples. It contains scores for each gene
#geneSets : list containing the vector with entrez id of genes of a geneset
pathway_analysis_fisher <- function(data,geneSets,top=1000,perSample = TRUE,
                                    level=0.05,p.adj = TRUE,adj.method='BH',perc_samples=0.5){
  #data_ranked <- rank(-abs(data))
  #data_ranked <- data_ranked/nrow(data)
  #names(data_ranked) <- names(data)
  
  data_ranked <- apply(-abs(data),1,rank)
  data_ranked <- data_ranked/nrow(data)
  
  
  topgenes <- apply(data_ranked,2,findTop)
  
  fisherExactTest <- function(geneSet,Top,measGenes){
    not_in_geneset = sum(!(measGenes %in% geneSet ))
    contigencyMat <- data.frame('dex'=c(sum(geneSet %in% Top),length(geneSet)-sum(geneSet %in% Top)),
                                'not_dex'= c(sum(!(Top %in% geneSet)),not_in_geneset-sum(!(Top %in% geneSet))))
    rownames(contigencyMat) <- c('in_set','not_in_set')
    return(fisher.test(contigencyMat)$p.value)
  }
  pathEnrich <- function(Top,geneSets,measGenes){
    return(unlist(lapply(geneSets,fisherExactTest,Top,measGenes)))
  }
  
  get_signigicant_counts <- function(pvals,level=0.05){
    return(length(which(pvals<level))/length(pvals))
  }
  
  if (perSample==T){
    kegg_pvals <- apply(topgenes,2,pathEnrich,geneSets,rownames(data_ranked))
    if (p.adj==T){
      kegg_pvals <- apply(kegg_pvals,2,p.adjust,adj.method)
    }
    significants <- apply(kegg_pvals, 1 ,get_signigicant_counts,level=level)
    significants <- which(significants>=perc_samples)
    sig_paths <- rownames(kegg_pvals)[significants]
  } else{
    topgenes <- unique(as.vector(topgenes))
    kegg_pvals <- pathEnrich(topgenes,geneSets,rownames(data_ranked))
    if (p.adj==T){
      kegg_pvals <- p.adjust(kegg_pvals,adj.method)
    }
    sig_paths <- names(kegg_pvals[which(kegg_pvals<level)])
  }
  return(list(kegg_pvals,sig_paths))
}
paths <- pathway_analysis_fisher(df_per_sample,human_keggs,top=1000,p.adj = T,adj.method='BH',perc_samples=0.5)
print(paths[[2]])

kegg_pvals <- paths[[1]]
kegg_paths <- paths[[2]]
## Important to encode in the common latent space----------------
important_forCommon_1 <- data.table::fread('../results/Importance_results/important_scores_pc3_to_encode.csv',header=T) %>% column_to_rownames('V1')
important_forCommon_2 <- data.table::fread('../results/Importance_results/important_scores_ha1e_to_encode.csv',header=T) %>% column_to_rownames('V1')

### kmeans approach
kmeans_1 <- kmeans(as.matrix(imp_enc_1[,1:ncol(imp_enc_1)-1]),centers = 3,iter.max = 1000, nstart = 200)
df_1 <-  data.frame(gene=rownames(imp_enc_1),mean_score = imp_enc_1$mean_imp,cluster =kmeans_1$cluster)
df_1_summary <- df_1 %>% group_by(cluster) %>% summarise(counts = n()) %>% arrange(counts)
cl1 <- df_1_summary$cluster[1]
cl2 <- df_1_summary$cluster[2]
hist(df_1$mean_score[df_1$cluster==cl1])
hist(df_1$mean_score[df_1$cluster==cl2])
if (mean(df_class_2$score[df_class_2$cluster==cl1])>mean(df_class_2$score[df_class_2$cluster==cl2])){
  df_1 <- df_1 %>% filter(cluster==cl1)
} else if (mean(df_class_2$score[df_class_2$cluster==cl1])<mean(df_class_2$score[df_class_2$cluster==cl2])){
  df_1 <- df_1 %>% filter(cluster==cl2)
}else{
  print('Cannot discern two cell-lines')
}
kmeans_2 <- kmeans(as.matrix(imp_enc_2[,1:ncol(imp_enc_2)-1]),centers = 3,iter.max = 1000, nstart = 200)
df_2 <-  data.frame(gene=rownames(imp_enc_2),mean_score = imp_enc_2$mean_imp,cluster =kmeans_2$cluster)
df_2_summary <- df_2 %>% group_by(cluster) %>% summarise(counts = n()) %>% arrange(counts)
cl1 <- df_2_summary$cluster[1]
cl2 <- df_2_summary$cluster[2]
hist(df_2$mean_score[df_2$cluster==cl1])
hist(df_2$mean_score[df_2$cluster==cl2])
if (mean(df_class_2$score[df_class_2$cluster==cl1])<mean(df_class_2$score[df_class_2$cluster==cl2])){
  df_2 <- df_2 %>% filter(cluster==cl1)
} else if (mean(df_class_2$score[df_class_2$cluster==cl1])>mean(df_class_2$score[df_class_2$cluster==cl2])){
  df_2 <- df_2 %>% filter(cluster==cl2)
}else{
  print('Cannot discern two cell-lines')
}

df1 <- left_join(df_1,geneInfo,by=c("gene"="gene_id"))
g1 <- df1$gene
df2 <- left_join(df_2,geneInfo,by=c("gene"="gene_id"))
df1 <- df1 %>% filter(!(gene %in% df2$gene))
df2 <- df2 %>% filter(!(gene %in% g1))
df1 <- df1 %>% filter(mean_score>0)
df2 <- df2 %>% filter(mean_score<0)
####


## Important genes from direct ANN classifier-------------------
importance_class_1 <- data.table::fread('../results/Importance_results/important_scores_pc3_to_encode.csv',header=T) %>% column_to_rownames('V1')
importance_class_2 <- data.table::fread('../results/Importance_results/important_scores_ha1e_to_encode.csv',header=T) %>% column_to_rownames('V1')

importance_class_1 <- apply(importance_class_1,2,mean)
importance_class_2 <- apply(importance_class_2,2,mean)

top10_1 <- order(-importance_class_1)[1:10]
top10_2 <- order(-abs(importance_class_2))[1:40]

df1 <- data.frame(scores_1=importance_class_1[top10_1]) %>% rownames_to_column('Genes1')
df2 <- data.frame(scores_2=importance_class_2[top10_2]) %>% rownames_to_column('Genes2')

ggplot(df1,aes(x=Genes1,y=scores_1)) + geom_bar(stat='identity',fill='#0077b3') + xlab('Latent variable') + ylab('Importance score')+
  ggtitle('Top 20 important latent variables for cell classification') + 
  theme(text = element_text(family = "serif",size = 14))
#ggplot(df2,aes(x=Genes2,y=scores_2)) + geom_bar(stat='identity')

var_1 <- names(importance_class_1)[top10_1[1]]
var_2 <- names(importance_class_2)[top10_2[1]]

p <- ggplot(gex,aes(x=`1026`,y=`1475`,col=cell)) + geom_point()+
  ggtitle('Scatter plot for 2 cell-lines')+
  theme(text = element_text(size=13))
print(p)

library(plotly)
fig <- plot_ly(gex, x = ~`1475`, y = ~`5950`, z = ~`11098`, color = ~cell, colors = c('#BF382A', '#0C4B8E'))
fig <- fig %>% add_markers(size=1)
fig

## Do the same plot for 10fold validation
library(tidyverse)
library(ggplot2)
library(ggpubr)
plotList <- NULL
for (i in 0:9){
  importance_class_1 <- data.table::fread(paste0('../results/MI_results/embs/TwoEncoders_TwoDecoders_PC3_HA1E_withclass/scores_class_validation/important_scores_to_classify_as_pc3_val_',
                                          i,'.csv'),header=T) %>% column_to_rownames('V1')
  importance_class_2 <- data.table::fread(paste0('../results/MI_results/embs/TwoEncoders_TwoDecoders_PC3_HA1E_withclass/scores_class_validation/important_scores_to_classify_as_ha1e_val_',
                                          i,'.csv'),header=T) %>% column_to_rownames('V1')
  
  importance_class_1 <- apply(importance_class_1,2,mean)
  importance_class_2 <- apply(importance_class_2,2,mean)
  
  top10_1 <- order(-abs(importance_class_1))[1:15]
  top10_2 <- order(-abs(importance_class_2))[1:15]
  
  df1 <- data.frame(scores_1=importance_class_1[top10_1]) %>% rownames_to_column('latent1')
  df2 <- data.frame(scores_2=importance_class_2[top10_2]) %>% rownames_to_column('latent2')
  
  # ggplot(df1,aes(x=latent1,y=scores_1)) + geom_bar(stat='identity',fill='#0077b3') + xlab('Latent variable') + ylab('Importance score')+
  #   ggtitle('Top 20 important latent variables for cell classification') + 
  #   theme(text = element_text(family = "serif",size = 14))
  # ggplot(df2,aes(x=latent2,y=scores_2)) + geom_bar(stat='identity')
  # 
  var_1 <- names(importance_class_1)[top10_1[1]]
  var_2 <- names(importance_class_2)[top10_1[2]]
  
  emb1 <- distinct(data.table::fread(paste0('../results/MI_results/embs/TwoEncoders_TwoDecoders_PC3_HA1E_withclass/validation/valEmbs_',i,'_pc3.csv'),
                                     header = T)) %>% column_to_rownames('V1')
  colnames(emb1) <- paste0('z',seq(0,ncol(emb1)-1))
  emb1 <- emb1 %>% mutate(cell='PC3')
  emb2 <- distinct(data.table::fread(paste0('../results/MI_results/embs/TwoEncoders_TwoDecoders_PC3_HA1E_withclass/validation/valEmbs_',i,'_ha1e.csv'),
                                     header = T)) %>% column_to_rownames('V1')
  colnames(emb2) <- paste0('z',seq(0,ncol(emb2)-1))
  emb2 <- emb2 %>% mutate(cell='HA1E')
  all_embs <- rbind(emb1,emb2)
  all_embs <- all_embs %>% select(all_of(c(var_1,var_2)),cell)
  colnames(all_embs)[1:2] <- c('z1','z2')
  p <- ggplot(all_embs,aes(x=z1,y=z2)) + geom_point(aes(color=cell)) +
    xlab(var_1) +ylab(var_2)+
    theme(text = element_text(family = "serif",size = 15),
          legend.title = element_text(size = 25, face = "plain",family='serif'),
          legend.text = element_text(size=25,face='plain',family='serif'))
  plotList[[i+1]] <- p
}
png(file="../figures/MI_results/validation_latent_separation.png",width=11,height=16,units = "in",res=600)
p <- ggarrange(plotlist=plotList,ncol=2,nrow=5,common.legend = TRUE,legend = 'bottom',
               labels = paste0(rep('Split ',10),seq(1,10)),
               font.label = list(size = 8, color = "black", face = "bold", family = NULL),
               hjust=-0.25,vjust=1.1)

annotate_figure(p, top = text_grob("Scatter plot using only 2 latent variable", 
                                   color = "black",face = 'bold', size = 16,family = "serif"))
dev.off()
