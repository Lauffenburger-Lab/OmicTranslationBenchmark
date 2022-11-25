library(tidyverse)

# df <- read.csv('../results/Importance_results/important_scores_a375_to_ht29_allgenes.csv') %>% column_to_rownames('X')
df <- read.csv('../results/Importance_results/important_scores_pc3_to_ha1e_per_sample_allgenes.csv')
df <- df %>% unique()
rownames( df ) <- NULL
df <- df %>% column_to_rownames('X')
colnames(df) <- str_remove_all(colnames(df),'X')
df_ranked <- apply(-abs(df),1,rank)
df_ranked <- df_ranked/nrow(df)
df_aggragated <- apply(df_ranked,1,median)
top1000 <- names(df_aggragated[order(df_aggragated)])[1:1000]
#top1000 <- str_remove_all(top1000,'X')
df_lands <- read.csv('../results/Importance_results/important_scores_a375_to_ht29_lands.csv')  %>% column_to_rownames('X')
#df_ranked <- apply(-abs(df),2,rank)
#df_ranked <- df_ranked/nrow(df)
#png('../figures/ranks_of_self_a375_translate_allgenes.png',width=16,height=8,units = "in",res=300)
#hist(diag(df_ranked*100),breaks = 40,main= 'Distribution of self-gene ranks in ~10k genes',xlab='Percentage rank (%)')
#axis(side=1, at=seq(0,100, 10), labels=seq(0,100, 10))
#dev.off()
#plot(ecdf(diag(df_ranked*100)),main='Cumulative probability distribution',xlab='Percentage rank (%)')

geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
lands <- geneInfo %>% filter(feature_space=='landmark')
lands <- as.character(lands$gene_id)

#df_aggragated <- apply(df_ranked,1,median)
#top1000 <- names(df_aggragated[order(df_aggragated)])[1:1000]

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

#### Use per sample analysis
for (i in 1:nrow(cmap)){
  #gradients <- data.table::fread(paste0('../results/ImportantGenesResults/gradient_scores_allgenes_a375_to_ht29_pairedsample',
  #                                      i,'.csv'),header=T) %>% column_to_rownames('V1')
  #grad_rank <- apply(-abs(grad_rank),2,rank)
  #grad_rank <- grad_rank/nrow(gradients)
  GeX <- cmap[i,]
  genes_regulated <- names(GeX)[order(GeX,decreasing = T)][1:1000]
  
  percentage_of_importants_in_top_regulated <- apply(df_ranked,2,FindPercentageIntersection,DeXs=genes_regulated)
  mean_perc[i] <- mean(percentage_of_importants_in_top_regulated)
  sd_perc[i] <- sd(percentage_of_importants_in_top_regulated)
}

# percentage_of_importants_in_top_regulated <- NULL
# for (j in 1:ncol(df_rank)){
#   df_gene <- df_rank[,j]
#   top1000 <- names(df_gene[order(df_gene)])[1:1000]
#   percentage_of_importants_in_top_regulated[j] <- length(which(top1000 %in% genes_regulated))/length(top1000)
# }

png('../figures/mean_percentage_of_important_in_1000gex_pc3_to_ha1e.png',width=16,height=8,units = "in",res=300)
hist(mean_perc*100,breaks = 40,
     main= 'Average percentages of top 1000 important genes per sample present in top 1000 expressed genes',xlab='Percentage (%)')
dev.off()

png('../figures/std_percentage_of_important_in_1000gex_pc3_to_ha1e.png',width=16,height=8,units = "in",res=300)
hist(sd_perc*100,breaks = 40,
     main= 'Percentages s.d. of top 1000 important genes per sample present in top 1000 expressed genes',xlab='Percentage (%)')
dev.off()

png('../figures/cv_percentage_of_important_in_1000gex_pc3_to_ha1e.png',width=16,height=8,units = "in",res=300)
hist(100*sd_perc/mean_perc,breaks = 40,
     main= 'Coefficent of variation of number of top 1000 important genes in most expressed genes',xlab='CV (%)')
dev.off()

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

### Perform pathway analysis in important----

# Per sample analysis
df_per_sample <- read.csv('../results/Importance_results/important_scores_ha1e_to_pc3_per_sample_allgenes.csv')
df_per_sample <- df_per_sample %>% unique()
rownames( df_per_sample ) <- NULL
df_per_sample <- df_per_sample %>% column_to_rownames('X')
colnames(df_per_sample) <- str_remove_all(colnames(df_per_sample),'X')
df_ranked_per_sample <- apply(-abs(df_per_sample),1,rank)
df_ranked_per_sample <- df_ranked_per_sample/nrow(df_per_sample)
#df_aggragated <- apply(df_ranked_per_sample,1,median)

library(EGSEAdata)
egsea.data(species = "human",returnInfo = TRUE)
human_keggs <- kegg.pathways$human$kg.sets

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

### Importance scores to encode-----
library(tidyverse)
library(ggVennDiagram)
library(ggplot2)
gex <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header = T) %>% column_to_rownames('V1')
scores_1 <- data.table::fread('../results/Importance_results/important_scores_pc3_encode_per_sample_allgenes.csv',header = T) 
scores_1 <- distinct(scores_1)
rownames( scores_1 ) <- NULL
scores_1 <- scores_1 %>% column_to_rownames('V1')
scores_2 <- data.table::fread('../results/Importance_results/important_scores_ha1e_encode_per_sample_allgenes.csv',header = T) 
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
all_embs <- all_embs %>% select(-cell)

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

### Find important genes to encode into each latent variable important for classification
importance_class_1 <- data.table::fread('../results/Importance_results/important_scores_to_classify_as_pc3_cpa.csv',header=T) %>% column_to_rownames('V1')
importance_class_2 <- data.table::fread('../results/Importance_results/important_scores_to_classify_as_ha1e_cpa.csv',header=T) %>% column_to_rownames('V1')

importance_class_1 <- apply(importance_class_1,2,mean)
importance_class_2 <- apply(importance_class_2,2,mean)

top10_1 <- order(-abs(importance_class_1))[1:10]
top10_2 <- order(-abs(importance_class_2))[1:10]

df1 <- data.frame(scores_1=importance_class_1[top10_1]) %>% rownames_to_column('Genes1')
df2 <- data.frame(scores_2=importance_class_2[top10_2]) %>% rownames_to_column('Genes2')

ggplot(df1,aes(x=Genes1,y=scores_1)) + geom_bar(stat='identity',fill='#0077b3') + xlab('Latent variable') + ylab('Importance score')+
  ggtitle('Top 20 important latent variables for cell classification') + 
  theme(text = element_text(family = "serif",size = 14))
ggplot(df2,aes(x=Genes2,y=scores_2)) + geom_bar(stat='identity')

var_1 <- names(importance_class_1)[top10_1[1:10]]
var_2 <- names(importance_class_2)[top10_2[1:10]]

emb1 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_CPA_pc3.csv',header = T)) %>% column_to_rownames('V1')
emb1 <- emb1 %>% mutate(cell='PC3')
emb2 <- distinct(data.table::fread('../results/trained_embs_all/AllEmbs_CPA_ha1e.csv',header = T)) %>% column_to_rownames('V1')
emb2 <- emb2 %>% mutate(cell='HA1E')
all_embs <- rbind(emb1,emb2)

# x=`z1009`,y=`z263` for AE with classifier only and MI not CPA
ggplot(all_embs,aes(x=`z853`,y=`z465`)) + geom_point(aes(color=cell)) +
  ggtitle('Scatter plot using only 2 latent variable') +theme(text = element_text(family = "serif",size = 14))

# Load importance to encode
#important_scores_pc3_encode_allgenes_withclass_noabs.csv and var_x[1:2]
imp_enc_1 <- data.table::fread('../results/Importance_results/important_scores_pc3_to_encode_cpa.csv') %>%
  select(c('Gene_1'='V1'),all_of(var_1[1:2])) %>% column_to_rownames('Gene_1')
imp_enc_1 <- imp_enc_1 %>% mutate(mean_imp=rowMeans(imp_enc_1))
imp_enc_2 <- data.table::fread('../results/Importance_results/important_scores_ha1e_to_encode_cpa.csv')%>%
  select(c('Gene_2'='V1'),all_of(var_2[1:2])) %>% column_to_rownames('Gene_2')
imp_enc_2 <- imp_enc_2 %>% mutate(mean_imp=rowMeans(imp_enc_2))


gex <- data.table::fread('../preprocessing/preprocessed_data/cmap_HA1E_PC3.csv',header = T) %>% column_to_rownames('V1')
ind1 <- grep('PC3',rownames(gex))
ind2 <- grep('HA1E',rownames(gex))
gex <- gex %>% mutate(cell='None')
gex$cell[ind1] <- 'PC3'
gex$cell[ind2] <- 'HA1E'

p <- ggplot(gex,aes(x=`23636`,y=`1465`)) + geom_point(aes(color=cell))+
  ggtitle('Scatter plot of transcriptomic data for 2 cell-lines') +
  theme(text = element_text(size=13))
print(p)
png('../figures/2dscatterplot_pc3_ha1e_gex_withclass.png',width=10,height = 10,units = "in",res=300)
print(p)
dev.off()

library(plotly)
fig <- plot_ly(gex, x = ~`5997`, y = ~`3162`, z = ~`84617`, color = ~cell, colors = c('#BF382A', '#0C4B8E'))
fig <- fig %>% add_markers(size=1)
fig

## Get top 100 (50 for every cell) genes important to encode into cell-associated latent space
#imp_enc_1 <- imp_enc_1 %>% column_to_rownames('Gene_1')
#imp_enc_2 <- imp_enc_2 %>% column_to_rownames('Gene_2')

# to eixa order by absolute value kai sta 2
top50_1 <- rownames(imp_enc_1)[order(-imp_enc_1$mean_imp)][1:50]
top50_2 <- rownames(imp_enc_2)[order(-imp_enc_2$mean_imp)][1:50]

df1 <- as.data.frame(top50_1)
df2 <- as.data.frame(top50_2)
geneInfo$gene_id <- as.character(geneInfo$gene_id) 
df1 <- left_join(df1,geneInfo,by=c("top50_1"="gene_id"))
df2 <- left_join(df2,geneInfo,by=c("top50_2"="gene_id"))

gex_filtered <- gex %>% select(all_of(unique(c(top50_1,top50_2))),cell)

pca_filt <- prcomp(gex_filtered[,1:(ncol(gex_filtered)-1)],scale = F)
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
p <- ggplot(gex,aes(x=`8942`,y=`1906`,col=cell)) + geom_point()+
  ggtitle('Scatter plot for 2 cell-lines')+
  theme(text = element_text(size=13))
print(p)

library(plotly)
fig <- plot_ly(gex_filtered, x = ~`8942`, y = ~`1906`, z = ~`7867`, color = ~cell, colors = c('#BF382A', '#0C4B8E'))
fig <- fig %>% add_markers(size=1)
fig

## Important genes from direct ANN classifier
importance_class_1 <- data.table::fread('../results/Importance_results/important_scores_to_classify_gex_as_pc3.csv',header=T) %>% column_to_rownames('V1')
importance_class_2 <- data.table::fread('../results/Importance_results/important_scores_to_classify_gex_as_ha1e.csv',header=T) %>% column_to_rownames('V1')

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
