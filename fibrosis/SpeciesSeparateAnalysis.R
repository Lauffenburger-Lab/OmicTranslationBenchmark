library(tidyverse)
library(reshape2)
library(gg3D)
library(ggVennDiagram)
library(ggsignif)
library(ggpubr)
library(caret)
library(irlba)
library(factoextra)
library(plotly)

# Load all human and mouse data and preprocess/scale them------
homologues_summary <- data.table::fread('results/HumanMouseHomologuesMap.csv') %>% select(-V1)
mouse_data <- data.table::fread('data/all_mouse_lung.csv') %>% select(-V1)
human_data <- data.table::fread('data/all_human_lung.csv') %>% select(-V1)
gc()

human_scaled <- as.matrix(human_data %>% select(-cell_type,-Species,-Diagnosis))
human_scaled <- (10^6) * human_scaled/sum(human_scaled)
#hist(human_scaled)
gc()
human_scaled <- log10(human_scaled+0.5)
#hist(human_scaled)
gc()
#human_scaled <- scale(human_scaled,scale = F)
##hist(human_scaled)
#gc()

human_labels <-  human_data %>% select(cell_type,Species,Diagnosis)
human_data <- NULL
gc()

mouse_scaled <- as.matrix(mouse_data %>% select(-cell_type,-Species,-Diagnosis))
mouse_scaled <- (10^6) * mouse_scaled/sum(mouse_scaled)
mouse_scaled <- log10(mouse_scaled+0.5)
mouse_scaled <- scale(mouse_scaled)
gc()

mouse_labels <-  mouse_data %>% select(cell_type,Species,Diagnosis)
mouse_data <- NULL
gc()


# Perform PCA analysis separately for each species--------------------
#human_pca <- prcomp_irlba(human_scaled,n=200)
#mouse_pca <- prcomp_irlba(mouse_scaled,n=200)
#gc()
#saveRDS(human_pca,'results/human_pca.rds')
#saveRDS(mouse_pca,'results/mouse_pca.rds')
human_pca <- readRDS('results/human_pca.rds')
gc()

fviz_eig(human_pca, addlabels = TRUE,ncp = 20)
summ <- summary(human_pca)

df_pca<- human_pca$x[,1:3]
df_pca <- as.data.frame(df_pca)
colnames(df_pca) <- c('PC1','PC2','PC3')
df_pca <- df_pca %>% mutate(diagnosis=human_labels$Diagnosis)
df_pca <- df_pca %>% mutate(diagnosis=ifelse(diagnosis==1,'fibrosis','normal'))
df_pca$diagnosis <- factor(df_pca$diagnosis,levels = c('normal','fibrosis'))
df_pca <- df_pca %>% mutate(cell=human_labels$cell_type)
df_pca <- df_pca %>% mutate(cell=ifelse(cell==1,'immune',
                                        ifelse(cell==2,'mesenchymal',
                                               ifelse(cell==3,'epithelial',
                                                      ifelse(cell==4,'endothelial','stem cell')))))

pca_plot <- ggplot(df_pca,aes(PC1,PC2)) +geom_point(aes(col=diagnosis,shape=cell))+
  scale_color_manual(values = c('#4878CF','#D65F5F'))+
  ggtitle('PCA plot of human single cell data') +
  xlab(paste0('PC1'))+ ylab(paste0('PC2'))+theme_minimal()+
  theme(text = element_text(size=16),plot.title = element_text(hjust = 0.5),
        legend.text=element_text(size=16))
print(pca_plot)
ggsave('/results/pca_2d_human_gex.eps', 
       device = cairo_ps,
       scale = 1,
       width = 12,
       height = 8,
       units = "in",
       dpi = 600)
plot_ly(df_pca) %>% add_trace(x = ~PC1, y = ~PC2, z = ~PC3,color = ~diagnosis,shape=~cell,
                        type = "scatter3d",
                        mode = "markers",
                        colors = c('#4878CF','#D65F5F'),
                        showlegend = TRUE,
                        size = 2
)
# It seems ~3 (or even 2 PCs) are enough to explain the diagnosis in humans

# ## Now do the same for mouse
# fviz_eig(mouse_pca, addlabels = TRUE,ncp = 20)
# summ <- summary(mouse_pca)
# 
# df_pca<- mouse_pca$x[,1:3]
# df_pca <- as.data.frame(df_pca)
# colnames(df_pca) <- c('PC1','PC2','PC3')
# df_pca <- df_pca %>% mutate(diagnosis=mouse_labels$Diagnosis)
# df_pca <- df_pca %>% mutate(diagnosis=ifelse(diagnosis==1,'fibrosis','normal'))
# df_pca$diagnosis <- factor(df_pca$diagnosis,levels = c('normal','fibrosis'))
# df_pca <- df_pca %>% mutate(cell=mouse_labels$cell_type)
# df_pca <- df_pca %>% mutate(cell=ifelse(cell==1,'immune',
#                                         ifelse(cell==2,'mesenchymal',
#                                                ifelse(cell==3,'epithelial',
#                                                       ifelse(cell==4,'endothelial','stem cell')))))
# 
# pca_plot <- ggplot(df_pca,aes(PC1,PC2)) +geom_point(aes(col=diagnosis,shape=cell))+
#   scale_color_manual(values = c('#4878CF','#D65F5F'))+
#   ggtitle('PCA plot of mouse single cell data') +
#   xlab(paste0('PC1'))+ ylab(paste0('PC2'))+theme_minimal()+
#   theme(text = element_text(size=16),plot.title = element_text(hjust = 0.5),
#         legend.text=element_text(size=16))
# print(pca_plot)
# ggsave('/results/pca_2d_mouse_gex.eps', 
#        device = cairo_ps,
#        scale = 1,
#        width = 12,
#        height = 8,
#        units = "in",
#        dpi = 600)
# plot_ly(df_pca) %>% add_trace(x = ~PC1, y = ~PC2, z = ~PC3,color = ~diagnosis,shape=~cell,
#                               type = "scatter3d",
#                               mode = "markers",
#                               colors = c('#4878CF','#D65F5F'),
#                               showlegend = TRUE,
#                               size = 2
# )

## Start building logistic moles with less and less  PCs----------
df_pca<- human_pca$x[,1:200]
df_pca <- as.data.frame(df_pca)
colnames(df_pca) <- paste0('PC',seq(1:200))
df_pca <- df_pca %>% mutate(diagnosis=human_labels$Diagnosis)
df_pca$diagnosis <- factor(df_pca$diagnosis,levels = c(0,1))
df_pca <- df_pca %>% mutate(cell=human_labels$cell_type)
df_pca$cell <- factor(df_pca$cell,levels = c(1,2,3,4))
# Train test split
train_indices <- createDataPartition(df_pca$diagnosis, p = 0.75, list = FALSE)
# Subset your data into training and testing sets based on the indices
train_data <- df_pca[train_indices, ]
test_data <- df_pca[-train_indices, ]

pcs_to_keep <- c(200,150,100,75,50,40,30,25,20,15,10,5,3,2,1)
F1 <- NULL
for (i in 1:length(pcs_to_keep)){
  df <- train_data %>% select(all_of(c(paste0('PC',seq(1:pcs_to_keep[i])),'diagnosis')))
  # Define training control method
  ctrl <- trainControl(method = "cv", number = 10)
  mdl <- train(diagnosis ~ ., data = df, method = "glm", trControl = ctrl,family = "binomial",trace=F)
  y <- predict(mdl,newdata =test_data %>% select(all_of(c(paste0('PC',seq(1:pcs_to_keep[i]))))))
  conf <- confusionMatrix(test_data$diagnosis,y)
  F1[i] <- conf$byClass['F1']
  message(paste0('Done PCs ',pcs_to_keep[i]))
}

#saveRDS(F1,'/PCGlm_F1.rds')

pc_results <- data.frame(PCs_number=pcs_to_keep,F1=F1)
ggplot(pc_results,aes(x=PCs_number,y=F1*100)) + geom_point(color='black') + 
  geom_smooth(se=F,color='#4878CF') + ylim(c(0,100)) +
  geom_hline(yintercept = 90,color='red',lty='dashed',linewidth=1) + 
  annotate('text',x=53,y=85,label = "90% F1 threshold",size=6)+
  xlab(paste0('number of PCs used'))+ ylab(paste0('F1 score (%)'))+theme_minimal()+
  ggtitle('GLM performance for classifying disease state in humans')+
  theme(text = element_text(size=16),plot.title = element_text(hjust = 0.5),
        legend.text=element_text(size=16))
  
ggsave('/results/PC_glm_human_fibrosis.eps', 
       device = cairo_ps,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
png('/results/PC_glm_human_fibrosis.png',units = 'in',width = 9,height = 9,res = 600)
ggplot(pc_results,aes(x=PCs_number,y=F1*100)) + geom_point(color='black') + 
  geom_smooth(se=F,color='#4878CF') + ylim(c(0,100)) +
  geom_hline(yintercept = 90,color='red',lty='dashed',linewidth=1) + 
  annotate('text',x=53,y=85,label = "90% F1 threshold",size=6)+
  xlab(paste0('number of PCs used'))+ ylab(paste0('F1 score (%)'))+theme_minimal()+
  ggtitle('GLM performance for classifying disease state in humans')+
  theme(text = element_text(size=16),plot.title = element_text(hjust = 0.5),
        legend.text=element_text(size=16))
dev.off()

# Check loadings for the first 15 PCs
# Total contribution on PC1 and PC2
fviz_contrib(human_pca, choice = "var", axes = 1:15,top = 100)
contrib <- get_pca_var(human_pca)
contrib <- contrib$contrib
contrib <- contrib[,1:15]
contrib <- rowMeans(contrib)
ordered_genes <- colnames(human_scaled)[order(-contrib)]

overlap_non_homologues_top5 <-  1-length(intersect(homologues_summary$human_gene,ordered_genes[1:5]))/5
overlap_non_homologues_top10 <-  1-length(intersect(homologues_summary$human_gene,ordered_genes[1:10]))/10 
overlap_non_homologues_top50 <-  1-length(intersect(homologues_summary$human_gene,ordered_genes[1:50]))/50 
overlap_non_homologues_top100 <-  1-length(intersect(homologues_summary$human_gene,ordered_genes[1:100]))/100
overlap_non_homologues_top200 <-  1-length(intersect(homologues_summary$human_gene,ordered_genes[1:200]))/200

# Venn diagram with homologues
x <- list('top 10 PC-contributing genes' = ordered_genes[1:10],
          'top 3 PC-contributing genes' = ordered_genes[1:3],
          'homologue genes' = homologues_summary$human_gene)
# 2D Venn diagram
v1 <- ggVennDiagram(x,label = "count") + scale_x_continuous(expand = expansion(mult = .22))+
  scale_fill_continuous(trans = "log1p") + 
  scale_color_manual(values = c('black','black','black'))+
  theme(legend.position = "none") 
print(v1)
# make the same only for contribution in the 2nd PC
contrib <- get_pca_var(human_pca)
contrib <- contrib$contrib
contrib <- contrib[,2]
ordered_genes <- colnames(human_scaled)[order(-contrib)]
x <- list('top 10 PC-contributing genes' = ordered_genes[1:10],
          'top 3 PC-contributing genes' = ordered_genes[1:3],
          'homologue genes' = homologues_summary$human_gene)
v2 <- ggVennDiagram(x,label = "count") + scale_x_continuous(expand = expansion(mult = .22))+
  scale_fill_continuous(trans = "log1p") + 
  scale_color_manual(values = c('black','black','black'))+
  theme(legend.position = "none") 
print(v2)
p <- ggarrange(plotlist=list(v1,v2),ncol=2,nrow=1,common.legend = TRUE,legend = 'none',
               labels = c('Contributing in 15 PCs','Contributing only in PC2'),
               font.label = list(size = 20, color = "black", face = "plain", family = 'Arial'),
               hjust=-1,vjust=6)
annotate_figure(p, top = text_grob("Overlap between diagnosis relevent genes and homologue genes in human", 
                                   color = "black",face = 'plain', size = 20),)

ggsave(
  '/results/venn_homologues_important_genes.eps', 
  device = cairo_ps,
  scale = 1,
  width = 16,
  height = 9,
  units = "in",
  dpi = 600,
)
png(file="/results/venn_homologues_important_genes.png",width=16,height=9,units = "in",res=600)
annotate_figure(p, top = text_grob("Overlap between diagnosis relevent genes and homologue genes in human", 
                                   color = "black",face = 'plain', size = 20),)
dev.off()

# Build a GLM only using genes by going down the rank list of genes until 1000 genes are included----------
contrib <- get_pca_var(human_pca)
contrib <- contrib$contrib
contrib <- contrib[,1:15]
contrib <- rowMeans(contrib)
ordered_genes <- colnames(human_scaled)[order(-contrib)]
model_human_data <- cbind(human_scaled,human_labels %>% select(c('diagnosis'='Diagnosis')))
model_human_data$diagnosis <- factor(model_human_data$diagnosis,levels = c(0,1))
model_human_data <- model_human_data %>% mutate(cell=human_labels$cell_type)
model_human_data$cell <- factor(model_human_data$cell,levels = c(1,2,3,4))
# Train test split
train_indices <- createDataPartition(df_pca$diagnosis, p = 0.75, list = FALSE)
# Subset your data into training and testing sets based on the indices
train_data <- model_human_data[train_indices, ]
test_data <- model_human_data[-train_indices, ]
gc()

genes_to_keep <- c(3,5,10,15,20,30,50,75,100,200,300,500,700,1000,1500,2000,2500,3000,4000,5000)
F1 <- NULL
for (i in 1:length(genes_to_keep)){
  df <- train_data %>% select(all_of(c(ordered_genes[genes_to_keep[i]],'diagnosis','cell')))
  # Define training control method
  ctrl <- trainControl(method = "cv", number = 10)
  mdl <- train(diagnosis ~ .:cell, data = df, method = "glm", trControl = ctrl,family = "binomial",trace=F)
  y <- predict(mdl,newdata =test_data %>% select(all_of(c(ordered_genes[genes_to_keep[i]],'cell'))))
  conf <- confusionMatrix(test_data$diagnosis,y)
  F1[i] <- conf$byClass['F1']
  message(paste0('Done top ',genes_to_keep[i],' genes'))
}

gene_results <- data.frame(genes_number=genes_to_keep,F1=F1)
ggplot(gene_results %>% filter(!is.na(F1)),aes(x=genes_number,y=F1*100)) + geom_point(color='black') + geom_line(color='#4878CF')+ylim(c(0,100))+
  #geom_smooth(se=F,color='#4878CF') + ylim(c(0,100)) +
  #geom_hline(yintercept = 90,color='red',lty='dashed',linewidth=1) + 
  #annotate('text',x=53,y=85,label = "90% F1 threshold",size=6)+
  xlab(paste0('number of genes used'))+ ylab(paste0('F1 score (%)'))+theme_minimal()+
  ggtitle('GLM performance for classifying disease state in humans')+
  theme(text = element_text(size=16),plot.title = element_text(hjust = 0.5),
        legend.text=element_text(size=16))
