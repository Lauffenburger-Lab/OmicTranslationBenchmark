library(tidyverse)
library(reshape2)
library(gg3D)
library(ggsignif)
library(ggpubr)
library(caret)

### First evaluate performance in fibrosis---------
#human_embs <- data.table::fread('results/embs/LiverEmbs_human.csv') %>% select(-V1)
#mouse_embs <- data.table::fread('results/embs/LiverEmbs_mouse.csv') %>% select(-V1)

ensemble_mouse <- NULL
acc_mouse <- c()
sens_mouse <- c()
spec_mouse <- c()
f1_mouse <- c()
ensemble_human <- NULL
acc_human <- c()
sens_human <- c()
spec_human <- c()
f1_human <- c()
for (i in 0:9){
  human_embs <- data.table::fread(paste0('results/embs/liver/LiverEmbs_human_',
                                         i,'.csv')) %>% select(-V1)
  mouse_embs <- data.table::fread(paste0('results/embs/liver/LiverEmbs_mouse_',
                                         i,'.csv')) %>% select(-V1)
  cf_matrix_mouse <- confusionMatrix(data=as.factor(mouse_embs$fibrosis_pred), 
                                     reference = as.factor(mouse_embs$fibrosis_true),
                                     positive = '1')
  acc_mouse[i+1] <- cf_matrix_mouse$overall['Accuracy']
  sens_mouse[i+1] <- cf_matrix_mouse$byClass['Sensitivity']
  spec_mouse[i+1] <- cf_matrix_mouse$byClass['Specificity']
  f1_mouse[i+1] <-  cf_matrix_mouse$byClass['F1']
  ensemble_mouse[[i+1]] <-  cf_matrix_mouse$table
  
  cf_matrix_human <- confusionMatrix(data=as.factor(human_embs$fibrosis_pred), 
                                     reference = as.factor(human_embs$fibrosis_true),
                                     positive = '1')
  acc_human[i+1] <- cf_matrix_human$overall['Accuracy']
  sens_human[i+1] <- cf_matrix_human$byClass['Sensitivity']
  spec_human[i+1] <- cf_matrix_human$byClass['Specificity']
  f1_human[i+1] <-  cf_matrix_human$byClass['F1']
  ensemble_human[[i+1]] <-  cf_matrix_human$table
}
# Calculate mean and standard error of confusion matrix counts
ensemble_mouse_mat <- do.call(cbind,ensemble_mouse)
ensemble_mouse_mat <- array(ensemble_mouse_mat,c(dim=dim(ensemble_mouse[[1]]),length(ensemble_mouse)))
ensemble_mouse_mean <- apply(ensemble_mouse_mat, c(1,2), mean, na.rm = TRUE)
colnames(ensemble_mouse_mean) <- colnames(ensemble_mouse[[1]])
rownames(ensemble_mouse_mean) <- rownames(ensemble_mouse[[1]])
ensemble_mouse_sd <- apply(ensemble_mouse_mat, c(1,2), sd, na.rm = TRUE)
colnames(ensemble_mouse_sd) <- colnames(ensemble_mouse[[1]])
rownames(ensemble_mouse_sd) <- rownames(ensemble_mouse[[1]])
ensemble_mouse_mean <- melt(ensemble_mouse_mean, variable.name = c("Reference", "Prediction"), value.name = "Count") %>% 
  mutate(Var1=ifelse(Var1==0,'no-fibrosis','fibrosis')) %>%
  mutate(Var2=ifelse(Var2==0,'no-fibrosis','fibrosis')) %>% 
  mutate(Var2=factor(Var2,levels=c('no-fibrosis','fibrosis')))
ensemble_mouse_sd <- melt(ensemble_mouse_sd, variable.name = c("Reference", "Prediction"), value.name = "Count_sd") %>% 
  mutate(Var1=ifelse(Var1==0,'no-fibrosis','fibrosis')) %>%
  mutate(Var2=ifelse(Var2==0,'no-fibrosis','fibrosis')) %>% 
  mutate(Var2=factor(Var2,levels=c('no-fibrosis','fibrosis')))
ensemble_mouse_final <- left_join(ensemble_mouse_mean,ensemble_mouse_sd)

ensemble_human_mat <- do.call(cbind,ensemble_human)
ensemble_human_mat <- array(ensemble_human_mat,c(dim=dim(ensemble_human[[1]]),length(ensemble_human)))
ensemble_human_mean <- apply(ensemble_human_mat, c(1,2), mean, na.rm = TRUE)
colnames(ensemble_human_mean) <- colnames(ensemble_human[[1]])
rownames(ensemble_human_mean) <- rownames(ensemble_human[[1]])
ensemble_human_sd <- apply(ensemble_human_mat, c(1,2), sd, na.rm = TRUE)
colnames(ensemble_human_sd) <- colnames(ensemble_human[[1]])
rownames(ensemble_human_sd) <- rownames(ensemble_human[[1]])
ensemble_human_mean <- melt(ensemble_human_mean, variable.name = c("Reference", "Prediction"), value.name = "Count") %>% 
  mutate(Var1=ifelse(Var1==0,'no-fibrosis','fibrosis')) %>%
  mutate(Var2=ifelse(Var2==0,'no-fibrosis','fibrosis')) %>% 
  mutate(Var2=factor(Var2,levels=c('no-fibrosis','fibrosis')))
ensemble_human_sd <- melt(ensemble_human_sd, variable.name = c("Reference", "Prediction"), value.name = "Count_sd") %>% 
  mutate(Var1=ifelse(Var1==0,'no-fibrosis','fibrosis')) %>%
  mutate(Var2=ifelse(Var2==0,'no-fibrosis','fibrosis')) %>% 
  mutate(Var2=factor(Var2,levels=c('no-fibrosis','fibrosis')))
ensemble_human_final <- left_join(ensemble_human_mean,ensemble_human_sd)
# Create plot
plot_fibrosis_mouse <- ggplot(ensemble_mouse_final, 
       aes(x = Var1, y = Var2, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, max(ensemble_mouse_final$Count))) +
  geom_text(aes(label = paste(round(Count, 3), round(Count_sd/sqrt(10), 2), sep = "\u00B1")), size = 6) +
  annotate('text',x = 1.5,y=1.5,
           label=paste0(paste0('F1 score = ',paste(round(mean(f1_mouse)*100,2),round(sd(f1_mouse)*100/sqrt(10),2),sep="\u00B1"),'%'),
                        "\n",
                        paste0('Accuracy = ',paste(round(mean(acc_mouse)*100,2),round(sd(acc_mouse)*100/sqrt(10),2),sep="\u00B1"),'%')),
           size=8,
           fontface =2)+
  labs(title = "External Mouse Lung Fibrosis Dataset",
       x = "Predicted Class",
       y = "True Class") +
  theme_minimal() +
  theme(text = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size = 24),
        axis.title = element_text(family = 'Arial',size = 24, face = "bold"),
        legend.text = element_text(family = 'Arial',size = 24),
        plot.title = element_text(family = 'Arial',size = 24, face = "bold", hjust = 0.5))
print(plot_fibrosis_mouse)
ggsave('results/confusion_LiverFibrosis_mouse.eps', 
       device = cairo_ps,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
plot_fibrosis_human <- ggplot(ensemble_human_final, 
                              aes(x = Var1, y = Var2, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, max(ensemble_human_final$Count))) +
  geom_text(aes(label = paste(round(Count, 3), round(Count_sd/sqrt(10), 2), sep = "\u00B1")), size = 6) +
  annotate('text',x = 1.5,y=1.5,
           label=paste0(paste0('F1 score = ',paste(round(mean(f1_human)*100,2),round(sd(f1_human)*100/sqrt(10),2),sep="\u00B1"),'%'),
                        "\n",
                        paste0('Accuracy = ',paste(round(mean(acc_human)*100,2),round(sd(acc_human)*100/sqrt(10),2),sep="\u00B1"),'%')),
           size=8,
           fontface =2)+
  labs(title = "Human Liver Cirrhosis Dataset",
       x = "Predicted Class",
       y = "True Class") +
  theme_minimal() +
  theme(text = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size = 24),
        axis.title = element_text(family = 'Arial',size = 24, face = "bold"),
        legend.text = element_text(family = 'Arial',size = 24),
        plot.title = element_text(family = 'Arial',size = 24, face = "bold", hjust = 0.5))
print(plot_fibrosis_human)
ggsave('results/confusion_LiverCirrhosis_human.eps', 
       device = cairo_ps,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
png('results/confusion_LiverFibrosis_mouse.png',units = 'in',width = 8,height = 8,res=600)
print(plot_fibrosis_mouse)
dev.off()

png('results/confusion_LiverCirrhosis_human.png',units = 'in',width = 8,height = 8,res=600)
print(plot_fibrosis_human)
dev.off()

external_classification <- data.frame(f1_human,f1_mouse,acc_human,acc_mouse) %>% gather('key','value') %>%
  separate(key,c('metric','datasets'),sep='_') %>% mutate(metric=ifelse(metric=='f1','F1','accuracy')) %>% 
  mutate(task = 'disease classification') %>% 
  mutate(datasets = ifelse(datasets=='human','human liver cirrhosis','mouse lung fibrosis'))

### Species classification-----------
ensemble <- NULL
acc <- c()
sens <- c()
spec <- c()
f1 <- c()
for (i in 0:9){
  human_embs <- data.table::fread(paste0('results/embs/liver/LiverEmbs_human_',
                                         i,'.csv')) %>% select(-V1)
  mouse_embs <- data.table::fread(paste0('results/embs/liver/LiverEmbs_mouse_',
                                         i,'.csv')) %>% select(-V1)
  all_embs <- rbind(human_embs,mouse_embs)
  cf_matrix <- confusionMatrix(data=as.factor(all_embs$species_pred), 
                                     reference = as.factor(all_embs$species_true),
                                     positive = '1')
  acc[i+1] <- cf_matrix$overall['Accuracy']
  sens[i+1] <- cf_matrix$byClass['Sensitivity']
  spec[i+1] <- cf_matrix$byClass['Specificity']
  f1[i+1] <-  cf_matrix$byClass['F1']
  ensemble[[i+1]] <-  cf_matrix$table
}
# Calculate mean and standard error of confusion matrix counts
ensemble_mat <- do.call(cbind,ensemble)
ensemble_mat <- array(ensemble_mat,c(dim=dim(ensemble[[1]]),length(ensemble)))
ensemble_mean <- apply(ensemble_mat, c(1,2), mean, na.rm = TRUE)
colnames(ensemble_mean) <- colnames(ensemble[[1]])
rownames(ensemble_mean) <- rownames(ensemble[[1]])
ensemble_sd <- apply(ensemble_mat, c(1,2), sd, na.rm = TRUE)
colnames(ensemble_sd) <- colnames(ensemble[[1]])
rownames(ensemble_sd) <- rownames(ensemble[[1]])
ensemble_mean <- melt(ensemble_mean, variable.name = c("Reference", "Prediction"), value.name = "Count") %>% 
  mutate(Var1=ifelse(Var1==0,'mouse','human')) %>%
  mutate(Var2=ifelse(Var2==0,'mouse','human')) %>% 
  mutate(Var2=factor(Var2,levels=c('mouse','human')))
ensemble_sd <- melt(ensemble_sd, variable.name = c("Reference", "Prediction"), value.name = "Count_sd") %>% 
  mutate(Var1=ifelse(Var1==0,'mouse','human')) %>%
  mutate(Var2=ifelse(Var2==0,'mouse','human')) %>% 
  mutate(Var2=factor(Var2,levels=c('mouse','human')))
ensemble_final <- left_join(ensemble_mean,ensemble_sd)

plot_species <- ggplot(ensemble_final, 
                              aes(x = Var1, y = Var2, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, max(ensemble_final$Count))) +
  geom_text(aes(label = paste(round(Count, 3), round(Count_sd/sqrt(10), 2), sep = "\u00B1")), size = 6) +
  annotate('text',x = 1.5,y=1.5,
           label=paste0(paste0('F1 score = ',paste(round(mean(f1)*100,2),round(sd(f1)*100/sqrt(10),2),sep="\u00B1"),'%'),
                        "\n",
                        paste0('Accuracy = ',paste(round(mean(acc)*100,2),round(sd(acc)*100/sqrt(10),2),sep="\u00B1"),'%')),
           size=8,
           fontface =2)+
  labs(title = "Species classification in different datasets",
       x = "Predicted Class",
       y = "True Class") +
  theme_minimal() +
  theme(text = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size = 24),
        axis.title = element_text(family = 'Arial',size = 24, face = "bold"),
        legend.text = element_text(family = 'Arial',size = 24),
        plot.title = element_text(family = 'Arial',size = 24, face = "bold", hjust = 0.5))
print(plot_species)
ggsave('results/confusion_LiverSpecies_species.eps', 
       device = cairo_ps,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
png('results/confusion_LiverSpecies_species.png',units = 'in',width = 9,height = 9,res=600)
print(plot_species)
dev.off()

external_classification <- rbind(external_classification,
                                 data.frame(f1,acc) %>% gather('metric','value') %>% 
                                   mutate(metric=ifelse(metric=='f1','F1','accuracy')) %>%
                                   mutate(datasets = 'combined') %>%
                                   mutate(task = 'species classification') %>%
                                   select('metric','datasets','value','task'))

### Cell classification---------------
ensemble <- NULL
acc <- c()
sens <- c()
spec <- c()
f1 <- c()
for (i in 0:9){
  human_embs <- data.table::fread(paste0('results/embs/liver/LiverEmbs_human_',
                                         i,'.csv')) %>% select(-V1)
  mouse_embs <- data.table::fread(paste0('results/embs/liver/LiverEmbs_mouse_',
                                         i,'.csv')) %>% select(-V1)
  all_embs <- rbind(human_embs,mouse_embs)
  cf_matrix <- confusionMatrix(data=factor(1+all_embs$cell_pred,levels = c(1,2,3,4,5)), 
                               reference = factor(all_embs$cell_true,levels = c(1,2,3,4,5)))
  acc[i+1] <- cf_matrix$overall['Accuracy']
  sens[i+1] <- mean(cf_matrix$byClass[,'Sensitivity'])
  spec[i+1] <- mean(cf_matrix$byClass[,'Specificity'])
  f1[i+1] <-  mean(cf_matrix$byClass[,'F1'])
  ensemble[[i+1]] <-  cf_matrix$table
}
# Calculate mean and standard error of confusion matrix counts
ensemble_mat <- do.call(cbind,ensemble)
ensemble_mat <- array(ensemble_mat,c(dim=dim(ensemble[[1]]),length(ensemble)))
ensemble_mean <- apply(ensemble_mat, c(1,2), mean, na.rm = TRUE)
colnames(ensemble_mean) <- colnames(ensemble[[1]])
rownames(ensemble_mean) <- rownames(ensemble[[1]])
ensemble_sd <- apply(ensemble_mat, c(1,2), sd, na.rm = TRUE)
colnames(ensemble_sd) <- colnames(ensemble[[1]])
rownames(ensemble_sd) <- rownames(ensemble[[1]])
ensemble_mean <- melt(ensemble_mean, variable.name = c("Reference", "Prediction"), value.name = "Count") %>% 
  mutate(Var1 =ifelse(Var1 ==1,'immune',
                   ifelse(Var1 ==2,'mesenchymal',
                          ifelse(Var1 ==3,'epithelial',
                                 ifelse(Var1 ==4,'endothelial','stem cell'))))) %>%
  mutate(Var2=ifelse(Var2==1,'immune',
                     ifelse(Var2==2,'mesenchymal',
                            ifelse(Var2==3,'epithelial',
                                   ifelse(Var2==4,'endothelial','stem cell'))))) %>%
  mutate(Var2 = factor(Var2,levels=c('stem cell','endothelial','epithelial','mesenchymal','immune'))) %>%
  mutate(Var1 = factor(Var1,levels=c('immune','mesenchymal','epithelial','endothelial','stem cell'))) 
ensemble_sd <- melt(ensemble_sd, variable.name = c("Reference", "Prediction"), value.name = "Count_sd") %>% 
  mutate(Var1 =ifelse(Var1 ==1,'immune',
                      ifelse(Var1 ==2,'mesenchymal',
                             ifelse(Var1 ==3,'epithelial',
                                    ifelse(Var1 ==4,'endothelial','stem cell'))))) %>%
  mutate(Var2=ifelse(Var2==1,'immune',
                     ifelse(Var2==2,'mesenchymal',
                            ifelse(Var2==3,'epithelial',
                                   ifelse(Var2==4,'endothelial','stem cell'))))) %>%
  mutate(Var2 = factor(Var2,levels=c('stem cell','endothelial','epithelial','mesenchymal','immune')))%>%
  mutate(Var1 = factor(Var1,levels=c('immune','mesenchymal','epithelial','endothelial','stem cell'))) 
ensemble_final <- left_join(ensemble_mean,ensemble_sd)

plot_cells <- ggplot(ensemble_final, 
                       aes(x = Var1, y = Var2, fill = Count)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, max(ensemble_final$Count))) +
  geom_text(aes(label = paste(round(Count, 3), round(Count_sd/sqrt(10), 2), sep = "\u00B1")), size = 6) +
  annotate('text',x =4,y=5.3,
           label=paste0(paste0('F1 score = ',paste(round(mean(f1)*100,2),round(sd(f1)*100/sqrt(10),2),sep="\u00B1"),'%'),
                        "\n",
                        paste0('Accuracy = ',paste(round(mean(acc)*100,2),round(sd(acc)*100/sqrt(10),2),sep="\u00B1"),'%')),
           size=8,
           fontface =2,)+
  annotate("rect", xmin=2.5, xmax=5.5, ymin=5.1 , ymax=5.5, alpha=0.5, color="black",fill=NA)+
  labs(title = "Cell-type classification in different datasets",
       x = "Predicted Class",
       y = "True Class") +
  theme_minimal() +
  theme(text = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size = 24),
        axis.title = element_text(family = 'Arial',size = 24, face = "bold"),
        legend.text = element_text(family = 'Arial',size = 24),
        plot.title = element_text(family = 'Arial',size = 24, face = "bold", hjust = 0.5))
print(plot_cells)
ggsave('results/confusion_LiverSpecies_cells.eps', 
       device = cairo_ps,
       scale = 1,
       width = 12,
       height = 12,
       units = "in",
       dpi = 600)

png('results/confusion_LiverSpecies_cells.png',units = 'in',width = 12,height = 12,res=600)
print(plot_cells)
dev.off()

external_classification <- rbind(external_classification,
                                 data.frame(f1,acc) %>% gather('metric','value') %>% 
                                   mutate(metric=ifelse(metric=='f1','F1','accuracy')) %>%
                                   mutate(datasets = 'combined') %>%
                                   mutate(task = 'cell-type classification') %>%
                                   select('metric','datasets','value','task'))
external_classification$task <- factor(external_classification$task,
                                       levels = c('disease classification','cell-type classification','species classification'))

# plot barplot
p <- ggboxplot(external_classification,
                x='metric',y='value',color = 'datasets')  + xlab('metric')+ ylab('value')+
  scale_y_continuous(minor_breaks = waiver(),limits = c(0.125,1))+
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black",linewidth=1)+
  facet_wrap(~task,ncol = 3)+
  theme(panel.grid.major = element_line(color = "gray70", size = 0.5, linetype = "dashed"),
        panel.grid.minor =  element_line(color = "gray70", size = 0.5, linetype = "dashed"),
        text = element_text(family = 'Arial',size=24))
print(p)
ggsave('results/10fold_classification_external_datasets.png', 
       scale = 1,
       width = 12,
       height = 6,
       units = "in",
       dpi = 600)
setEPS()
postscript('results/10fold_classification_external_datasets.eps',width = 12,height = 6)
print(p)
dev.off()
