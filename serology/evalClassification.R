library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpubr)

# Load results
results <- data.table::fread('results_intermediate_encoders/10foldvalidation_wholeModel_32dim2000ep_serology.csv')
results_f1 <- results %>% select(F1Species,F1_global,F1Protection,KNNTranslationF1,ClassifierTranslationF1)
results_f1 <- results_f1 %>% gather('task','F1') %>% 
  mutate(task = ifelse(grepl('Species',task),'species classification',
                       ifelse(grepl('Protection',task),'protection classification',
                              ifelse(grepl('global',task),'vaccination classification',
                                     ifelse(grepl('KNN',task),'KNN species translation',
                                                  'classifier species translation')))))
results_f1$task <- factor(results_f1$task,
                           levels = c('protection classification',
                                      'vaccination classification',
                                      'species classification',
                                      'classifier species translation',
                                      'KNN species translation'))
results_acc <- results %>% select(AccuracySpecies,Accuracy_global,AccProtection,KNNTranslationAcc,ClassifierTranslationAcc)
results_acc <- results_acc %>% gather('task','Accuracy') %>% 
  mutate(task = ifelse(grepl('Species',task),'species classification',
                       ifelse(grepl('Protection',task),'protection classification',
                              ifelse(grepl('global',task),'vaccination classification',
                                     ifelse(grepl('KNN',task),'KNN species translation',
                                            'classifier species translation')))))
results_acc$task <- factor(results_acc$task,
                           levels = c('protection classification',
                                      'vaccination classification',
                                      'species classification',
                                      'classifier species translation',
                                      'KNN species translation'))

# Plot results
p1 <- ggboxplot(results_f1,x='task',y='F1',color = 'task') +
  scale_y_continuous(breaks = seq(0.5,1,0.1),minor_breaks = waiver(),limits = c(0.5,1))+
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black",linewidth=1)+
  theme(panel.grid.major = element_line(color = "gray70", linewidth = 0.5, linetype = "dashed"),
        panel.grid.minor =  element_line(color = "gray70", linewidth = 0.5, linetype = "dashed"),
        axis.text.x = element_blank(),
        legend.text = element_text(family = 'Arial',size=20),
        text = element_text(family = 'Arial',size=24))
print(p1)

p2 <- ggboxplot(results_acc,x='task',y='Accuracy',color = 'task') +
  scale_y_continuous(breaks = seq(0.5,1,0.1),minor_breaks = waiver(),limits = c(0.5,1))+
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black",linewidth=1)+
  theme(panel.grid.major = element_line(color = "gray70", linewidth = 0.5, linetype = "dashed"),
        panel.grid.minor =  element_line(color = "gray70", linewidth = 0.5, linetype = "dashed"),
        axis.text.x = element_blank(),
        legend.text = element_text(family = 'Arial',size=20),
        text = element_text(family = 'Arial',size=24))
print(p2)

# combine into one plot
p <- ggarrange(plotlist=list(p1,p2),ncol=2,nrow=1,common.legend = TRUE,legend = 'top')
annotate_figure(p, top = text_grob("Classification performance", 
                                   color = "black",face = 'plain', size = 24))
ggsave(
  'results_intermediate_encoders/classification_performance.eps', 
  device = cairo_ps,
  scale = 1,
  width = 22,
  height = 14,
  units = "in",
  dpi = 600,
)
