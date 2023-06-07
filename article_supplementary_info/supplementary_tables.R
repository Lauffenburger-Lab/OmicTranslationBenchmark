library(tidyverse)
library(mmtable2)
library(gg3D)
library(ggpubr)
library(gt)
#library(gridExtra)

style_list <- list(cell_borders(sides = "top",color = "black"),
                   cell_borders(sides = "bottom",color = "black"),
                   cell_text(align = "center",size = px(20)))

# Supplementary table 1-4-------------
results_landmarks_translation_a375 <- data.table::fread('../results/MI_results/landmarks_a375_autoencoders_comparison_translation_pvalues_table.csv')
colnames(results_landmarks_translation_a375)[1] <- 'model1'
results_landmarks_translation_a375 <- results_landmarks_translation_a375 %>% mutate(cell='A375')
results_landmarks_translation_ht29 <- data.table::fread('../results/MI_results/landmarks_ht29_autoencoders_comparison_translation_pvalues_table.csv')
colnames(results_landmarks_translation_ht29)[1] <- 'model1'
results_landmarks_translation_ht29 <- results_landmarks_translation_ht29 %>% mutate(cell='HT29')
results_landmarks_translation <- rbind(results_landmarks_translation_a375,results_landmarks_translation_ht29)
results_landmarks_translation <- results_landmarks_translation %>% gather('model2','p-value',-metric,-cell,-model1)
results_landmarks_translation <- results_landmarks_translation %>% mutate(`p-value` = ifelse(`p-value`==0,NA,`p-value`))
results_landmarks_translation <- results_landmarks_translation %>% 
  mutate(`p-value`=ifelse(`p-value`<=1e-03,'\u226410\u207B\u00B3',round(`p-value`,4)))

### Visualize ###
gm_table_1 <- 
  results_landmarks_translation %>% 
  mmtable(cells = `p-value`) +
  header_left(metric) +
  header_top(model2) +
  header_left_top(model1)  +
  header_top_left(cell) + 
  header_format(model2, scope = "table", style = style_list) 
print(gm_table_1)
#gtsave(apply_formats(gm_table_1), filename = "tables/supple.table_1.pdf")


results_allgenes_translation_a375 <- data.table::fread('../results/MI_results/allgenes_a375_autoencoders_comparison_allgenes_translation_pvalues_table.csv')
colnames(results_allgenes_translation_a375)[1] <- 'model1'
results_allgenes_translation_a375 <- results_allgenes_translation_a375 %>% mutate(cell='A375')
results_allgenes_translation_ht29 <- data.table::fread('../results/MI_results/allgenes_ht29_autoencoders_comparison_allgenes_translation_pvalues_table.csv')
colnames(results_allgenes_translation_ht29)[1] <- 'model1'
results_allgenes_translation_ht29 <- results_allgenes_translation_ht29 %>% mutate(cell='HT29')
results_allgenes_translation <- rbind(results_allgenes_translation_a375,results_allgenes_translation_ht29)
results_allgenes_translation <- results_allgenes_translation %>% gather('model2','p-value',-metric,-cell,-model1)
results_allgenes_translation <- results_allgenes_translation %>% mutate(`p-value` = ifelse(`p-value`==0,NA,`p-value`))
results_allgenes_translation <- results_allgenes_translation %>% 
  mutate(`p-value`=ifelse(`p-value`<=1e-03,'\u226410\u207B\u00B3',round(`p-value`,4)))

### Visualize ###
gm_table_3 <- 
  results_allgenes_translation %>% 
  mmtable(cells = `p-value`) +
  header_left(metric) +
  header_top(model2) +
  header_left_top(model1)  +
  header_top_left(cell) + 
  header_format(model2, scope = "table", style = style_list) 
print(gm_table_3)

results_landmarks_recon_a375 <- data.table::fread('../results/MI_results/landmarks_a375_autoencoders_comparison_reconstruction_pvalues_table.csv')
colnames(results_landmarks_recon_a375)[1] <- 'model1'
results_landmarks_recon_a375 <- results_landmarks_recon_a375 %>% mutate(cell='A375')
results_landmarks_recon_ht29 <- data.table::fread('../results/MI_results/landmarks_ht29_autoencoders_comparison_reconstruction_pvalues_table.csv')
colnames(results_landmarks_recon_ht29)[1] <- 'model1'
results_landmarks_recon_ht29 <- results_landmarks_recon_ht29 %>% mutate(cell='HT29')
results_landmarks_reconstruction <- rbind(results_landmarks_recon_a375,results_landmarks_recon_ht29)
results_landmarks_reconstruction <- results_landmarks_reconstruction %>% gather('model2','p-value',-metric,-cell,-model1)
results_landmarks_reconstruction <- results_landmarks_reconstruction %>% mutate(`p-value` = ifelse(`p-value`==0,NA,`p-value`))
results_landmarks_reconstruction <- results_landmarks_reconstruction %>% 
  mutate(`p-value`=ifelse(`p-value`<=1e-03,'\u226410\u207B\u00B3',round(`p-value`,4)))

### Visualize ###
gm_table_2 <- 
  results_landmarks_reconstruction %>% 
  mmtable(cells = `p-value`) +
  header_left(metric) +
  header_top(model2) +
  header_left_top(model1)  +
  header_top_left(cell) + 
  header_format(model2, scope = "table", style = style_list) 
print(gm_table_2)

results_allgenes_recon_a375 <- data.table::fread('../results/MI_results/allgenes_a375_autoencoders_comparison_allgenes_reconstruction_pvalues_table.csv')
colnames(results_allgenes_recon_a375)[1] <- 'model1'
results_allgenes_recon_a375 <- results_allgenes_recon_a375 %>% mutate(cell='A375')
results_allgenes_recon_ht29 <- data.table::fread('../results/MI_results/allgenes_ht29_autoencoders_comparison_allgenes_reconstruction_pvalues_table.csv')
colnames(results_allgenes_recon_ht29)[1] <- 'model1'
results_allgenes_recon_ht29 <- results_allgenes_recon_ht29 %>% mutate(cell='HT29')
results_allgenes_reconstruction <- rbind(results_allgenes_recon_a375,results_allgenes_recon_ht29)
results_allgenes_reconstruction <- results_allgenes_reconstruction %>% gather('model2','p-value',-metric,-cell,-model1)
results_allgenes_reconstruction <- results_allgenes_reconstruction %>% mutate(`p-value` = ifelse(`p-value`==0,NA,`p-value`))
results_allgenes_reconstruction <- results_allgenes_reconstruction %>% 
  mutate(`p-value`=ifelse(`p-value`<=1e-03,'\u226410\u207B\u00B3',round(`p-value`,4)))

### Visualize ###
gm_table_4 <- 
  results_allgenes_reconstruction %>% 
  mmtable(cells = `p-value`) +
  header_left(metric) +
  header_top(model2) +
  header_left_top(model1)  +
  header_top_left(cell) + 
  header_format(model2, scope = "table", style = style_list) 
print(gm_table_4)
