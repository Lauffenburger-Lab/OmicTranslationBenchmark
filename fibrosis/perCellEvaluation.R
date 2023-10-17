library(tidyverse)
library(reshape2)
library(ggsignif)
library(ggpubr)
library(scales)
library(patchwork)
library(gridExtra)
library(doFuture)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)

r2 <- function (x, y) cor(x, y)^2

scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}

homologues_map = data.table::fread('results/HumanMouseHomologuesMap.csv')

### Reconstruction analysis--------------------
all_species <- c('human','mouse')
models <- c('','homologues_','DCS_','TransCompR_')
results_reconstruction <- data.frame()
for (species in all_species){
  for (model in models){
    if (model==''){
      message(paste0('Started model : ','CPA',' in ',species))
    }else{
      message(paste0('Started model : ',substr(model,start = 1,stop=nchar(model)-1),' in ',species))
    }
    for (i in 0:9){
      #Load data
      val <- data.table::fread(paste0('data/10foldcrossval_lung/csvFiles/labeled_val_',species,'_',i,'.csv'))
      val <- val %>% select(-cell_type)
      
      pred <- data.table::fread(paste0('results/preds/validation/valPreds_',model,'reconstructed_',i,'_',species,'.csv'))
      pred <- pred %>% select(-V1,-cell_type)
      
      if (model!=''){
        if (species=='mouse'){
          val <- val %>% select(c(all_of(homologues_map$mouse_gene),"diagnosis","specific_cell",))
        }else{
          val <- val %>% select(c(all_of(homologues_map$human_gene),"diagnosis","specific_cell",))
        }
      }
      
      num_genes <- ncol(pred)-1
      pred$specific_cell <- val$specific_cell
      
      # keep cell_counts of specific cells
      cells <- data.table::fread(paste0('data/',species,'_cells_info.csv'))
      
      #calculate gene_means and gene_vars
      val_means <- aggregate(val[,1:num_genes],by=list(val$specific_cell),FUN=mean)
      val_variances <-  aggregate(val[,1:num_genes],by=list(val$specific_cell),FUN=var)
      
      # Predictions performance
      pred_means <- aggregate(pred[,1:num_genes],by=list(pred$specific_cell),FUN=mean)
      pred_variances <-  aggregate(pred[,1:num_genes],by=list(pred$specific_cell),FUN=var)
      
      # Calculate the per cell type performance
      # Calculate performance in predicting per gene mean
      means_r <- foreach(cell = val_means$Group.1) %dopar% {
        ytrue <- val_means %>% filter(Group.1==cell)
        ytrue <- ytrue[1,2:num_genes]
        ytrue <- as.matrix(ytrue)[1,]
        yhat <- pred_means %>% filter(Group.1==cell)
        yhat <- yhat[1,2:num_genes]
        yhat <- as.matrix(yhat)[1,]
        cor(ytrue ,
           yhat)
      }
      names(means_r) <- val_means$Group.1
      means_r <- do.call(rbind,means_r)
      colnames(means_r) <- 'r'
      means_r <- as.data.frame(means_r) %>% rownames_to_column('specific_cell')
      means_r <- suppressMessages(left_join(means_r,cells))
      
      # Calculate performance in predicting gene variance
      vars_r <- foreach(cell = val_variances$Group.1) %dopar% {
        ytrue <- val_variances %>% filter(Group.1==cell)
        ytrue <- ytrue[1,2:num_genes]
        ytrue <- as.matrix(ytrue)[1,]
        yhat <- pred_means %>% filter(Group.1==cell)
        yhat <- yhat[1,2:num_genes]
        yhat <- as.matrix(yhat)[1,]
        cor(ytrue ,
           yhat)
      }
      names(vars_r) <- val_variances$Group.1
      vars_r <- do.call(rbind,vars_r)
      colnames(vars_r) <- 'r'
      vars_r <- as.data.frame(vars_r) %>% rownames_to_column('specific_cell')
      vars_r <- suppressMessages(left_join(vars_r,cells))
      
      res_r <- rbind(means_r %>% mutate(predicting = 'mean'),
                      vars_r %>% mutate(predicting= 'variance'))
      if (model==''){
        res_r <- res_r %>% mutate(model = 'CPA')
      }else{
        res_r <- res_r %>% mutate(model = substr(model,start = 1,stop=nchar(model)-1))
      }
      res_r <- res_r %>% mutate(species = species)
      res_r <- res_r %>% mutate(fold = i)
      results_reconstruction <- rbind(results_reconstruction,res_r)
      message(paste0('Finished fold ',i))
    }
    if (model==''){
      message(paste0('Finished model : ','CPA',' in ',species))
    }else{
      message(paste0('Finished model : ',substr(model,start = 1,stop=nchar(model)-1),' in ',species))
    }
  }
}
# saveRDS(results_reconstruction,'results/results_reconstruction.rds')

results_reconstruction <- results_reconstruction %>% group_by(predicting,species,model,specific_cell) %>%
  mutate(mean_r = mean(r,na.rm = T)) %>% mutate(sd_r = sd(r,na.rm = T)) %>% ungroup()

p_scatter_1 <- ggscatter(results_reconstruction %>% filter(model=='CPA'),
                         x='cell_counts',y='mean_r',
          add = 'none',cor.coef = T,
          cor.coef.size = 6,add.params = list(color='blue'))+
  geom_errorbar(aes(ymin = mean_r - sd_r/sqrt(10),ymax=mean_r + sd_r/sqrt(10)),
                width=0.05)+
  facet_wrap(vars(species,predicting)) +
  scale_x_log10(labels = label_log(digits = 2),expand = c(0.1, 0.1))+
  geom_smooth(aes(x=cell_counts,y=r),method = 'lm')+
  xlab('# cells in the data')+
  # ylab(expression(R^2))+
  ylab('pearson`s r')+
  # ylim(c(0,1))+
  geom_hline(yintercept = 1,color='black',linetype='dashed')+
  geom_hline(yintercept = 0,color='black',linetype='dashed')+
  ggtitle('Reconstruction performance VS number of cells')+
  theme(text=element_text(family = 'Arial',size=20),
        plot.title = element_text(hjust = 0.5))
print(p_scatter_1)
ggsave('../figures/cellFrequency_vs_r_reconstruction.eps',
       plot = p_scatter_1,
       device = cairo_ps,
       height = 12,
       width = 12,
       units = 'in',
       dpi = 600)


results4Boxplot <- results_reconstruction %>% filter(model=='CPA')#%>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
# Do for human CPA
data <- results4Boxplot %>% filter(species=='human') %>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
data <- data %>% group_by(predicting,specific_cell) %>% mutate(mean_r = median(r,na.rm = T)) %>% ungroup()%>% arrange(desc(mean_r))
data_mean <- data %>% filter(predicting == 'Predicting per cell type mean') 
data_var <- data %>% filter(predicting == 'Predicting per cell type variance')
data_mean_tmp <- data_mean %>% select(specific_cell,mean_r) %>% unique()
data_mean_tmp <- data_mean_tmp %>% arrange(mean_r)
data_var_tmp <- data_var %>% select(specific_cell,mean_r) %>% unique()
data_var_tmp <- data_var_tmp %>% arrange(mean_r)
mean_levels <-  data_mean_tmp$specific_cell
var_levels <- data_var_tmp$specific_cell
data_mean$specific_cell <- factor(data_mean$specific_cell,
                                   levels = mean_levels)
data_var$specific_cell <- factor(data_var$specific_cell,
                                   levels = mean_levels)
p_human <- ggboxplot(rbind(data_mean,data_var),x='specific_cell',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(species,predicting))+
  xlab('cell type')+
  #ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  ggtitle('Reconstruction performance')+
  theme(text=element_text(family = 'Arial',size=20),
        axis.text.y = element_text(size=16),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
# print(p_human)
# Do for mouse CPA
data <- results4Boxplot %>% filter(species=='mouse') %>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
data <- data %>% group_by(predicting,specific_cell) %>% mutate(mean_r = median(r,na.rm = T)) %>% ungroup()%>% arrange(desc(mean_r))
data_mean <- data %>% filter(predicting == 'Predicting per cell type mean') 
data_var <- data %>% filter(predicting == 'Predicting per cell type variance')
data_mean_tmp <- data_mean %>% select(specific_cell,mean_r) %>% unique()
data_mean_tmp <- data_mean_tmp %>% arrange(mean_r)
data_var_tmp <- data_var %>% select(specific_cell,mean_r) %>% unique()
data_var_tmp <- data_var_tmp %>% arrange(mean_r)
mean_levels <-  data_mean_tmp$specific_cell
var_levels <- data_var_tmp$specific_cell
data_mean$specific_cell <- factor(data_mean$specific_cell,
                                  levels = mean_levels)
data_var$specific_cell <- factor(data_var$specific_cell,
                                 levels = mean_levels)
p_mouse <- ggboxplot(rbind(data_mean,data_var),x='specific_cell',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(species,predicting))+
  xlab('cell type')+
  # ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  ggtitle('')+
  theme(text=element_text(family = 'Arial',size=20),
        axis.text.y = element_text(size=16),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
# print(p_mouse)
# p <- p_human / p_mouse
p <- wrap_plots(list(p_human,p_mouse),nrow = 2)
print(p)
ggsave('../figures/cpa_fibrosis_reconstruction_perCelltype.eps',
       plot = p,
       device = cairo_ps,
       width = 16,
       height = 16,
       units = 'in',
       dpi=600)

### Compare the different approaches per cell type
results_reconstruction <- results_reconstruction %>% group_by(predicting,species,model,fold) %>%
  mutate(averageCellr = mean(r,na.rm = T)) %>% mutate(sdCellr = sd(r,na.rm = T)) %>% ungroup()

results <- results_reconstruction %>% mutate(model = ifelse(model=='CPA','CPA-based all genes',
                                                            ifelse(model=='homologues','CPA-based homologues',
                                                                   ifelse(model=='DCS','DCS modified v2',
                                                                          'TransCompR'))))
results <- results %>% mutate(predicting = paste0('Predicting per cell type ',predicting))
my_comparisons <- list( c('CPA-based all genes', 'CPA-based homologues'),
                        c('DCS modified v2', 'CPA-based homologues'),
                        c('CPA-based all genes', 'DCS modified v2'),
                        c('CPA-based all genes','TransCompR'))

p1 <- ggboxplot(results %>% select(model,averageCellr,fold,predicting,species) %>% unique(),
                x='model',y='averageCellr',color = 'model',
                add = 'jitter')  + 
  xlab('')+ 
  # ylab(expression('average R'^2))+
  ylab('pearson`s r')+
  ggtitle("Average performance across all cell types")+
  scale_y_continuous(breaks = seq(0.3,1,0.1),minor_breaks = waiver(),limits = c(0.3,1))+
  # geom_hline(yintercept = 0,lty='dashed',linewidth=1)+
  facet_wrap(vars(species,predicting)) +
  stat_compare_means(label = 'p.signif',
                     method = 'wilcox.test',
                     #step.increase = 0.02,
                     label.y = c(0.75,0.8,0.85,0.9),
                     tip.length=0.05,
                     size =6,
                     aes(group=model),
                     comparisons = my_comparisons) +
  theme(text = element_text(family = 'Arial',size=23),
        plot.title = element_text(hjust = 0.5,size=23),
        legend.title = element_blank(),
        axis.text.x = element_blank())
print(p1)
ggsave('../figures/fibrosis_performance_reconstruction_comparison_average_percelltype.eps', 
       plot=p1,
       device = cairo_ps,
       scale = 1,
       width = 12,
       height = 12,
       units = "in",
       dpi = 600)

#### Translation performance-----------------------------------
# First find equivalent cells to translate

### TIPS ###
### Perform analysis: 
### 1) per general cell_type
### 2) keep relevent genes and do global (perhaps not)
### 3) hand-pick good performing cells and important cells for the problem and map them to use those

cells_mouse <- data.table::fread('data/mouse_cells_info.csv')
colnames(cells_mouse) <- c('mouse cell','cell_type','mouse_cell_counts')
cells_human <- data.table::fread('data/human_cells_info.csv')
colnames(cells_human) <- c('human cell','cell_type','human_cell_counts')
cells_all <- left_join(cells_human,cells_mouse)
cells_all$keep <- FALSE
for (i in 1:nrow(cells_all)){
  if(grepl(tolower(cells_all$`mouse cell`[i]),tolower(cells_all$`human cell`[i])) | grepl(tolower(cells_all$`human cell`[i]),tolower(cells_all$`mouse cell`[i]))){
    cells_all$keep[i] <- TRUE
  }
  if(tolower(cells_all$`human cell`[i])==tolower(cells_all$`mouse cell`[i])){
    cells_all$keep[i] <- TRUE
  }
  if (grepl('lymphocytes',tolower(cells_all$`human cell`[i])) | grepl('lymphocytes',tolower(cells_all$`mouse cell`[i]))){
    cells_all$keep[i] <- TRUE
  }
  if (grepl('t cells',tolower(cells_all$`human cell`[i])) | grepl('t cells',tolower(cells_all$`mouse cell`[i]))){
    cells_all$keep[i] <- TRUE
  }
  if (grepl('b cells',tolower(cells_all$`human cell`[i])) | grepl('b cells',tolower(cells_all$`mouse cell`[i]))){
    cells_all$keep[i] <- TRUE
  }
  if (grepl('macrophages',tolower(cells_all$`human cell`[i])) | grepl('macrophages',tolower(cells_all$`mouse cell`[i]))){
    cells_all$keep[i] <- TRUE
  }
}
cells_all <- cells_all %>% filter(keep==TRUE)
all_species <- c('human','mouse')
all_trans <- c('tohuman','tomouse')
models <- c('','homologues_','DCS_','TransCompR_')
results_translation <- data.frame()
for (trans in all_trans){
  if (grepl('human',trans)){
    species <- 'human'
  }else{
    species <- 'mouse'
  }
  for (model in models){
    if (model==''){
      message(paste0('Started model : ','CPA translate',' into ',species))
    }else{
      message(paste0('Started model : ',substr(model,start = 1,stop=nchar(model)-1),' translate into ',species))
    }
    for (i in 0:9){
      #Load data
      val <- data.table::fread(paste0('data/10foldcrossval_lung/csvFiles/labeled_val_',species,'_',i,'.csv'))
      val <- val %>% select(-cell_type)
      
      ind <- which(all_species!=species)
      origin_species <- all_species[ind]
      val_origin <- data.table::fread(paste0('data/10foldcrossval_lung/csvFiles/labeled_val_',origin_species,'_',i,'.csv'))
      
      pred <- data.table::fread(paste0('results/preds/validation/valPreds_',model,'translated_',i,'_',trans,'.csv'))
      pred$specific_cell <- val_origin$specific_cell
      pred <- pred %>% select(-V1,-mouse_cell_type,-mouse_diagnosis)
      val_origin <- val_origin %>% select(-cell_type)
      
      if (model!=''){
        if (trans=='tomouse'){
          val <- val %>% select(c(all_of(homologues_map$mouse_gene),"diagnosis","specific_cell",))
        }else{
          val <- val %>% select(c(all_of(homologues_map$human_gene),"diagnosis","specific_cell",))
        }
      }
      
      num_genes <- ncol(pred)-1
      pred$specific_cell <- val$specific_cell
      
      # keep cell_counts of specific cells
      cells <- data.table::fread(paste0('data/',species,'_cells_info.csv'))
      cells_origin <- data.table::fread(paste0('data/',species,'_cells_info.csv'))
      if (species=='mouse'){
        cells <- cells %>% filter(specific_cell %in% cells_all$`mouse cell`)
        cells_origin <- cells_origin  %>% filter(specific_cell %in% cells_all$`human cell`)
      }else{
        cells <- cells %>% filter(specific_cell %in% cells_all$`human cell`)
        cells_origin <- cells_origin %>% filter(specific_cell %in% cells_all$`mouse cell`)
      }
      
      ## keep relevant cells ################ EDO EIXA MEINEI ###############################
      val <- val %>% filter(specific_cell %in% cells$specific_cell)
      pred <- pred %>% filter(specific_cell %in% cells_origin$specific_cell)
      val_origin <- pred %>% filter(specific_cell %in% cells_origin$specific_cell)
      
      #calculate gene_means and gene_vars
      val_means <- aggregate(val[,1:num_genes],by=list(val$specific_cell),FUN=mean)
      val_variances <-  aggregate(val[,1:num_genes],by=list(val$specific_cell),FUN=var)
      
      # Predictions performance
      pred_means <- aggregate(pred[,1:num_genes],by=list(pred$specific_cell),FUN=mean)
      pred_variances <-  aggregate(pred[,1:num_genes],by=list(pred$specific_cell),FUN=var)
      
      # Calculate the per cell type performance
      # Calculate performance in predicting per gene mean
      means_r <- foreach(cell = val_means$Group.1) %dopar% {
        ytrue <- val_means %>% filter(Group.1==cell)
        ytrue <- ytrue[1,2:num_genes]
        ytrue <- as.matrix(ytrue)[1,]
        yhat <- pred_means %>% filter(Group.1==cell)
        yhat <- yhat[1,2:num_genes]
        yhat <- as.matrix(yhat)[1,]
        cor(ytrue ,
            yhat)
      }
      names(means_r) <- val_means$Group.1
      means_r <- do.call(rbind,means_r)
      colnames(means_r) <- 'r'
      means_r <- as.data.frame(means_r) %>% rownames_to_column('specific_cell')
      means_r <- suppressMessages(left_join(means_r,cells))
      
      # Calculate performance in predicting gene variance
      vars_r <- foreach(cell = val_variances$Group.1) %dopar% {
        ytrue <- val_variances %>% filter(Group.1==cell)
        ytrue <- ytrue[1,2:num_genes]
        ytrue <- as.matrix(ytrue)[1,]
        yhat <- pred_means %>% filter(Group.1==cell)
        yhat <- yhat[1,2:num_genes]
        yhat <- as.matrix(yhat)[1,]
        cor(ytrue ,
            yhat)
      }
      names(vars_r) <- val_variances$Group.1
      vars_r <- do.call(rbind,vars_r)
      colnames(vars_r) <- 'r'
      vars_r <- as.data.frame(vars_r) %>% rownames_to_column('specific_cell')
      vars_r <- suppressMessages(left_join(vars_r,cells))
      
      res_r <- rbind(means_r %>% mutate(predicting = 'mean'),
                     vars_r %>% mutate(predicting= 'variance'))
      if (model==''){
        res_r <- res_r %>% mutate(model = 'CPA')
      }else{
        res_r <- res_r %>% mutate(model = substr(model,start = 1,stop=nchar(model)-1))
      }
      res_r <- res_r %>% mutate(translation = trans)
      res_r <- res_r %>% mutate(fold = i)
      results_reconstruction <- rbind(results_reconstruction,res_r)
      message(paste0('Finished fold ',i))
    }
    if (model==''){
      message(paste0('Finished model : ','CPA translate',' into ',species))
    }else{
      message(paste0('Finished model : ',substr(model,start = 1,stop=nchar(model)-1),' translate into ',species))
    }
  }
}
# saveRDS(results_reconstruction,'results/results_reconstruction.rds')

results_reconstruction <- results_reconstruction %>% group_by(predicting,translation,model,specific_cell) %>%
  mutate(mean_r = mean(r,na.rm = T)) %>% mutate(sd_r = sd(r,na.rm = T)) %>% ungroup()

p_scatter_1 <- ggscatter(results_reconstruction %>% filter(model=='CPA'),
                         x='cell_counts',y='mean_r',
                         add = 'none',cor.coef = T,
                         cor.coef.size = 6,add.params = list(color='blue'))+
  geom_errorbar(aes(ymin = mean_r - sd_r/sqrt(10),ymax=mean_r + sd_r/sqrt(10)),
                width=0.05)+
  facet_wrap(vars(translation,predicting)) +
  scale_x_log10(labels = label_log(digits = 2),expand = c(0.1, 0.1))+
  geom_smooth(aes(x=cell_counts,y=r),method = 'lm')+
  xlab('# cells in the data')+
  # ylab(expression(R^2))+
  ylab('pearson`s r')+
  # ylim(c(0,1))+
  geom_hline(yintercept = 1,color='black',linetype='dashed')+
  geom_hline(yintercept = 0,color='black',linetype='dashed')+
  ggtitle('Reconstruction performance VS number of cells')+
  theme(text=element_text(family = 'Arial',size=20),
        plot.title = element_text(hjust = 0.5))
print(p_scatter_1)
ggsave('../figures/cellFrequency_vs_r_reconstruction.eps',
       plot = p_scatter_1,
       device = cairo_ps,
       height = 12,
       width = 12,
       units = 'in',
       dpi = 600)


results4Boxplot <- results_reconstruction %>% filter(model=='CPA')#%>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
# Do for human CPA
data <- results4Boxplot %>% filter(translation=='tohuman') %>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
data <- data %>% group_by(predicting,specific_cell) %>% mutate(mean_r = median(r,na.rm = T)) %>% ungroup()%>% arrange(desc(mean_r))
data_mean <- data %>% filter(predicting == 'Predicting per cell type mean') 
data_var <- data %>% filter(predicting == 'Predicting per cell type variance')
data_mean_tmp <- data_mean %>% select(specific_cell,mean_r) %>% unique()
data_mean_tmp <- data_mean_tmp %>% arrange(mean_r)
data_var_tmp <- data_var %>% select(specific_cell,mean_r) %>% unique()
data_var_tmp <- data_var_tmp %>% arrange(mean_r)
mean_levels <-  data_mean_tmp$specific_cell
var_levels <- data_var_tmp$specific_cell
data_mean$specific_cell <- factor(data_mean$specific_cell,
                                  levels = mean_levels)
data_var$specific_cell <- factor(data_var$specific_cell,
                                 levels = mean_levels)
p_human <- ggboxplot(rbind(data_mean,data_var),x='specific_cell',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(translation,predicting))+
  xlab('cell type')+
  #ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  ggtitle('Reconstruction performance')+
  theme(text=element_text(family = 'Arial',size=20),
        axis.text.y = element_text(size=16),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
# print(p_human)
# Do for mouse CPA
data <- results4Boxplot %>% filter(translation=='tomouse') %>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
data <- data %>% group_by(predicting,specific_cell) %>% mutate(mean_r = median(r,na.rm = T)) %>% ungroup()%>% arrange(desc(mean_r))
data_mean <- data %>% filter(predicting == 'Predicting per cell type mean') 
data_var <- data %>% filter(predicting == 'Predicting per cell type variance')
data_mean_tmp <- data_mean %>% select(specific_cell,mean_r) %>% unique()
data_mean_tmp <- data_mean_tmp %>% arrange(mean_r)
data_var_tmp <- data_var %>% select(specific_cell,mean_r) %>% unique()
data_var_tmp <- data_var_tmp %>% arrange(mean_r)
mean_levels <-  data_mean_tmp$specific_cell
var_levels <- data_var_tmp$specific_cell
data_mean$specific_cell <- factor(data_mean$specific_cell,
                                  levels = mean_levels)
data_var$specific_cell <- factor(data_var$specific_cell,
                                 levels = mean_levels)
p_mouse <- ggboxplot(rbind(data_mean,data_var),x='specific_cell',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(translation,predicting))+
  xlab('cell type')+
  # ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  ggtitle('')+
  theme(text=element_text(family = 'Arial',size=20),
        axis.text.y = element_text(size=16),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
# print(p_mouse)
# p <- p_human / p_mouse
p <- wrap_plots(list(p_human,p_mouse),nrow = 2)
print(p)
ggsave('../figures/cpa_fibrosis_reconstruction_perCelltype.eps',
       plot = p,
       device = cairo_ps,
       width = 16,
       height = 16,
       units = 'in',
       dpi=600)

### Compare the different approaches per cell type
results_reconstruction <- results_reconstruction %>% group_by(predicting,translation,model,fold) %>%
  mutate(averageCellr = mean(r,na.rm = T)) %>% mutate(sdCellr = sd(r,na.rm = T)) %>% ungroup()

results <- results_reconstruction %>% mutate(model = ifelse(model=='CPA','CPA-based all genes',
                                                            ifelse(model=='homologues','CPA-based homologues',
                                                                   ifelse(model=='DCS','DCS modified v2',
                                                                          'TransCompR'))))
results <- results %>% mutate(predicting = paste0('Predicting per cell type ',predicting))
my_comparisons <- list( c('CPA-based all genes', 'CPA-based homologues'),
                        c('DCS modified v2', 'CPA-based homologues'),
                        c('CPA-based all genes', 'DCS modified v2'),
                        c('CPA-based all genes','TransCompR'))

p1 <- ggboxplot(results %>% select(model,averageCellr,fold,predicting,translation) %>% unique(),
                x='model',y='averageCellr',color = 'model',
                add = 'jitter')  + 
  xlab('')+ 
  # ylab(expression('average R'^2))+
  ylab('pearson`s r')+
  ggtitle("Average performance across all cell types")+
  scale_y_continuous(breaks = seq(0.3,1,0.1),minor_breaks = waiver(),limits = c(0.3,1))+
  # geom_hline(yintercept = 0,lty='dashed',linewidth=1)+
  facet_wrap(vars(translation,predicting)) +
  stat_compare_means(label = 'p.signif',
                     method = 'wilcox.test',
                     #step.increase = 0.02,
                     label.y = c(0.75,0.8,0.85,0.9),
                     tip.length=0.05,
                     size =6,
                     aes(group=model),
                     comparisons = my_comparisons) +
  theme(text = element_text(family = 'Arial',size=23),
        plot.title = element_text(hjust = 0.5,size=23),
        legend.title = element_blank(),
        axis.text.x = element_blank())
print(p1)
ggsave('../figures/fibrosis_performance_reconstruction_comparison_average_percelltype.eps', 
       plot=p1,
       device = cairo_ps,
       scale = 1,
       width = 12,
       height = 12,
       units = "in",
       dpi = 600)
