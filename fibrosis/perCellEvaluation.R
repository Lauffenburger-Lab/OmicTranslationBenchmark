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
data <- data %>% group_by(predicting,species,specific_cell) %>% mutate(mean_r = median(r,na.rm = T)) %>% ungroup()%>% arrange(desc(mean_r))
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
p_human2 <- ggboxplot(rbind(data_mean,data_var)%>% filter(predicting=="Predicting per cell type mean"),
                     x='specific_cell',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(species,predicting))+
  xlab('cell type')+
  #ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  #ggtitle('Reconstruction performance')+
  theme(text=element_text(family = 'Arial',size=20),
        axis.text.y = element_text(size=18),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
# print(p_human)
# Do for mouse CPA
data <- results4Boxplot %>% filter(species=='mouse') %>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
data <- data %>% group_by(predicting,species,specific_cell) %>% mutate(mean_r = median(r,na.rm = T)) %>% ungroup()%>% arrange(desc(mean_r))
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
p_mouse2 <- ggboxplot(rbind(data_mean,data_var) %>% filter(predicting=="Predicting per cell type mean"),
                     x='specific_cell',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(species,predicting))+
  xlab('cell type')+
  # ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  ggtitle('')+
  theme(text=element_text(family = 'Arial',size=20),
        axis.text.y = element_text(size=18),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
# print(p_mouse)
# p <- p_human / p_mouse
p <- wrap_plots(list(p_human,p_mouse),nrow = 2)
print(p)
ggsave('../figures/cpa_fibrosis_reconstruction_perCelltype.png',
       plot = p,
       width = 16,
       height = 16,
       units = 'in',
       dpi=600)

p <- ggarrange(plotlist = list(p_human2,p_mouse2),common.legend = T,legend = 'right')
print(p)
# ggsave('../figures/cpa_fibrosis_reconstruction_perCelltype.pdf',
#        plot = p,
#        device = cairo_pdf,
#        width = 18,
#        height = 18,
#        units = 'in',
#        dpi=600)
postscript('../figures/cpa_fibrosis_reconstruction_perCelltype.eps',width = 18,height = 18)
print(p)
dev.off()
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
p2 <- ggboxplot(results %>% select(model,r,specific_cell,fold,predicting,species) %>% unique(),
                x='model',y='r',color = 'model',
                add = 'jitter')  + 
  xlab('')+ 
  # ylab(expression('average R'^2))+
  ylab('pearson`s r')+
  ggtitle("Performance across all cell types")+
  scale_y_continuous(breaks = seq(0,1,0.2),minor_breaks = waiver(),limits = c(-0.001,1.35))+
  # geom_hline(yintercept = 0,lty='dashed',linewidth=1)+
  facet_wrap(vars(species,predicting)) +
  stat_compare_means(label = 'p.signif',
                     method = 'wilcox.test',
                     #step.increase = 0.02,
                     label.y = c(0.95,1.05,1.15,1.25),
                     tip.length=0.05,
                     size =6,
                     aes(group=model),
                     comparisons = my_comparisons) +
  theme(text = element_text(family = 'Arial',size=23),
        plot.title = element_text(hjust = 0.5,size=23),
        legend.title = element_blank(),
        axis.text.x = element_blank())
print(p2)
ggsave('../figures/fibrosis_performance_reconstruction_comparison_allgrouped.png', 
       plot=p2,
       scale = 1,
       width = 12,
       height = 12,
       units = "in",
       dpi = 600)
postscript('../figures/fibrosis_performance_reconstruction_comparison_allgrouped.eps',width = 16,height = 16)
print(p2)
dev.off()

#### Translation performance for specific cells-----------------------------------
# First find equivalent cells to translate
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
cells_human <- cells_human %>% filter(`human cell` %in% cells_all$`human cell`)
cells_mouse <- cells_mouse %>% filter(`mouse cell` %in% cells_all$`mouse cell`)

cells_human <- cells_human %>% mutate(specific_cell_mapped = toupper(`human cell`))
for (i in 1:nrow(cells_human)){
  if (grepl('t cells',tolower(cells_human$specific_cell_mapped[i]))){
    cells_human$specific_cell_mapped[i] <- 'T CELLs'
  }
  if (grepl('b cells',tolower(cells_human$specific_cell_mapped[i]))){
    cells_human$specific_cell_mapped[i] <- 'B CELLs'
  }
  if (grepl('macrophages',tolower(cells_human$specific_cell_mapped[i]))){
    cells_human$specific_cell_mapped[i] <- 'MACROPHAGES'
  }
}
cells_mouse <- cells_mouse %>% mutate(specific_cell_mapped = toupper(`mouse cell`))
for (i in 1:nrow(cells_mouse)){
  if (grepl('t-lymphocytes',tolower(cells_mouse$specific_cell_mapped[i]))){
    cells_mouse$specific_cell_mapped[i] <- 'T CELLs'
  }
  if (grepl('b-lymphocytes',tolower(cells_mouse$specific_cell_mapped[i]))){
    cells_mouse$specific_cell_mapped[i] <- 'B CELLs'
  }
  if (grepl('macrophages',tolower(cells_mouse$specific_cell_mapped[i]))){
    cells_mouse$specific_cell_mapped[i] <- 'MACROPHAGES'
  }
  if (grepl('at2',tolower(cells_mouse$specific_cell_mapped[i]))){
    cells_mouse$specific_cell_mapped[i] <- 'AT2'
  }
  if (grepl('at1',tolower(cells_mouse$specific_cell_mapped[i]))){
    cells_mouse$specific_cell_mapped[i] <- 'AT1'
  }
}
colnames(cells_human)[2] <- 'cell_type_human'
colnames(cells_mouse)[2] <- 'cell_type_mouse'
cells_all <- left_join(cells_human,cells_mouse)
cells_all <- cells_all %>% filter(!is.na(`mouse cell`))


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
      
      # keep cell_counts of specific cells
      cells <- cells_all
      cells_origin <- cells_all
      if (species=='mouse'){
        cells <- cells %>% select(c('specific_cell'='mouse cell'),c('cell_counts'='mouse_cell_counts'),specific_cell_mapped) %>% unique()
        cells_origin <- cells_origin  %>% select(c('specific_cell'='human cell'),c('cell_counts'='human_cell_counts'),specific_cell_mapped) %>% unique()
      }else{
        cells <- cells %>% select(c('specific_cell'='human cell'),c('cell_counts'='human_cell_counts'),specific_cell_mapped) %>% unique()
        cells_origin <- cells_origin  %>% select(c('specific_cell'='mouse cell'),c('cell_counts'='mouse_cell_counts'),specific_cell_mapped) %>% unique()
      }
      ## keep relevant cells
      val_origin <- val_origin %>% filter(specific_cell %in% cells_origin$specific_cell)
      pred <- pred %>% filter(specific_cell %in% cells_origin$specific_cell)
      pred <- suppressMessages(left_join(pred,cells_origin) %>% filter(!is.na(specific_cell_mapped)))
      if (origin_species=='mouse'){
        pred <- pred %>% select(-V1,-mouse_cell_type,-mouse_diagnosis,-cell_counts,-specific_cell)
      }else{
        pred <- pred %>% select(-V1,-human_cell_type,-human_diagnosis,-cell_counts,-specific_cell)
      }
      pred <- distinct(pred)
      val <- val %>% filter(specific_cell %in% cells$specific_cell)
      val <- suppressMessages(left_join(val,cells) %>% filter(!is.na(specific_cell_mapped)))
      val <- val %>% select(-diagnosis,-cell_counts,-specific_cell)
      val <- distinct(val)
      
      if (model!=''){
        if (trans=='tomouse'){
          val <- val %>% select(c(all_of(homologues_map$mouse_gene),"specific_cell_mapped",))
        }else{
          val <- val %>% select(c(all_of(homologues_map$human_gene),"specific_cell_mapped",))
        }
      }
      num_genes <- ncol(pred)-1
      
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
      means_r <- as.data.frame(means_r) %>% rownames_to_column('specific_cell_mapped')
      means_r <- suppressMessages(left_join(means_r,cells %>% select(cell_counts,specific_cell_mapped) %>% unique()))
      means_r <- means_r %>% group_by(specific_cell_mapped) %>% mutate(cell_counts = sum(cell_counts)) %>% unique()
      
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
      vars_r <- as.data.frame(vars_r) %>% rownames_to_column('specific_cell_mapped')
      vars_r <- suppressMessages(left_join(vars_r,cells %>% select(cell_counts,specific_cell_mapped) %>% unique()))
      vars_r <- vars_r %>% group_by(specific_cell_mapped) %>% mutate(cell_counts = sum(cell_counts)) %>% unique()
      
      res_r <- rbind(means_r %>% mutate(predicting = 'mean'),
                     vars_r %>% mutate(predicting= 'variance'))
      if (model==''){
        res_r <- res_r %>% mutate(model = 'CPA')
      }else{
        res_r <- res_r %>% mutate(model = substr(model,start = 1,stop=nchar(model)-1))
      }
      res_r <- res_r %>% mutate(translation = trans)
      res_r <- res_r %>% mutate(fold = i)
      results_translation <- rbind(results_translation,res_r)
      message(paste0('Finished fold ',i))
    }
    if (model==''){
      message(paste0('Finished model : ','CPA translate',' into ',species))
    }else{
      message(paste0('Finished model : ',substr(model,start = 1,stop=nchar(model)-1),' translate into ',species))
    }
  }
}
# saveRDS(results_translation,'results/results_translation_specific_cells.rds')

results_translation <- results_translation %>% group_by(predicting,translation,model,specific_cell_mapped) %>%
  mutate(mean_r = mean(r,na.rm = T)) %>% mutate(sd_r = sd(r,na.rm = T)) %>% ungroup()

p_scatter_1 <- ggscatter(results_translation %>% filter(model=='CPA') %>% mutate(translation=ifelse(grepl('human',translation),
                                                                                                   'mouse to human',
                                                                                                   'human to mouse')),
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
ggsave('../figures/cellFrequency_vs_r_translation.eps',
       plot = p_scatter_1,
       device = cairo_ps,
       height = 12,
       width = 12,
       units = 'in',
       dpi = 600)


results4Boxplot <- results_translation %>% filter(model=='CPA')#%>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
# Do for human CPA
data <- results4Boxplot %>% filter(translation=='tohuman') %>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
data <- data %>% group_by(predicting,translation,specific_cell_mapped) %>% mutate(mean_r = median(r,na.rm = T)) %>% ungroup()%>% arrange(desc(mean_r))
data_mean <- data %>% filter(predicting == 'Predicting per cell type mean') 
data_var <- data %>% filter(predicting == 'Predicting per cell type variance')
data_mean_tmp <- data_mean %>% select(specific_cell_mapped,mean_r) %>% unique()
data_mean_tmp <- data_mean_tmp %>% arrange(mean_r)
data_var_tmp <- data_var %>% select(specific_cell_mapped,mean_r) %>% unique()
data_var_tmp <- data_var_tmp %>% arrange(mean_r)
mean_levels <-  data_mean_tmp$specific_cell_mapped
var_levels <- data_var_tmp$specific_cell_mapped
data_mean$specific_cell_mapped <- factor(data_mean$specific_cell_mapped,
                                  levels = mean_levels)
data_var$specific_cell_mapped <- factor(data_var$specific_cell_mapped,
                                 levels = mean_levels)
data <- data %>% mutate(translation = 'mouse to human')
p_human <- ggboxplot(rbind(data_mean,data_var) %>% mutate(translation=ifelse(grepl('human',translation),'mouse to human','human to mouse')),
                     x='specific_cell_mapped',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(translation,predicting))+
  xlab('cell type')+
  #ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  ggtitle('Translation performance')+
  theme(text=element_text(family = 'Arial',size=22),
        axis.text.y = element_text(size=19),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
p_human2 <- ggboxplot(rbind(data_mean,data_var)%>% filter(predicting=="Predicting per cell type mean") %>% 
                        mutate(translation=ifelse(grepl('human',translation),'mouse to human','human to mouse')),
                      x='specific_cell_mapped',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(translation,predicting))+
  xlab('cell type')+
  #ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  #ggtitle('Translation performance')+
  theme(text=element_text(family = 'Arial',size=22),
        axis.text.y = element_text(size=19),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
# print(p_human)
# Do for mouse CPA
data <- results4Boxplot %>% filter(translation=='tomouse') %>% mutate(predicting = paste0('Predicting per cell type ',predicting)) %>% arrange(desc(mean_r))
data <- data %>% group_by(predicting,specific_cell_mapped) %>% mutate(mean_r = median(r,na.rm = T)) %>% ungroup()%>% arrange(desc(mean_r))
data_mean <- data %>% filter(predicting == 'Predicting per cell type mean') 
data_var <- data %>% filter(predicting == 'Predicting per cell type variance')
data_mean_tmp <- data_mean %>% select(specific_cell_mapped,mean_r) %>% unique()
data_mean_tmp <- data_mean_tmp %>% arrange(mean_r)
data_var_tmp <- data_var %>% select(specific_cell_mapped,mean_r) %>% unique()
data_var_tmp <- data_var_tmp %>% arrange(mean_r)
mean_levels <-  data_mean_tmp$specific_cell_mapped
var_levels <- data_var_tmp$specific_cell_mapped
data_mean$specific_cell_mapped <- factor(data_mean$specific_cell_mapped,
                                  levels = mean_levels)
data_var$specific_cell_mapped <- factor(data_var$specific_cell_mapped,
                                 levels = mean_levels)
data <- data %>% mutate(translation = 'human to mouse')
p_mouse <- ggboxplot(rbind(data_mean,data_var) %>% mutate(translation=ifelse(grepl('human',translation),'mouse to human','human to mouse')),
                     x='specific_cell_mapped',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(translation,predicting))+
  xlab('cell type')+
  # ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  ggtitle('')+
  theme(text=element_text(family = 'Arial',size=22),
        axis.text.y = element_text(size=19),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
p_mouse2 <- ggboxplot(rbind(data_mean,data_var) %>% filter(predicting=="Predicting per cell type mean") %>%
                        mutate(translation=ifelse(grepl('human',translation),'mouse to human','human to mouse')),
                     x='specific_cell_mapped',y='r',add = 'jitter',color = 'cell_counts') +
  scale_color_gradient(trans = "log10",high = 'red',low='white',limits = c(0.2,max(rbind(data_mean,data_var)$cell_counts)))+
  facet_wrap(vars(translation,predicting))+
  xlab('cell type')+
  # ylab(expression(R^2))+
  ylab('pearson`s r')+
  ylim(c(0,1))+
  labs(color='cell counts')+
  ggtitle('')+
  theme(text=element_text(family = 'Arial',size=22),
        axis.text.y = element_text(size=19),
        plot.title = element_text(hjust = 0.5),
        legend.position = 'right')+
  coord_flip()
# print(p_mouse)
# p <- p_human / p_mouse
p <- wrap_plots(list(p_human,p_mouse),nrow = 2)
print(p)
ggsave('../figures/cpa_fibrosis_translation_perCelltype.png',
       plot = p,
       width = 16,
       height = 8,
       units = 'in',
       dpi=600)
p <- ggarrange(plotlist = list(p_human2,p_mouse2),common.legend = T,legend = 'right')
print(p)
postscript('../figures/cpa_fibrosis_translation_perCelltype.eps',width = 16,height = 8)
print(p)
dev.off()

### Compare the different approaches per cell type
results_translation <- results_translation %>% group_by(predicting,translation,model,fold) %>%
  mutate(averageCellr = mean(r,na.rm = T)) %>% mutate(sdCellr = sd(r,na.rm = T)) %>% ungroup()

results <- results_translation %>% mutate(model = ifelse(model=='CPA','CPA-based all genes',
                                                            ifelse(model=='homologues','CPA-based homologues',
                                                                   ifelse(model=='DCS','DCS modified v2',
                                                                          'TransCompR'))))
results <- results %>% mutate(predicting = paste0('Predicting per cell type ',predicting))
my_comparisons <- list( c('CPA-based all genes', 'CPA-based homologues'),
                        c('DCS modified v2', 'CPA-based homologues'),
                        c('CPA-based all genes', 'DCS modified v2'),
                        c('CPA-based all genes','TransCompR'))
results <- results %>% mutate(translation = ifelse(translation=='tohuman','mouse to human','human to mouse'))
# tmp <- results %>% mutate() %>% 
#   group_by(specific_cell_mapped,model,predicting,translation) %>% mutate(mu = mean(r,na.rm=T)) %>% ungroup() %>% 
#   select(specific_cell_mapped,mu) %>% 
#   group_by(specific_cell_mapped) %>% mutate(mu =mean(mu)) %>% ungroup() %>% unique() 
# tmp$specific_cell_mapped <- factor(tmp$specific_cell_mapped,levels = tmp$specific_cell_mapped[order(-tmp$mu)])
# results$specific_cell_mapped <- factor(results$specific_cell_mapped,levels = tmp$specific_cell_mapped)
mean_cell_values <- aggregate(r ~ specific_cell_mapped, results, mean)
results$specific_cell_mapped <- factor(results$specific_cell_mapped, levels = mean_cell_values$specific_cell_mapped[order(-mean_cell_values$r)])
results$model <- factor(results$model,
                        levels = c('TransCompR','DCS modified v2',
                                   'CPA-based homologues','CPA-based all genes'))
p1 <- ggboxplot(results %>% select(specific_cell_mapped,r,model,averageCellr,fold,predicting,translation) %>% unique(),
                x='model',y='r',color = 'specific_cell_mapped',
                size=0.75,add='jitter',add.params = list(size=1))  + 
  xlab('')+ 
  # ylab(expression('average R'^2))+
  ylab('pearson`s r')+
  ggtitle("Performance in translating between species")+
  scale_y_continuous(breaks = seq(0,1,0.2),minor_breaks = waiver(),limits = c(0.0,1.4))+
  # geom_hline(yintercept = 0,lty='dashed',linewidth=1)+
  facet_wrap(vars(translation,predicting)) +
  stat_compare_means(label = 'p.signif',
                     method = 'wilcox.test',
                     #step.increase = 0.02,
                     label.y = c(1,1.1,1.2,1.3),
                     tip.length=0.05,
                     size =9,
                     aes(group=model),
                     comparisons = my_comparisons) +
  theme(text = element_text(family = 'Arial',size=23),
        plot.title = element_text(hjust = 0.5,size=23),
        legend.title = element_blank(),
        axis.text.x = element_text(size=25)) +
  coord_flip() #,axis.text.x = element_blank()
print(p1)
ggsave('../figures/fibrosis_performance_translation_comparison_percelltype.eps', 
       plot=p1,
       device = cairo_ps,
       scale = 1,
       width = 16,
       height = 16,
       units = "in",
       dpi = 600)
results$model <- factor(results$model,
                        levels = c('CPA-based all genes','CPA-based homologues',
                                   'DCS modified v2','TransCompR'))
p2 <- ggboxplot(results %>% select(model,averageCellr,fold,predicting,translation) %>% unique(),
                x='model',y='averageCellr',color = 'model',
                size=1,add='jitter',add.params = list(size=1))  + 
  xlab('')+ 
  # ylab(expression('average R'^2))+
  ylab('pearson`s r')+
  ggtitle("Performance in translating between species")+
  scale_y_continuous(breaks = seq(0.4,1,0.1),minor_breaks = waiver(),limits = c(0.4,1.1))+
  # geom_hline(yintercept = 0,lty='dashed',linewidth=1)+
  facet_wrap(vars(translation,predicting)) +
  stat_compare_means(label = 'p.signif',
                     method = 'wilcox.test',
                     #step.increase = 0.02,
                     label.y = c(0.75,0.85,0.95,1.05),
                     tip.length=0.05,
                     size =6,
                     aes(group=model),
                     comparisons = my_comparisons) +
  theme(text = element_text(family = 'Arial',size=23),
        plot.title = element_text(hjust = 0.5,size=23),
        legend.title = element_blank(),
        axis.text.x = element_blank())
print(p2)
ggsave('../figures/fibrosis_performance_translation_comparison_average.eps', 
       plot=p2,
       device = cairo_ps,
       scale = 1,
       width = 12,
       height = 12,
       units = "in",
       dpi = 600)

p3 <- ggboxplot(results %>% select(model,r,fold,predicting,translation,specific_cell_mapped) %>% unique(),
                x='model',y='r',color = 'model',
                size=1,add='jitter',add.params = list(size=1))  + 
  xlab('')+ 
  # ylab(expression('average R'^2))+
  ylab('pearson`s r')+
  ggtitle("Performance in translating between species")+
  scale_y_continuous(breaks = seq(0.0,1,0.2),minor_breaks = waiver(),limits = c(0.0,1.3))+
  # geom_hline(yintercept = 0,lty='dashed',linewidth=1)+
  facet_wrap(vars(translation,predicting)) +
  stat_compare_means(label = 'p.signif',
                     method = 'wilcox.test',
                     #step.increase = 0.02,
                     label.y = c(0.85,0.95,1.05,1.15),
                     tip.length=0.05,
                     size =6,
                     aes(group=model),
                     comparisons = my_comparisons) +
  theme(text = element_text(family = 'Arial',size=24),
        plot.title = element_text(hjust = 0.5,size=24),
        legend.title = element_blank(),
        axis.text.x = element_blank())
print(p3)
ggsave('../figures/fibrosis_performance_translation_comparison_allgrouped.png', 
       plot=p3,
       scale = 1,
       width = 16,
       height = 8,
       units = "in",
       dpi = 600)
postscript('../figures/fibrosis_performance_translation_comparison_allgrouped.eps',width = 16,height = 8)
print(p3)
dev.off()
