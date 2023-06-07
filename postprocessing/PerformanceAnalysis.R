library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpubr)
  
### Performance of different size inputs-----------------
data <- data.table::fread('../results/MI_results/diffsize_allgenes_10foldvalidation_withCPA_1000ep512bs_a375_ht29.csv')
colnames(data)[1] <- 'fold'
results <- data[,c(1,4,5,6,7,8,9)]

data_origin <- data.table::fread('../results/MI_results/allgenes_10foldvalidation_withCPA_1000ep512bs_a375_ht29.csv')
res_direct <- data_origin[,c(16,17,18,19)]
res_direct$fold <- as.numeric(rownames(res_direct))-1

results <- results %>% gather('metric','value',-fold)
results <- results %>% mutate(translation = ifelse(grepl('HT29',metric),'A375 to HT29','HT29 to A375'))
results <- results %>% 
  mutate(metric = ifelse(grepl('spear',metric),'spearman',ifelse(grepl('acc',metric),'accuracy','pearson')))

res_direct <- res_direct %>% gather('metric','value',-fold)
res1 <- res_direct %>% filter(metric!='DirectAcc_ht29')
res1 <- res1 %>% mutate(translation='A375 to HT29') %>% 
  mutate(metric=ifelse(grepl('Acc',metric),'accuracy',ifelse(grepl('spearman',metric),'spearman','pearson')))
res2 <- res_direct %>% filter(metric!='DirectAcc_a375')
res2 <- res2 %>% mutate(translation='HT29 to A375') %>% 
  mutate(metric=ifelse(grepl('Acc',metric),'accuracy',ifelse(grepl('spearman',metric),'spearman','pearson')))

all_results <- rbind(results %>% mutate(model='model translation'),
                     res1%>% mutate(model='direct translation in 10k genes'),
                     res2%>% mutate(model='direct translation in 10k genes'))
all_results$model <- factor(all_results$model,levels = c('direct translation in 10k genes','model translation'))

p <- ggboxplot(all_results,x='metric',y='value',color='model',add='jitter',
               add.params = list(size = 2),size = 1) + 
  scale_color_manual(breaks = c('model translation','direct translation in 10k genes'),
                     values=scales::hue_pal()(length(unique(all_results$model))))+
  ggtitle('Performance with different input sizes (A375 with 978 genes, HT29 with ~10k genes)') + ylim(c(0,0.75))+
  facet_wrap(~ translation)+
  theme_minimal(base_family = "Arial",base_size = 38)+ 
  theme(text = element_text("Arial",size = 38),
        axis.title = element_text("Arial",size = 38,face = "bold"),
        axis.text = element_text("Arial",size = 38,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 38,face = "bold"),
        plot.title = element_text(hjust = 0.75,size=30,face='bold'),legend.position='bottom')
p <- p+ stat_compare_means(aes(group=model),method='wilcox.test',label='p.signif',size=8)
p <- p+ scale_x_discrete(expand = c(0.2, 0))
print(p)
ggsave(
  '../figures/performance_diffsize_a375_ht29.eps',
  plot = p,
  device = 'eps',
  scale = 1,
  width = 16,
  height = 12,
  units = "in",
  dpi = 600,
)
png('../figures/performance_diffsize_a375_ht29.png',width=12,height=9,units = "in",res = 600)
print(p)
dev.off()

# Load folders--------------
folders <- c('A375_HT29','A375_PC3','HA1E_VCAP',
             'HT29_MCF7','HT29_PC3','MCF7_HA1E',
             'PC3_HA1E','MCF7_PC3')

cell_pairs <- strsplit(folders,'_')
cells <- unique(unlist(strsplit(folders,'_')))

# Load all cmap and keep only these cell-lines and calculate correlation-------------
cmap <- data.table::fread('../preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv',header=T) %>% 
  column_to_rownames('V1')

## Run the following for landmarks analysis
lands = data.table::fread('../preprocessing/preprocessed_data/cmap_landmarks_HT29_A375.csv',header=T) %>%
  column_to_rownames('V1')
lands = colnames(lands)
cmap <- cmap[,lands]
gc()

#Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% 
  mutate(quality_replicates = ifelse(is_exemplar_sig==1 & qc_pass==1 & nsample>=3,1,0))
sigInfo <- sigInfo %>% filter(pert_type=='trt_cp')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
sigInfo <- sigInfo %>% filter(tas>=0.3)
sigInfo <- sigInfo %>% 
  mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>%
  mutate(dupl_counts = n()) %>% ungroup()
sigInfo <- sigInfo  %>% mutate(conditionId = paste0(cmap_name,"_",pert_idose,"_",pert_itime))
sigInfo <- sigInfo %>% filter(cell_iname %in% cells)
cmap <- cmap[which(rownames(cmap) %in% sigInfo$sig_id),]
cmap <- t(cmap)

conds_cor <- NULL
i <- 1
for (pair in cell_pairs){
  pairedInfo <- left_join(sigInfo %>% filter(cell_iname==pair[1]) %>% 
                            select(c('sig_id.x'='sig_id'),conditionId,cmap_name) %>% unique(), 
                          sigInfo %>% filter(cell_iname==pair[2]) %>% 
                            select(c('sig_id.y'='sig_id'),conditionId,cmap_name) %>% unique()) %>% 
    filter(!is.na(sig_id.x) & !is.na(sig_id.y))
  
  sig1 <- pairedInfo$sig_id.x
  sig2 <- pairedInfo$sig_id.y
  cmap1 <- t(cmap[,sig1])
  cmap2 <- t(cmap[,sig2])
  conds_cor[i] <- cor(c(cmap1),c(cmap2))
  i <- i+1
} 

# Load and prepare data------------------
i <- 1
for (folder in folders){
  files <- list.files(paste0('../preprocessing/preprocessed_data/sampledDatasetes/',folder),
                      recursive = F,full.names = T)
  #files <- files[which(!grepl('landmarks',files))]
  files <- files[which(grepl('landmarks',files))]
  files <- as.data.frame(files)
  files <- files %>% 
    mutate(csv=grepl('.csv',files)) %>% 
    mutate(rds=grepl('.rds',files)) %>% 
    filter(csv==T) %>% select(-rds,-csv) %>% unique() %>%
    mutate(size=strsplit(files,'/')) %>% unnest(size) %>%
    mutate(keep = grepl('.csv',size)) %>% filter(keep==T) %>%
    select(-keep) %>% mutate(size=strsplit(size,'.csv')) %>% unnest(size) %>% unique() %>%
    mutate(size=strsplit(size,'sample_len')) %>% unnest(size) %>% 
    filter(size!='') %>% unique()
  
  #uncomment this for landmarks
  files <- files %>% mutate(size=strsplit(size,'_')) %>% unnest(size) %>%
    filter(size!='') %>% filter(size!='landmarks') %>% unique()
  files$size <- as.numeric(files$size)
  for (j in 1:nrow(files)){
    data <- data.table::fread(files$files[j])
    rownames(data) <- NULL
    colnames(data)[1] <- 'fold'
    data <- data %>% gather('metric','value',-fold)
    
    # comment this if we use landmarks and use the line bellow
    data <- data %>% mutate(metric=ifelse(grepl('pearson',metric) & !grepl('Direct',metric) & !grepl('recon',metric),'pearson translation',
                                   ifelse(grepl('spear',metric) & !grepl('Direct',metric) & !grepl('recon',metric),'spearman translation',
                                   ifelse(grepl('acc',metric) & !grepl('Direct',metric) & !grepl('recon',metric),'sign accuracy translation',
                                   ifelse(metric=='F1_score','F1 score',
                                   ifelse(metric=='ClassAccuracy','Accuracy','other'))))))
    # if (folder == 'A375_HT29'){
    #   data <- data %>% mutate(metric=ifelse(metric=='model_pearsonHT29' | metric=='model_pearsonA375','pearson translation',
    #                           ifelse(metric=='model_spearHT29' | metric=='model_spearA375','spearman translation',
    #                           ifelse(metric=='model_accHT29' | metric=='model_accA375','sign accuracy translation',
    #                           ifelse(metric=='recon_pear_ht29' | metric=='recon_pear_a375','pearson reconstruction',
    #                           ifelse(metric=='recon_spear_ht29' | metric=='recon_spear_a375','spearman reconstruction',
    #                           ifelse(metric=='F1_score','F1 score',
    #                           ifelse(metric=='ClassAccuracy','Accuracy','other'))))))))
    # } else {
    #   data <- data %>% mutate(metric=ifelse(metric=='model_pearsonPC3' | metric=='model_pearsonA375','pearson translation',
    #                           ifelse(metric=='model_spearPC3' | metric=='model_spearA375','spearman translation',
    #                           ifelse(metric=='model_accPC3' | metric=='model_accA375','sign accuracy translation',
    #                           ifelse(metric=='recon_pear_pc3' | metric=='recon_pear_a375','pearson reconstruction',
    #                           ifelse(metric=='recon_spear_pc3' | metric=='recon_spear_a375','spearman reconstruction',
    #                           ifelse(metric=='F1_score','F1 score',
    #                           ifelse(metric=='ClassAccuracy','Accuracy','other'))))))))
    # }
    
    data <- data %>% filter(metric!='other')
    data <- data %>% group_by(fold,metric) %>% mutate(value=mean(value)) %>% ungroup() %>% unique()
    data <- data %>% mutate(size=files$size[j])
    data <- data %>% group_by(metric,size) %>%
      mutate(mean_value=mean(value)) %>% 
      mutate(std_value=sd(value)) %>% 
      ungroup()
    if (j==1){
      results <- data
    } else{
      results <- rbind(results,data)
    }
  }
  results <- results %>% mutate(pair=folder)
  results <- results %>% mutate(direct_translation = conds_cor[i])
  if (folder == folders[1]){
    all_results <- results
  }else{
    all_results <- rbind(all_results,results)
  }
  i <- i+1
}

# Visualize sample size and baseline correlation performance-----------------
#png('../figures/Corr_vs_samplesize_lands.png',width=9,height=8,units = "in",res = 600)
p <- ggplot(all_results %>% filter(metric=='pearson translation'),aes(size,mean_value,color=pair)) +
  geom_ribbon(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5),fill = pair),
              linetype=0,alpha=0.1) +
  #geom_smooth(aes(color=pair),se = F)+
  geom_line(aes(color=pair),size=1) +
  #geom_line(aes(x=size,y=direct_translation,color=pair),linetype = 2)+
  xlab('total sample size (equal size per cell line)') + ylab('Average pearson`s r for translation')+
  ggtitle('Performance curve for increasing number of available data') +
  ylim(c(0,max(all_results$value)))+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 26,face = "bold"),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        plot.title = element_text(hjust = 0.5,size=26,face = "bold"),
        legend.position='top',
        panel.grid.major = element_line(linewidth=1.5))
#dev.off()
print(p)
ggsave(
  '../figures/Corr_vs_samplesize_lands.eps',
  plot = last_plot(),
  device=cairo_ps,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)

correlation_res <- all_results %>% filter(metric=='pearson translation') %>% 
  mutate(better=value>direct_translation) %>%
  mutate(required_size=ifelse(better==T,size,10000)) %>%
  group_by(pair,fold) %>% mutate(required_size = min(required_size)) %>% ungroup()

png('../figures/required_samplesize_lands.png',width=9,height=6,units = "in",res = 600)
ggboxplot(correlation_res %>% select(pair,required_size) %>% unique(),
       x='pair',y='required_size',add='jitter') +
  geom_hline(yintercept = mean(correlation_res$required_size),color='red',linetype=2,size=1.5)+
  xlab('Pair of cell-lines') + ylab('Required sample size')+ ylim(c(0,max(correlation_res$required_size)))+
  ggtitle('Sample size required to overcome direct translation in 5-fold cross validation') +
  theme_gray(base_family = "Arial",base_size = 20)+
  theme(plot.title = element_text(hjust = 0.5,size=20))
dev.off()

correlation_res <- correlation_res %>% group_by(pair) %>% mutate(max_size = max(size)) %>% ungroup() %>%
  filter(size==max_size) %>% select(-size)
correlation_res <- correlation_res %>% group_by(pair) %>% 
  mutate(DeltaCorr = value-direct_translation) %>% 
  mutate(mean_delta=mean(DeltaCorr)) %>% 
  mutate(std_delta=sd(DeltaCorr)) %>%ungroup()

# Power law
performance_mdl <- nls(mean_value ~ 1-b*direct_translation^c, data = correlation_res, start = list(b=1,c=-0.5), trace = T)
#predict(performance_mdl,newdata = list('direct_translation'=0.5))
ys <- predict(performance_mdl,newdata = list('direct_translation'=correlation_res$direct_translation))
correlation_res$smoothed <- ys


# Plot performace vs direct pearson
png('../figures/MaxCorr_vs_directCorr_lands.png',width=10,height=8,units = "in",res = 600)
ggplot(correlation_res %>% select(direct_translation,value,mean_value,std_value,smoothed) %>% unique(),
       aes(direct_translation,mean_value))+ geom_point(size=2) +
  geom_errorbar(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5)),
                width = 0.01,size=0.7)+
  geom_line(alpha=0.5)+
  geom_smooth(aes(direct_translation,smoothed),linetype=2,color='red')+
  scale_x_continuous(name="Pearson correlation of direct translation",
                     limits=c(0.25, 0.5))+
  scale_y_continuous(name="Average pearson correlation of model translation",
                     breaks = c(0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7),
                     #breaks = c(0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6),
                     limits=c(0.3, 0.7))+
  coord_equal(ratio = 1/2.5)+
  ggtitle('Maximum possible performance as a function of base correlation between cell-line pairs')+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 22,face = "bold"),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        plot.title = element_text(hjust = 0.8,size=18,face = "bold"),
        panel.grid.major = element_line(linewidth=1.5))
dev.off()
ggsave(
  '../figures/MaxCorr_vs_directCorr_lands.eps',
  plot = last_plot(),
  device=cairo_ps,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
#Plot increase in performance vs direct correlation
png('../figures/DeltaCorr_vs_directCorr_lands.png',width=9,height=8,units = "in",res = 600)
ggplot(correlation_res %>% select(direct_translation,DeltaCorr,mean_delta,std_delta) %>% unique(),
       aes(direct_translation,mean_delta))+ geom_point() +
  geom_errorbar(aes(ymin = mean_delta - std_delta/sqrt(5), ymax = mean_delta + std_delta/sqrt(5)))+
  geom_line(linetype=2,alpha=0.35)+
  geom_hline(aes(yintercept=mean(correlation_res$DeltaCorr)),color='red',linetype=2,size=1.3)+
  scale_x_continuous(name="Pearson correlation of direct translation",
                     limits=c(0.25, 0.5))+
  scale_y_continuous(name=expression(Delta*"Correlation"),
                     limits=c(0.0, 0.3))+
  coord_equal(ratio = 1/2.5)+
    ggtitle('Improvement in performance as a function of base correlation between cell-line pairs')+
  theme_minimal(base_family = "Arial",base_size = 20)+
  theme(plot.title = element_text(hjust = 0.5,size=16))
dev.off()

# Visualize pair ratios correlation performance-----------------
folders <- c('ratiosA375','ratiosHT29')

i <- 1
for (folder in folders){
  cell <- strsplit(folder,'ratios')[[1]][2]
  files <- list.files(paste0('../preprocessing/preprocessed_data/sampledDatasetes/',folder),
                      recursive = F,full.names = T)
  #files <- files[which(!grepl('landmarks',files))]
  files <- files[which(grepl('landmarks',files) | grepl('.rds',files))]
  files <- as.data.frame(files)
  filesRDS <- files %>% 
    mutate(csv=grepl('.csv',files)) %>% 
    mutate(rds=grepl('.rds',files)) %>% 
    filter(rds==T) %>% select(-rds,-csv) %>% unique()
  ratio <- rep(0,length(filesRDS$files))
  total_samples<- rep(0,length(filesRDS$files))
  for (k in 1:length(filesRDS$files)){
    f <- filesRDS$files[k]
    data <- readRDS(f)
    ratio[k] <- nrow(data %>% filter(cell_iname==cell))/nrow(data %>% filter(cell_iname!=cell))
    total_samples[k] <- nrow(data)
    
  }
  files <- files %>% 
    mutate(csv=grepl('.csv',files)) %>% 
    mutate(rds=grepl('.rds',files)) %>% 
    filter(csv==T) %>% select(-rds,-csv) %>% unique()
  files$ratio <- ratio
  files$total_samples <- total_samples
  
  for (j in 1:nrow(files)){
    data <- data.table::fread(files$files[j])
    rownames(data) <- NULL
    colnames(data)[1] <- 'fold'
    data <- data %>% gather('metric','value',-fold)
    
    data <- data %>% filter(metric=='model_pearsonHT29' | metric=='model_pearsonA375')
    
    data <- data %>% mutate(metric=ifelse(metric=='model_pearsonHT29','A375 to HT29','HT29 to A375'))
    data <- data %>% group_by(fold,metric) %>% mutate(value=mean(value)) %>% ungroup() %>% unique()
    data <- data %>% mutate(ratio=files$ratio[j])
    data <- data %>% group_by(metric,ratio) %>%
      mutate(mean_value=mean(value)) %>% 
      mutate(std_value=sd(value)) %>% 
      ungroup()
    data <- data %>% mutate(total_samples=files$total_samples[j])
    if (j==1){
      results <- data
    } else{
      results <- rbind(results,data)
    }
  }
  results <- results %>% mutate(cell=cell)
  results <- results %>% mutate(direct_translation = conds_cor[i])
  if (folder == folders[1]){
    all_results <- results
  }else{
    all_results <- rbind(all_results,results)
  }
  i <- i+1
}

#png('../figures/Corr_vs_cellinesratio_lands.png',width=9,height=8,units = "in",res = 600)
p <- ggplot(all_results,aes(ratio,mean_value)) +
  geom_line(aes(color=metric,linetype=cell)) + geom_point(aes(color=metric),size=1.2) +
  geom_errorbar(aes(ymin = mean_value - std_value/sqrt(5), 
                    ymax = mean_value + std_value/sqrt(5),color=metric),
                size=0.5,
                width=0.01)+
  xlab('ratio of number of data') + ylab('Average pearson`s r for translation')+
  ggtitle('Performance as a function of the ratio of the number of data of the 2 cell-lines') +
  ylim(c(0,max(results$value)))+
  labs(color='Translation',linetype='Cell with increasing data')+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 22,face = "bold"),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        plot.title = element_text(hjust = 0.5,size=18,face = "bold"),
        legend.text = element_text("Arial",size = 20),
        legend.title = element_text("Arial",size = 20),
        panel.grid.major = element_line(linewidth=1.5))+
  geom_hline(aes(yintercept= 0.43, linetype = 2),linetype=1,color='red')

#dev.off()
print(p)
ggsave(
  '../figures/Corr_vs_cellinesratio_lands.eps',
  plot = p,
  device = cairo_ps,
  scale = 1,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
#### Paired ratio results ------------
folder <- 'pairedPercs'
files <- list.files(paste0('../preprocessing/preprocessed_data/sampledDatasetes/',folder),
                    recursive = F,full.names = T)
#files <- files[which(!grepl('landmarks',files))]
files <- files[which(grepl('landmarks',files) | grepl('.rds',files))]
files <- as.data.frame(files)
filesRDS <- files %>% 
  mutate(csv=grepl('.csv',files)) %>% 
  mutate(rds=grepl('.rds',files)) %>% 
  filter(rds==T) %>% select(-rds,-csv) %>% unique()
ratio <- rep(0,length(filesRDS$files))
total_samples<- rep(0,length(filesRDS$files))
for (i in 1:length(filesRDS$files)){
  f <- filesRDS$files[i]
  data <- readRDS(f)
  pairedInfo <- left_join(data %>% filter(cell_iname=='A375') %>% 
                            select(c('sig_id.x'='sig_id'),conditionId,cmap_name) %>% unique(), 
                          data %>% filter(cell_iname=='HT29') %>% 
                            select(c('sig_id.y'='sig_id'),conditionId,cmap_name) %>% unique()) %>% 
    filter(!is.na(sig_id.x) & !is.na(sig_id.y))
  #data <- data %>% filter(!(sig_id %in% unique(c(pairedInfo$sig_id.x,pairedInfo$sig_id.y))))
  ratio[i] <- 2*nrow(pairedInfo)/nrow(data)
  total_samples[i] <- nrow(data)
  
}
files <- files %>% 
  mutate(csv=grepl('.csv',files)) %>% 
  mutate(rds=grepl('.rds',files)) %>% 
  filter(csv==T) %>% select(-rds,-csv) %>% unique()
files$ratio <- ratio
files$total_samples <- total_samples

for (j in 1:nrow(files)){
  data <- data.table::fread(files$files[j])
  rownames(data) <- NULL
  colnames(data)[1] <- 'fold'
  data <- data %>% gather('metric','value',-fold)
  
  data <- data %>% mutate(metric=ifelse(metric=='model_pearsonHT29' | metric=='model_pearsonA375','pearson translation',
                                        ifelse(metric=='model_spearHT29' | metric=='model_spearA375','spearman translation',
                                        ifelse(metric=='model_accHT29' | metric=='model_accA375','sign accuracy translation',
                                        ifelse(metric=='recon_pear_ht29' | metric=='recon_pear_a375','pearson reconstruction',
                                        ifelse(metric=='recon_spear_ht29' | metric=='recon_spear_a375','spearman reconstruction',
                                        ifelse(metric=='F1_score','F1 score',
                                        ifelse(metric=='ClassAccuracy','Accuracy','other'))))))))
  data <- data %>% filter(metric!='other')
  data <- data %>% group_by(fold,metric) %>% mutate(value=mean(value)) %>% ungroup() %>% unique()
  data <- data %>% mutate(ratio=files$ratio[j])
  data <- data %>% group_by(metric,ratio) %>%
    mutate(mean_value=mean(value)) %>% 
    mutate(std_value=sd(value)) %>% 
    ungroup()
  data <- data %>% mutate(total_samples=files$total_samples[j])
  if (j==1){
    results <- data
  } else{
    results <- rbind(results,data)
  }
}
results <- results %>% mutate(pair=folder)
results <- results %>% mutate(direct_translation = conds_cor[i])

p <- ggplot(results %>% filter(metric=='pearson translation'),aes(ratio*100,mean_value)) +
  geom_ribbon(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5)),
              linetype=0,alpha=0.2,fill = "grey70") +
  geom_line(size=0.8,alpha=0.8) + geom_point(aes(color =total_samples),size=5,shape=18) +
  xlab('Percentage of paired conditions (%)') + ylab('Average pearson`s r for translation')+
  ggtitle('Performance for increasing percentage of paired conditions to the total number of conditions') +
  ylim(c(0,max(results$value)))+
  theme_minimal(base_family = "Arial",base_size = 26)+
  theme(text = element_text("Arial",size = 26),
        axis.title = element_text("Arial",size = 22,face = "bold"),
        axis.text = element_text("Arial",size = 26,face = "bold"),
        axis.text.x = element_text("Arial",angle = 0,size = 26,face = "bold"),
        plot.title = element_text(hjust = 0.0,size=16,face = "bold"),
        legend.text = element_text("Arial",size = 20),
        legend.title = element_text("Arial",size = 20),
        panel.grid.major = element_line(linewidth=1.5))+
  scale_color_gradient2('total number of samples',
                        low = "yellow", 
                        mid = "orange", 
                        high = "red", 
                        midpoint = 1900)+
  geom_hline(aes(yintercept= 0.43, linetype = 2),linetype=2,color='red')
print(p)
png('../figures/Corr_vs_pairedratio_lands.png',width=9,height=8,units = "in",res = 600)
print(p)
dev.off()
ggsave(
  '../figures/Corr_vs_pairedratio_lands.eps',
  plot = p,
  device=cairo_ps,
  width = 12,
  height = 9,
  units = "in",
  dpi = 600,
)
