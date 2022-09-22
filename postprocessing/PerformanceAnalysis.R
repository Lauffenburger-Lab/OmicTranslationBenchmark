library(tidyverse)
library(ggplot2)
library(ggsignif)
library(ggpubr)

# Load folders--------------
folders <- c('A375_HT29','A375_PC3','HA1E_VCAP',
             'HT29_MCF7','HT29_PC3','MCF7_HA1E',
             'PC3_HA1E','MCF7_PC3')

cell_pairs <- strsplit(folders,'_')
cells <- unique(unlist(strsplit(folders,'_')))

# Load all cmap and keep only these cell-lines and calculate correlation-------------
cmap <- data.table::fread('../preprocessing/preprocessed_data/cmap_all_genes_q1_tas03.csv',header=T) %>% 
  column_to_rownames('V1')

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
  files$size <- as.numeric(files$size)
  for (j in 1:nrow(files)){
    data <- data.table::fread(files$files[j])
    rownames(data) <- NULL
    colnames(data)[1] <- 'fold'
    data <- data %>% gather('metric','value',-fold)
    
    if (folder == 'A375_HT29'){
      data <- data %>% mutate(metric=ifelse(metric=='model_pearsonHT29' | metric=='model_pearsonA375','pearson translation',
                              ifelse(metric=='model_spearHT29' | metric=='model_spearA375','spearman translation',
                              ifelse(metric=='model_accHT29' | metric=='model_accA375','sign accuracy translation',
                              ifelse(metric=='recon_pear_ht29' | metric=='recon_pear_a375','pearson reconstruction',
                              ifelse(metric=='recon_spear_ht29' | metric=='recon_spear_a375','spearman reconstruction',
                              ifelse(metric=='F1_score','F1 score',
                              ifelse(metric=='ClassAccuracy','Accuracy','other'))))))))
    } else {
      data <- data %>% mutate(metric=ifelse(metric=='model_pearsonPC3' | metric=='model_pearsonA375','pearson translation',
                              ifelse(metric=='model_spearPC3' | metric=='model_spearA375','spearman translation',
                              ifelse(metric=='model_accPC3' | metric=='model_accA375','sign accuracy translation',
                              ifelse(metric=='recon_pear_pc3' | metric=='recon_pear_a375','pearson reconstruction',
                              ifelse(metric=='recon_spear_pc3' | metric=='recon_spear_a375','spearman reconstruction',
                              ifelse(metric=='F1_score','F1 score',
                              ifelse(metric=='ClassAccuracy','Accuracy','other'))))))))
    }
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
png('../figures/Corr_vs_samplesize.png',width=9,height=8,units = "in",res = 600)
ggplot(all_results %>% filter(metric=='pearson translation'),aes(size,mean_value,color=pair)) +
  geom_ribbon(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5),fill = pair),
              linetype=0,alpha=0.1) +
  #geom_smooth(aes(color=pair),se = F)+
  geom_line(aes(color=pair),size=1) +
  #geom_line(aes(x=size,y=direct_translation,color=pair),linetype = 2)+
  xlab('total sample size (equal size per cell line)') + ylab('Average pearson`s r for translation')+
  ggtitle('Performance curve for increasing number of available data') +
  ylim(c(0,max(all_results$value)))+
  theme_minimal(base_family = "serif",base_size = 15)+
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

correlation_res <- all_results %>% filter(metric=='pearson translation') %>% 
  mutate(better=value>direct_translation) %>%
  mutate(required_size=ifelse(better==T,size,10000)) %>%
  group_by(pair,fold) %>% mutate(required_size = min(required_size)) %>% ungroup()

png('../figures/required_samplesize.png',width=9,height=6,units = "in",res = 600)
ggboxplot(correlation_res %>% select(pair,required_size) %>% unique(),
       x='pair',y='required_size',add='jitter') +
  geom_hline(yintercept = mean(correlation_res$required_size),color='red',linetype=2,size=1.5)+
  xlab('Pair of cell-lines') + ylab('Required sample size')+ ylim(c(0,max(correlation_res$required_size)))+
  ggtitle('Sample size required to overcome direct translation in 5-fold cross validation') +
  theme_gray(base_family = "serif",base_size = 13)+
  theme(plot.title = element_text(hjust = 0.5,size=14))
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
png('../figures/MaxCorr_vs_directCorr.png',width=9,height=8,units = "in",res = 600)
ggplot(correlation_res %>% select(direct_translation,value,mean_value,std_value,smoothed) %>% unique(),
       aes(direct_translation,mean_value))+ geom_point() +
  geom_errorbar(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5)))+
  geom_line(alpha=0.35)+
  geom_smooth(aes(direct_translation,smoothed),linetype=2,color='red')+
  scale_x_continuous(name="Pearson correlation of direct translation",
                     limits=c(0.25, 0.45))+
  scale_y_continuous(name="Average pearson correlation of model translation",
                     breaks = c(0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6),
                     limits=c(0.25, 0.6))+
  coord_equal(ratio = 1/2.5)+
  ggtitle('Maximum possible performance as a function of base correlation between cell-line pairs')+
  theme_minimal(base_family = "serif",base_size = 13)+
  theme(plot.title = element_text(hjust = 0.5,size=14))
dev.off()

#Plot increase in performance vs direct correlation
png('../figures/DeltaCorr_vs_directCorr.png',width=9,height=8,units = "in",res = 600)
ggplot(correlation_res %>% select(direct_translation,DeltaCorr,mean_delta,std_delta) %>% unique(),
       aes(direct_translation,mean_delta))+ geom_point() +
  geom_errorbar(aes(ymin = mean_delta - std_delta/sqrt(5), ymax = mean_delta + std_delta/sqrt(5)))+
  geom_line(linetype=2,alpha=0.35)+
  geom_hline(aes(yintercept=mean(correlation_res$DeltaCorr)),color='red',linetype=2,size=1.3)+
  scale_x_continuous(name="Pearson correlation of direct translation",
                     limits=c(0.25, 0.45))+
  scale_y_continuous(name=expression(Delta*"Correlation"),
                     limits=c(0.0, 0.3))+
  coord_equal(ratio = 1/2.5)+
    ggtitle('Improvement in performance as a function of base correlation between cell-line pairs')+
  theme_minimal(base_family = "serif",base_size = 13)+
  theme(plot.title = element_text(hjust = 0.5,size=14))
dev.off()

# Visualize pair ratios correlation performance-----------------
folders <- c('ratiosA375','ratiosHT29')

i <- 1
for (folder in folders){
  cell <- strsplit(folder,'ratios')[[1]][2]
  files <- list.files(paste0('../preprocessing/preprocessed_data/sampledDatasetes/',folder),
                      recursive = F,full.names = T)
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

png('../figures/Corr_vs_cellinesratio.png',width=9,height=8,units = "in",res = 600)
ggplot(all_results,aes(ratio,mean_value)) +
  geom_line(aes(color=metric,linetype=cell)) + geom_point(aes(color=metric),size=1.2) +
  geom_errorbar(aes(ymin = mean_value - std_value/sqrt(5), 
                    ymax = mean_value + std_value/sqrt(5),color=metric),
                size=0.5,
                width=0.01)+
  xlab('ratio of number of data') + ylab('Average pearson`s r for translation')+
  ggtitle('Performance as a function of the ratio of the number of data of the 2 cell-lines') +
  ylim(c(0,max(results$value)))+
  labs(color='Translation',linetype='Cell with increasing data')+
  theme_minimal(base_family = "serif",base_size = 15)+
  theme(plot.title = element_text(hjust = 0.5,size=15))+
  geom_hline(aes(yintercept= 0.43, linetype = 2),linetype=1,color='red')

dev.off()

#### Paired ratio results ------------
folder <- 'pairedPercs'
files <- list.files(paste0('../preprocessing/preprocessed_data/sampledDatasetes/',folder),
                    recursive = F,full.names = T)
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

png('../figures/Corr_vs_pairedratio.png',width=9,height=8,units = "in",res = 600)
ggplot(results %>% filter(metric=='pearson translation'),aes(ratio*100,mean_value)) +
  geom_ribbon(aes(ymin = mean_value - std_value/sqrt(5), ymax = mean_value + std_value/sqrt(5)),
              linetype=0,alpha=0.2,fill = "grey70") +
  geom_line(size=0.8,alpha=0.8) + geom_point(aes(color =total_samples),size=5,shape=18) +
  xlab('Percentage of paired conditions (%)') + ylab('Average pearson`s r for translation')+
  ggtitle('Performance for increasing the percentage of paired conditions to the total number of conditions') +
  ylim(c(0,max(results$value)))+
  theme_minimal(base_family = "serif",base_size = 13)+
  theme(plot.title = element_text(hjust = 0.5,size=13))+
  scale_color_gradient2('total number of samples',
                        low = "yellow", 
                        mid = "orange", 
                        high = "red", 
                        midpoint = 1900)+
  geom_hline(aes(yintercept= 0.43, linetype = 2),linetype=2,color='red')

dev.off()
