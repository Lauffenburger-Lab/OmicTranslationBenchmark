### Nightingale Rose Plots of NHP feature importance by epitope and feature type

dictionary <- read.csv(file = "/Users/kpullen/autoencoder/serology_feature_dictionary.csv",header= TRUE)
scores <- read.csv(file = "/Users/kpullen/autoencoder/importance_Primates2Humans_protection_merged.csv",header= TRUE)
all<-merge(dictionary,scores,by="feature")
all<-all[-c(178),]
all$epitope<-tolower(all$epitope)

all$rank<-rank(abs(all$mean_percentage_score))
all$rank_abs<-rank(abs(all$mean_percentage_score))
all$percent_rank<- dplyr::percent_rank(abs(all$mean_percentage_score))

## Plot epitope Nightingale Rose Plot
epitope_plot<- all[,c("epitope","percent_rank")]
epitope_plot<- aggregate(epitope_plot$percent_rank, by=list(epitope_plot$epitope), FUN=median)
colnames(epitope_plot)<-c('variable','value')
epitope_plot$order<-c(1,4,8,6,3,5,7,11,9,13,12,10,2) # manual ordering of the barplots/petals based on epitope location on the virus
ggplot2::ggplot(epitope_plot, ggplot2::aes(x=reorder(variable,order),y= value, fill = factor(variable))) +ggplot2::geom_bar(stat='identity',width = 1)+ggplot2::coord_polar() + ggplot2::geom_hline(yintercept = seq(0,0.8, by = 0.2),color = 'grey', size = 1) +ggplot2::geom_vline(xintercept = seq(.5, 20, by = 1),color = 'grey', size = 1) +ggplot2::theme(panel.background = ggplot2::element_blank())

## Plot feature type Nightingale Rose Plot
featuretype_plot<- all[,c("type","percent_rank")]
featuretype_plot<- aggregate(featuretype_plot$percent_rank, by=list(featuretype_plot$type), FUN=median)
colnames(featuretype_plot)<-c('variable','value')
featuretype_plot$order<-c(1,3,5,7,9,8,4,6,11,10,12,2)  # manual ordering of the barplots/petals based on human or NHP receptor
ggplot2::ggplot(featuretype_plot, ggplot2::aes(x=reorder(variable,order),y= value, fill = factor(variable))) +ggplot2::geom_bar(stat='identity',width = 1)+ggplot2::coord_polar() + ggplot2::geom_hline(yintercept = seq(0,0.8, by = 0.2),color = 'grey', size = 1) +ggplot2::geom_vline(xintercept = seq(.5, 20, by = 1),color = 'grey', size = 1) +ggplot2::theme(panel.background = ggplot2::element_blank())


### Network Analysis of Relatiionship of NHP Features to Human Features

library(reshape2)
library(gplots)
library(RColorBrewer)

dictionary <- read.csv(file = "/Users/kpullen/autoencoder/serology_feature_dictionary.csv",header= TRUE)
scores <- read.csv(file = "/Users/kpullen/autoencoder/important_scores_primates_to_human_0.csv",header= TRUE,row.names=1)
merged <- read.csv(file = "/Users/kpullen/autoencoder/importance_Primates2Humans_protection_merged.csv",header= TRUE)
my_files <- paste0("/Users/kpullen/autoencoder/important_scores_primates_to_human_", 0:9, ".csv")

my_data <- list()
for (i in 1:10){
  temp<-read.csv(paste0("/Users/kpullen/autoencoder/important_scores_primates_to_human_", i-1, ".csv"),header=TRUE,row.names=1)
  temp$fold<-NULL
  temp$mean_score<-NULL
  temp<-temp[(merged$quality == "high"),]
  temp<-(temp/max(abs(temp)))*100
  my_data[[i]]<-temp
}

average_dataframes <- function(df_list) {
  if (length(df_list) == 0) {
    stop("Input list is empty.")
  }
  
  # Check if all dataframes have the same structure
  first_df <- df_list[[1]]
  if (!all(sapply(df_list, function(df) identical(names(df), names(first_df))))){
    stop("Dataframes in the list have different structures.")
  }
  
  # Calculate the average dataframe
  result_df <- df_list[[1]]  # Initialize result_df with the structure of the first dataframe
  
  for (i in seq_along(df_list)) {
    result_df <- result_df + df_list[[i]]
  }
  
  result_df <- result_df / length(df_list)
  
  return(result_df)
}
cellwise_standard_deviation_dataframes <- function(df_list) {
  if (length(df_list) == 0) {
    stop("Input list is empty.")
  }
  
  # Check if all data frames have the same structure
  first_df <- df_list[[1]]
  if (!all(sapply(df_list, function(df) identical(names(df), names(first_df))))){
    stop("Data frames in the list have different structures.")
  }
  
  result_df <- first_df  # Initialize result_df with the structure of the first data frame
  
  for (i in 1:length(df_list)) {
    for (row in 1:nrow(first_df)) {
      for (col in 1:ncol(first_df)) {
        result_df[row, col] <- sd(sapply(df_list, function(df) df[row, col]))
      }
    }
  }
  
  return(result_df)
}

avg_df <- average_dataframes(my_data)
sd_df <- cellwise_standard_deviation_dataframes(my_data)

sd_nhp_feat<-rowMeans(sd_df[])
low_sd_nhp_feat<-sd_nhp_feat[sd_nhp_feat<quantile(sd_nhp_feat, 0.25)]

avg_df_copy<-avg_df
avg_df_copy[abs(avg_df)<10]<-0
avg_df_copy[abs(avg_df)>10]<-1
avg_df_copy<-avg_df_copy[rowSums(avg_df_copy[])>0,]
avg_df_copy<-avg_df_copy[names(low_sd_nhp_feat),]
feats<-row.names(avg_df_copy)
avg_df_final<-avg_df[feats,]
avg_df_final<-na.omit(avg_df_final)
avg_df_final$id<-row.names(avg_df_final)
avg_df_melted<-melt(avg_df_final)

avg_df_heatmap<-avg_df_final[,1:14]
avg_df_heatmap[abs(avg_df_heatmap)<quantile(abs(avg_df_melted$value), 0.75)]<-0
heatmap.2(data.matrix(avg_df_heatmap),trace='none',cexRow=0.25,cexCol = 0.25,col=rev(brewer.pal(n = 11, name = 'RdBu')),breaks = seq(-23, 23, length.out = 12),dendrogram='none')
avg_df_melted<-avg_df_melted[avg_df_melted$value>quantile(avg_df_melted$value, 0.75),]
write.csv(avg_df_melted,"Feature_Connections_for_Network.csv") #export to Cytoscape for network development
