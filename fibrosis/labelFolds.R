library(tidyverse)
library(reshape2)
library(ggsignif)
library(ggpubr)

### Assign labels and specific cell-types in validation data-------------
alldata <-data.table::fread('../../../Fibrosis Species Translation/human lung fibrosis/all_mouse_lung.csv')
allmeatadata <- data.table::fread('../../../Fibrosis Species Translation/human lung fibrosis/Strunz_2020_meta.csv')
allmeatadata <- allmeatadata %>% select(V1,c('specific_cell'='cell.type')) %>% unique()
alldata <- left_join(alldata,allmeatadata)
colnames(alldata)[(ncol(alldata)-3):(ncol(alldata)-1)] <- c("diagnosis","species","cell_type")
alldata <- alldata %>% select(-species)

# save cells agrregated data
cells <- alldata %>% select(specific_cell,cell_type) %>% group_by(specific_cell) %>% mutate(cell_counts= n()) %>% ungroup() %>% unique()
cells <- cells %>% mutate(cell_type=ifelse(cell_type==1,'immune',
                                           ifelse(cell_type==2,'mesenchymal',
                                                  ifelse(cell_type==3,'epithelial',
                                                         ifelse(cell_type==4,'endothelial','stem cell')))))
data.table::fwrite(cells,'data/mouse_cells_info.csv')

for (i in 1:9){
  ## Load validation fold
  val <- data.table::fread(paste0('data/10foldcrossval_lung/csvFiles/val_mouse_',i,'.csv'))
  val <- val %>% select(-V1)
  val <- val %>% select(-species)
  colnames(val)[1:(ncol(val)-2)] <- colnames(alldata)[2:(ncol(alldata)-3)]
  val <- left_join(val,alldata) %>% column_to_rownames('V1')
  val <- val %>% filter(!is.na(specific_cell))
  data.table::fwrite(val,paste0('data/10foldcrossval_lung/csvFiles/labeled_val_mouse_',i,'.csv'))
  print(paste0('Finished fold ',i))
}