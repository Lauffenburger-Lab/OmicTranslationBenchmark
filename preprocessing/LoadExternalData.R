library(tidyverse)

### Load FIT training data
#data <- read.delim2('FIT Data/FIT_training_data.txt')
load(file = 'FIT Data/AllData_V2.0.rda')

# Convert to 2 data matrices with GeX
# Rownames CSP_ID and colnames the Entrez.id for
# mouse or human gene.
# Also keep original for cross validation splitting purposes
human_gex <- AllData_V2.0 %>% select(CSP_ID,HS.Entrez,EffSize.HS) %>% unique()
human_gex <- human_gex %>% spread(HS.Entrez,EffSize.HS)

mouse_gex <- AllData_V2.0 %>% select(CSP_ID,MM.Entrez,EffSize.MM) %>% unique()
mouse_gex <- mouse_gex %>% spread(MM.Entrez,EffSize.MM)

# Load GEO Brubacker data
paired_data <- read.delim2('Brubacker Data/table_paired_conditions.txt')

# Get GEO
library(GEOquery)
geo_mouse <-  getGEO(GEO = 'GSE5663', AnnotGPL = TRUE)
raw_mouse <- geo_mouse[[2]]
mouse_gene_info <-  raw_mouse@featureData@data
mouse_pheno <- raw_mouse@phenoData@data
pheno_group <- 'Group: CLP'



geo_human <-  getGEO(GEO = 'GSE28750', AnnotGPL = TRUE)
raw_human <- geo_human[[1]]
human_gene_info <-  raw_human@featureData@data
human_pheno <- raw_human@phenoData@data


