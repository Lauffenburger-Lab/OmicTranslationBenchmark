library(tidyverse)
library(rhdf5)
library(biomaRt)

# Define input file for mouse and human to parse them
inputFile_human = 'data/archs4_gene_human_v2.1.2.h5'
inputFile_mouse = 'data/archs4_gene_mouse_v2.1.2.h5'

# Retrieve information from compressed data
# samples = h5read(inputFile_human, "meta/samples/geo_accession")
# genes = h5read(inputFile_human, "meta/genes")
# meta <- h5read(inputFile_human, "meta")
# meta <- data.frame(meta[['samples']])
# meta <- meta %>% filter(molecule_ch1=='total RNA')
# meta <- meta %>% filter(library_source=='transcriptomic')
# platoforms_to_keep <- c("Illumina HiSeq 4000","Illumina HiSeq 3000","Illumina NovaSeq 6000")
# meta <- meta %>% filter(instrument_model %in% platoforms_to_keep)
# meta <- meta %>% filter(singlecellprobability<0.5)
# sample_locations = which(samples %in% meta$geo_accession)

destination_file = "archs4_gene_human_v2.1.2.h5"
extracted_expression_file = "human_immune_cells_archs4_gene_human_v212.tsv"
url = "https://s3.dev.maayanlab.cloud/archs4/archs4_gene_human_v2.1.2.h5"

# destination_file = "archs4_gene_mouse_v2.1.2.h5"
# extracted_expression_file = "mouse_immune_cells_archs4_gene_mouse_v212.tsv"
# url = "https://s3.dev.maayanlab.cloud/archs4/archs4_gene_mouse_v2.1.2.h5"

# Check if gene expression file was already downloaded, if not in current directory download file form repository
if(!file.exists(destination_file)){
  print("Downloading compressed gene expression matrix.")
  download.file(url, destination_file, quiet = FALSE, mode = 'wb')
}
# Retrieve information from compressed data
samples = h5read(destination_file, "meta/samples/geo_accession")
genes = h5read(destination_file, "meta/genes/gene_symbol")

# human_cell_lines <- list.files('data/human_cell_lines/')
human_cell_immune  <- list.files('data/human_immune_cells/')
samp <- c()
for (file in human_cell_immune){
  samp <- unique(c(samp,readRDS(paste0('data/human_immune_cells/',file))))
}

# # mouse_cell_lines <- list.files('data/mouse_cell_lines/')
# mouse_cell_immune  <- list.files('data/mouse_immune_cells/')
# samp <- c()
# for (file in mouse_cell_immune){
#   samp <- unique(c(samp,readRDS(paste0('data/mouse_immune_cells/',file))))
# }
# Identify columns to be extracted
sample_locations = which(samples %in% samp)

# Load gene info
## uSE biomart to find protein-coding genes only
head(listMarts(), 10)
head(listDatasets(useMart("ensembl")), 3)
ensembl <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")
head(listFilters(ensembl), 3)
myFilter <- "hgnc_symbol"
myAttributes <- c("hgnc_symbol","description","gene_biotype")
## assemble and query the mart
res <- getBM(attributes =  myAttributes, filters =  myFilter,
             values = genes, mart = ensembl)
res <- res %>% filter(gene_biotype=='protein_coding')
human_genes <- which(genes %in% unique(res$hgnc_symbol))
# ensembl <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")
# head(listFilters(ensembl), 3)
# myFilter <- "mgi_symbol"
# myAttributes <- c("mgi_symbol","description","gene_biotype")
# ## assemble and query the mart
# res <- getBM(attributes =  myAttributes, filters =  myFilter,
#              values = genes, mart = ensembl)
# res <- res %>% filter(gene_biotype=='protein_coding')
# mouse_genes <- which(genes %in% unique(res$mgi_symbol))

# extract gene expression from compressed data
expression = h5read(destination_file, "data/expression", index=list(sample_locations, human_genes))
# expression = h5read(destination_file, "data/expression", index=list(sample_locations, mouse_genes))
H5close()
colnames(expression) = genes[human_genes]
rownames(expression) = samples[sample_locations]

# Print file
write.table(expression, file=extracted_expression_file, sep="\t", quote=FALSE, col.names=NA)
print(paste0("Expression file was created at ", getwd(), "/", extracted_expression_file))

# Load metadata for these samples
meta <- h5read(destination_file, "meta")
meta <- data.frame(meta[['samples']])
meta <- meta %>% filter(geo_accession %in% samples[sample_locations])
write.table(meta, file='human_immune_cells_metadata.tsv', sep="\t", quote=FALSE, col.names=NA)
print(paste0("Expression file was created at ", getwd(), "/human_immune_cells_metadata.tsv"))
