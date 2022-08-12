### This script checks if a library is installed 
### and installs it if not.
### It will not install the specific versions used in the study
## but the latest versions of each library.

### Inform user
print('Keep in mind this may change/update your own setup')

# First install tidyverse
if (!require("tidyverse", quietly = TRUE)){
  install.packages("tidyverse")
}

# Install ggplot2 and ggpubr for visualization
if (!require("ggplot2", quietly = TRUE)){
  install.packages("ggplot2")
  
}

if (!require("ggpubr", quietly = TRUE)){
  install.packages("ggpubr")
  
}

# Install the Bioconductor resource.
# This is used to install multiply other
# resources.
if (!require("BiocManager", quietly = TRUE)){
  install.packages("BiocManager")
}

# Install many important packages of bioconductor
# It is recommended because it installs a lot of
# dependencies and core libraries
# for the next packages.
BiocManager::install()

# Install the whole bioconductor 
# Install all bioconductor packages 
# Even without checking if already installed,
# they will only be updated if out of date.
BiocManager::install(c("cmapR",
                       "org.Hs.eg.db",
                       "rhdf5",
                       "GeneExpressionSignature"))

# Install doFuture and dorng for parallel processing
if (!require("doFuture", quietly = TRUE)){
  install.packages('doFuture')
}

if (!require("doRNG", quietly = TRUE)){
  install.packages('doRNG')
  
}

#devtools::install_github("saezlab/CARNIVAL"))
