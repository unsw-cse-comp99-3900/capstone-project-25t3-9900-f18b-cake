suppressMessages({
    #library("reticulate",  quiet = TRUE)
    #library("anndata",  quiet = TRUE)
    library("limma", quiet = TRUE)
    library("arrow", quiet = TRUE)
})


#path <- .libPaths()
#path = c(path, "/data/projects/classifiers/bin/R-packages")

runDMC <- function(data){
    data <- as.data.frame(data)
    #print(data)
    data <- arrow_table(data)
    return (data)
}

runDM <- function(dataFile, designFile){
    data = read_feather(dataFile)
    designdf = read_feather(designFile)
    

    designdf <- as.data.frame(designdf)
    rownames(designdf) <- designdf$sampleId

    data <- as.data.frame(data)
    rownames(data) <- data[["sampleId"]]
    data <- data[, !names(data) %in% c("sampleId", "Name"), drop = FALSE]

    non_numeric_columns <- sapply(data, function(x) !all(is.numeric(x)))
    problematic_columns <- names(data)[non_numeric_columns]
    #print(problematic_columns)

    data=t(data)
    #print(dim(data))
    #print(dim(designdf))

    data <- as.data.frame(data, stringsAsFactors = FALSE)
    data[] <- lapply(data, as.numeric)
    

    #print(head(data))
    #data <- t(subSampleAdata$X)
    targets <- designdf$sample_id

    #print(designdf$cancerType)
    group <- factor(na.omit(designdf$cancerType))
    
    design <- model.matrix(~group)

    fit.reduced <- lmFit(data, design)
    fit.reduced <- eBayes(fit.reduced, robust = TRUE)
    
    top <- topTable(fit.reduced, coef = 2, number = 50)
    
    return(top)

}
