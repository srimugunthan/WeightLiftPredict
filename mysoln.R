library(caret)
library(randomForest)

mainDir="/home/srimugunthan/Dropbox/HopkinsProject"

preclean_explore <- function(traindata,testdata)
{
 #sink(file = "preclean-exploration.txt")
  cat("=============================\n")
  str(traindata)
  cat("\n")
  str(testdata)
  
  
  
  trainfeatures <- names(traindata)
  testfeatures <- names(testdata)
  

  cat("\nFeatures in traindata that is not in testdata:")
  
  print(trainfeatures[which(trainfeatures != testfeatures)])
  
  
  cat("\nFeatures in testdata that is not in traindata:")
  print(testfeatures[which(trainfeatures != testfeatures)])
  
  #sink(NULL)
  
}

postclean_explore <- function(traindata,testdata)
{
  # str(train, test)
  # summary(train, test)
  # feature plot
  # classe with every variable
  
  #sink(file = "preclean-exploration.txt")
  cat("=============================\n")
  str(traindata)
  cat("\n")
  str(testdata)
  
  
  cat("=============================\n")
  summary(traindata)
  cat("=============================\n")
  summary(testdata)
  
  
  
  
  trainfeatures <- names(traindata)
  testfeatures <- names(testdata)
  
  
  
  cat("\nFeatures in traindata that is not in testdata:")
  
  print(trainfeatures[which(trainfeatures != testfeatures)])
  
  
  cat("\nFeatures in testdata that is not in traindata:")
  print(testfeatures[which(trainfeatures != testfeatures)])
  
  #sink(NULL)
  
  
}

eliminate_NAs <- function(tdata)
{

  #if 80% of a feature variable is NA , eliminate it 
  
  features <- names(tdata)
  cleant <- tdata
  for (f in features)
  {
  
    nasum <- sum(is.na(tdata[f]))
    if (nasum > 0.2*nrow(tdata[f]))
    {
     # remove it
      colnum = which( colnames(cleant)==f )
      cleant <- cleant[-colnum]
    }
                 
  }
    
  return (cleant)
  
}


eliminate_Nulls <- function(tdata)
{
  
  #if 80% of a feature variable is Null , eliminate it 
  
  features <- names(tdata)
  cleant <- tdata
  for (f in features)
  {
    
    nasum <- sum(tdata[f] == "")
    if (nasum > 0.2*nrow(tdata[f]))
    {
      # remove it
      colnum = which( colnames(cleant)==f )
      cleant <- cleant[-colnum]
    }
    
  }
  
  return (cleant)
  
}

eliminate_zeroVarFactors <- function(tdata)
{
  nzv <- nearZeroVar(tdata)
  filteredtrain <- tdata[, -nzv]
  return (filteredtrain)
  
}

printmodel_diagnostics <- function(modelObj, predicted, expected)
{
  
  print(modelObj)
  print("======out of sample error:======")
  print(sum(predicted != expected)/length(expected))
  print(confusionMatrix(predicted, expected))
}

#**************************
#return the rules of a tree
#**************************
getConds<-function(tree){
  #store all conditions into a list
  conds<-list()
  #start by the terminal nodes and find previous conditions
  id.leafs<-which(tree$status==-1)
  j<-0
  for(i in id.leafs){
    j<-j+1
    prevConds<-prevCond(tree,i)
    conds[[j]]<-prevConds$cond
    while(prevConds$id>1){
      prevConds<-prevCond(tree,prevConds$id)
      conds[[j]]<-paste(conds[[j]]," & ",prevConds$cond)
      if(prevConds$id==1){
        conds[[j]]<-paste(conds[[j]]," => ",tree$prediction[i])
        break()
      }
    }
    
  }
  
  return(conds)
}

#**************************
#find the previous conditions in the tree
#**************************
prevCond<-function(tree,i){
  if(i %in% tree$right_daughter){
    id<-which(tree$right_daughter==i)
    cond<-paste(tree$split_var[id],">",tree$split_point[id])
  }
  if(i %in% tree$left_daughter){
    id<-which(tree$left_daughter==i)
    cond<-paste(tree$split_var[id],"<",tree$split_point[id])
  }
  
  return(list(cond=cond,id=id))
}

#remove spaces in a word
collapse<-function(x){
  x<-sub(" ","_",x)
  
  return(x)
}




rforestmodel <- function(train, test) {
  subDir="rforestmodel"  
  dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
  setwd(file.path(mainDir, subDir))
  
  
  
  set.seed(33833)
  model <- randomForest(classe ~ ., train)
  pred_training <- predict(model, train)
  
  printmodel_diagnostics(rforestmodel, pred_training, train$classe)
  
  tree <- getTree(model,1,labelVar=TRUE)
  
  #rename the name of the column
  colnames(tree)<-sapply(colnames(tree),collapse)
  rules<-getConds(tree)
  print(rules)
  
  
}

do_some_visualisation <- function(tdata)
{
  #pdf("mygraph.pdf", width=7, height=7)
  featurePlot(x=tdata[,1:7], y=tdata$classe, plot="pairs")
  #dev.off()
}
#setwd("/home/srimugunthan/Dropbox/HopkinsProject")
setwd(mainDir)
pmltraindata <- read.table("pml-training.csv",sep=",",header=TRUE)
pmltestdata <- read.table("pml-testing.csv",sep=",",header=TRUE)

preclean_explore(pmltraindata, pmltestdata)

cleantrain <- eliminate_NAs(pmltraindata)
cleantest <- eliminate_NAs(pmltestdata)

cleantrain <- eliminate_Nulls(cleantrain)
cleantest <- eliminate_Nulls(cleantest)

cleantrain <- eliminate_zeroVarFactors(cleantrain)
cleantest <- eliminate_zeroVarFactors(cleantest)

#do_some_visualisation(cleantrain)
  
postclean_explore(cleantrain, cleantest)

# attach  the classe variable to clean train.
cleanTrainFinal <- cleantrain
cleanTestFinal <- cleantest
rforestmodel(cleanTrainFinal, cleanTestFinal)
