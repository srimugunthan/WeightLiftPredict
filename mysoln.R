library(caret)
library(randomForest)

mainDir="/home/sdhandap/WeightLiftPredict"

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

answer <- cleantest["problem_id"]
drops <- c("problem_id","X")
cleantest <- cleantest[ , !(names(cleantest) %in% drops)]
cleantrain <- cleantrain[ , !(names(cleantrain) %in% drops)]
#do_some_visualisation(cleantrain)
  




postclean_explore(cleantrain, cleantest)

# now combine the training and test data so that when we do prediction we dont get the  error 
# ("Type of predictors in new data do not match that of the training data.")  
# 
# 
testnumrows <- nrow(cleantest)
cleantest[,"classe"] <- NA
combinedData <- rbind(cleantrain,cleantest)
allrows <- nrow(combinedData)
finaltest <- combinedData[(allrows-testnumrows+1):allrows, ]
trainingset <- combinedData[1:(allrows-testnumrows), ]

## do PCA 

nonNumericVars <- c("user_name","classe","cvtd_timestamp")
pcadata <- combinedData[ , !(names(combinedData) %in% nonNumericVars)]
pca <- prcomp(pcadata, scale = TRUE)
biplot(pca, scale = 0)
std_dev <- pca$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)

plot(prop_varex, xlab = "Principal Component",
           ylab = "Proportion of Variance Explained",
           type = "b")

plot(cumsum(prop_varex), xlab = "Principal Component",
           ylab = "Cumulative Proportion of Variance Explained",
           type = "b")


selectcols = which(prop_varex >= 0.002)
dataAfterPCA <- pcadata[,selectcols]
pr_cols <- names(dataAfterPCA)
pr_cols <- c(pr_cols, "classe")



train.pca.data <- trainingset[,(names(trainingset) %in% pr_cols)]
test.pca.data <- finaltest[,(names(finaltest) %in% pr_cols)]

# cross validation. First do simple train test and validation split
training.rows <- createDataPartition(train.pca.data$classe,  p = 0.8, list = FALSE)


train.batch <- train.pca.data[training.rows, ]
test.batch <- train.pca.data[-training.rows, ]
#k <- 5 # k-fold cross-validation
#folds <- createFolds(y = trainingset$classe, k = k, list = TRUE, returnTrain = TRUE)




dependentvarname <- "classe"
AllVariables <- names(train.pca.data)
PredictorVariables <- setdiff(AllVariables, dependentvarname)
Formula <- formula(paste( paste(dependentvarname, " ~ ", sep =""), 
                            paste(PredictorVariables, collapse=" + ")))
print(Formula)


rf_fit <- randomForest(Formula,
                       data=train.batch, 
                       importance=TRUE, 
                       ntree=2000)
pred.rf <- predict(rf_fit, test.batch)
confusionMatrix(pred.rf, test.batch$classe)
accuracy.rf <- (round(mean(pred.rf == test.batch$classe),3))
print(accuracy.rf)

model_lda <- train(Formula, method = "lda", data = train.batch)
pred.lda <- predict(model_lda, test.batch)
confusionMatrix(pred.lda, test.batch$classe)
accuracy.lda <-(round(mean(pred.lda == test.batch$classe),3))

print(accuracy.lda)



if(accuracy.rf >= accuracy.lda) {

  pred.final <- predict(rf_fit, finaltest)
  
  
}else {
  pred.final <- predict(model_lda, finaltest)
}

print(pred.final)
  







