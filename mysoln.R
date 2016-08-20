#' # Project for Practical Machine learning course
#' ****
#' The solution follows the below steps
#' 
#' 1. Read in the input train and test data and understand it
#' 
#' 2. Clean up the data
#' 
#' 3. Do Principal component analysis and find the most important variables.
#' 
#' 4. Create cross validation datasets
#' 
#' 5. Fit two models, LDA and random forests
#' 
#' We select the model with the greater accuracy
#' 
#' ******
#' 
library(caret)
library(randomForest)


mainDir="/home/sdhandap/WeightLiftPredict"
setwd(mainDir)
source("helperfuncs.R")

#' ##STEP1: Read in the input train and test data and understand it
#' 
#'  Read in the training and test data
#'  
#'  
pmltraindata <- read.table("pml-training.csv",sep=",",header=TRUE)
pmltestdata <- read.table("pml-testing.csv",sep=",",header=TRUE)

#' 
#'  Explore the structure of the data 
#'  
#' 
preclean_explore(pmltraindata, pmltestdata)


#' 
#' ##STEP2: Clean up the data
#' 
#'
#'  start with eliminating the NA variables
#'  
cleantrain <- eliminate_NAs(pmltraindata)
cleantest <- eliminate_NAs(pmltestdata)

#' 
#' next eliminating the NULLs 
#' 
cleantrain <- eliminate_Nulls(cleantrain)
cleantest <- eliminate_Nulls(cleantest)

#' next eliminating the near zero variance variables 
#' 
cleantrain <- eliminate_zeroVarFactors(cleantrain)
cleantest <- eliminate_zeroVarFactors(cleantest)

#'
#'  Drop unnecessary varibles for prediction
#' 
answer <- cleantest["problem_id"]
drops <- c("problem_id","X")
cleantest <- cleantest[ , !(names(cleantest) %in% drops)]
cleantrain <- cleantrain[ , !(names(cleantrain) %in% drops)]
#do_some_visualisation(cleantrain)
  



#'
#' Just explore the data again after exploration
#' 

postclean_explore(cleantrain, cleantest)

#'
#' now combine the training and test data so that when we do prediction we dont get the  errors like mismatch of type of predictors
#' ("Type of predictors in new data do not match that of the training data.")  
#' 
 
testnumrows <- nrow(cleantest)
cleantest[,"classe"] <- NA
combinedData <- rbind(cleantrain,cleantest)
allrows <- nrow(combinedData)
finaltest <- combinedData[(allrows-testnumrows+1):allrows, ]
trainingset <- combinedData[1:(allrows-testnumrows), ]


#'
#' We are done with the data clean up stage
#' 

#' 
#' ##STEP3: Do Principal component analysis
#' 
#' 

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


#'
#' Now we are done with the principal component analysis.
#' Next step is to split the data for cross validatiopn
#' do the  simple train test and validation split
#' 


#' 
#' ##STEP4: Create cross validation datasets
#' 
#' 
training.rows <- createDataPartition(train.pca.data$classe,  p = 0.8, list = FALSE)


train.batch <- train.pca.data[training.rows, ]
test.batch <- train.pca.data[-training.rows, ]



#'
#' Now we have the data split into three: the train.batch, test.batch and the finaltest dataset
#' 
#' 

#' 
#' ##STEP5: Model creation and Validation
#' 
#' 



dependentvarname <- "classe"
AllVariables <- names(train.pca.data)
PredictorVariables <- setdiff(AllVariables, dependentvarname)
Formula <- formula(paste( paste(dependentvarname, " ~ ", sep =""), 
                            paste(PredictorVariables, collapse=" + ")))
print(Formula)

#'
#' Fit the random forest model
#' 
#' 
rf_fit <- randomForest(Formula,
                       data=train.batch, 
                       importance=TRUE, 
                       ntree=2000)
pred.rf <- predict(rf_fit, test.batch)
confusionMatrix(pred.rf, test.batch$classe)
accuracy.rf <- (round(mean(pred.rf == test.batch$classe),3))

#'
#' The accuracy in the random forest model
#' 
#' 
print(accuracy.rf)

#'
#' Fit the Linear discriminant analysis model
#' 
#' 

model_lda <- train(Formula, method = "lda", data = train.batch)
pred.lda <- predict(model_lda, test.batch)
confusionMatrix(pred.lda, test.batch$classe)
accuracy.lda <-(round(mean(pred.lda == test.batch$classe),3))

#'
#' The accuracy in the LDA model is
#' 
#' 
print(accuracy.lda)



if(accuracy.rf >= accuracy.lda) {

  #'
  #' Select Random Forest model for the final prediction
  #' 
  #' 
  pred.final <- predict(rf_fit, finaltest)
  
  
}else {
  #'
  #' Select LDA model for the final prediction
  #' 
  #' 
  pred.final <- predict(model_lda, finaltest)
}
#'
#' This is the final prediction output.
#' 
#' 

print(pred.final)
  