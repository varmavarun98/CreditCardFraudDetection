library(tidyverse)
library(ggplot2)
library(glmnet)
library(corrplot)


set.seed(12345)
creditdata <- read.csv("creditcard.csv")

#-------------------------------------EDA----------------------------------------------------

str(creditdata)
summary(creditdata)


creditdata$Time <- scale(creditdata$Time)
creditdata$Amount <- scale(creditdata$Amount)
summary(creditdata)

par(mfrow = c(3,5))
i <- 1
for (i in 1:30) 
{
  hist((creditdata[,i]), main = paste("Distibution of ", colnames(creditdata[i])), xlab = colnames(creditdata[i]), col = "light blue")
}

par(mfrow = c(1,1))
barplot(table(creditdata$Class), main = "Frequency of Fraud", col = "light blue")
table(creditdata$Class)

cor(creditdata)
M <- cor(creditdata)
corrplot(M, method = "circle")
round(M, 2)

#-------------------------SMOTE: For Making Data Balanced-----------------------

library(smotefamily)
library(DMwR)

table(creditdata$Class)
## now using SMOTE to create a more "balanced problem"
newData <- SMOTE(Class ~ ., creditdata, perc.over = 6000,perc.under=100)
barplot(table(newData$Class), main = "Frequency of Fraud- Updated", col = "light blue")
table(newData$Class)

index <- sample(nrow(newData),nrow(newData)*.7)
credit.train <- newData[index,]
credit.test <- newData[-index,]



#----------------------------KNN------------------------------------------------

library(class)
#In Sample
knn_credit <- knn(train = credit.train[, -31], test = credit.train[, -31], cl=credit.train[,31], k=5)
table(credit.train[,31], knn_credit, dnn = c("True", "Predicted"))
MR<- mean(credit.train[,31] != knn_credit)
1-MR

#Out of Sample
knn_credit <- knn(train = credit.train[, -31], test = credit.test[, -31], cl=credit.train[,31], k=5)
table(credit.test[,31], knn_credit, dnn = c("True", "Predicted"))
MR<- mean(credit.test[,31] != knn_credit)
1-MR


#----------------Logistic Regression--------------------------------------------

null_model <- glm(Class~1, data = credit.train, family = binomial)
full_model<- glm(Class~., family=binomial, data=credit.train)

#AIC
credit.glm.step.AIC <- step(null_model,scope = list(lower = null_model, upper = full_model), direction= "forward")
summary(credit.glm.step.AIC)

#BIC
credit.glm.step.BIC <- step(null_model,scope = list(lower = null_model, upper = full_model), direction = "forward", k= log(nrow(credit.train)))
summary(credit.glm.step.BIC)

#LASSO
credit.lasso.cv<- cv.glmnet(x=as.matrix(credit.train[,-c(31)]), y=as.matrix(credit.train$Class), 
                            family = "binomial", type.measure = "class",  alpha=1, nfolds=10)
plot(credit.lasso.cv)

credit.lasso<- glmnet(x=as.matrix(credit.train[,-c(31)]), y=as.matrix(credit.train$Class), 
                      family = "binomial",)
coef(credit.lasso, s=credit.lasso.cv$lambda.1se)
credit.glm.lasso <- glm(Class~Time+V1+V2+V4+V5+V6+V8+V9+V10+V11+V12+V13+V14+V15+V16+V20+V21+
                          V22+V23+V24+V25+V26+V28+Amount, family=binomial, data=credit.train)
summary(credit.glm.lasso)

#Final Model
credit.glm <- credit.glm.step.AIC

# define a cost function with input "obs" being observed response 
# and "pi" being predicted probability, and "pcut" being the threshold.
costfunc = function(obs, pred.p, pcut){
  weight1 = 10   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function


# write a loop for all p-cut to see which one provides the smallest cost
# first, need to define a 0 vector in order to save the value of cost from all pcut
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = credit.train$Class, pred.p = pred.glm0.train, pcut = p.seq[i])  
} # end of the loop


# define a sequence from 0.01 to 1 by 0.01
p.seq = seq(0.01, 1, 0.01) 
# draw a plot with X axis being all pcut and Y axis being associated cost
plot(p.seq, cost)

optimal.pcut.glm0 = p.seq[which(cost==min(cost))]
optimal.pcut.glm0

#In Sample- AUC and Error
library(ROCR)
pred.glm0.train<- predict(credit.glm, type="response")
pred <- prediction(pred.glm0.train, credit.train$Class)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

class.glm0.train.opt<- (pred.glm0.train>optimal.pcut.glm0)*1
table(credit.train$Class, class.glm0.train.opt, dnn = c("True", "Predicted"))
MR<- mean(credit.train$Class!= class.glm0.train.opt)
1-MR

#Out of Sample - AUC and Error

pred.glm0.test<- predict(credit.glm, newdata = credit.test, type="response")
pred <- prediction(pred.glm0.test, credit.test$Class)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

class.glm0.test.opt<- (pred.glm0.test>optimal.pcut.glm0)*1
table(credit.test$Class, class.glm0.test.opt, dnn = c("True", "Predicted"))
MR<- mean(credit.test$Class!= class.glm0.test.opt)
1-MR


#--------------------------Decision Tree----------------------------------------

library(rpart)
library(rpart.plot)
credit.tree <- rpart(formula = Class ~ ., data = credit.train, cp = 0.001,
                          method = "class", parms = list(loss=matrix(c(0,10,1,0), nrow = 2)))
plotcp(credit.tree)
printcp(credit.tree)
prp(credit.tree, extra = 1)


#InSample- AUC and Error

credit.train.prob.rpart = predict(credit.tree,credit.train, type="prob")
pred = prediction(credit.train.prob.rpart[,2], credit.train$Class)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
slot(performance(pred, "auc"), "y.values")[[1]]

credit.train.pred.rpart = as.numeric(credit.train.prob.rpart[,2] > 1/11)
table(credit.train$Class, credit.train.pred.rpart, dnn=c("Truth","Predicted"))
MR <- mean(ifelse(credit.train$Class != credit.train.pred.rpart, 1, 0))
1-MR


#Out of Sample- AUC and Error

credit.test.prob.rpart = predict(credit.tree,credit.test, type="prob")
pred = prediction(credit.test.prob.rpart[,2], credit.test$Class)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
slot(performance(pred, "auc"), "y.values")[[1]]


credit.test.pred.rpart = as.numeric(credit.test.prob.rpart[,2] > 1/11)
table(credit.test$Class, credit.test.pred.rpart, dnn=c("Truth","Predicted"))
MR <- mean(ifelse(credit.test$Class != credit.test.pred.rpart, 1, 0))
1-MR


#---------------------------RF--------------------------------------------------

library(randomForest)
credit.rf <- randomForest((credit.train$Class)~., data = credit.train)
credit.rf


credit.rf.pred<- predict(credit.rf, type = "prob")[,2]

costfunc = function(obs, pred.p, pcut){
  weight1 = 10   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} 
p.seq = seq(0.01, 0.5, 0.01)
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = credit.train$Class, pred.p = credit.rf.pred, pcut = p.seq[i])  
}
plot(p.seq, cost)


## Optimal Cutoff
optimal.pcut= p.seq[which(cost==min(cost))]
optimal.pcut

#Insample - AUC and Error
pred <- prediction(credit.rf.pred, credit.train$Class)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

credit.rf.pred.train<- predict(credit.rf, newdata=credit.train, type = "prob")[,2]
credit.rf.class.train<- (credit.rf.pred.train>optimal.pcut)*1
table(credit.train$Class, credit.rf.class.train, dnn = c("True", "Pred"))
MR<- mean(credit.train$Class!= credit.rf.class.train)
1-MR

#Out of Sample
pred <- prediction(credit.rf.pred, credit.test$Class)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

credit.rf.pred.test<- predict(credit.rf, newdata=credit.test, type = "prob")[,2]
credit.rf.class.test<- (credit.rf.pred.test>optimal.pcut)*1
table(credit.test$Class, credit.rf.class.test, dnn = c("True", "Pred"))
MR<- mean(credit.test$Class!= credit.rf.class.test)
1-MR


#----------------------NN-------------------------------------------------------

library(nnet)
credit.nnet <- nnet(Class~., data=credit.train, size=1, maxit=500)

#InSample
prob.nnet= predict(credit.nnet,credit.train)
pred.nnet = as.numeric(prob.nnet > 1/11)
table(credit.train$Class,pred.nnet, dnn=c("Observed","Predicted"))
MR <- mean(ifelse(credit.train$Class != pred.nnet, 1, 0))
1-MR

#Out of Sample
prob.nnet= predict(credit.nnet,credit.test)
pred.nnet = as.numeric(prob.nnet > 1/11)
table(credit.test$Class,pred.nnet, dnn=c("Observed","Predicted"))
MR <- mean(ifelse(credit.test$Class != pred.nnet, 1, 0))
1-MR
