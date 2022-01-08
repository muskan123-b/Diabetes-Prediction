#Setting the working directory
setwd("C:/Users/muska/Desktop/Minor Poject")

#Importing the dataset
df <- read.csv("C:/Users/muska/Desktop/Minor Poject/diabetes.csv")
View(df)

#Importing Libraries
library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(caTools)
library(e1071)
library (caret)
#Data Exploration
nrow(df) #No. of Rows
ncol(df) #No. of Columns

head(df)

str(df)
summary(df)

View(data.frame(sapply(df,class))) #To find Data Types of columns

colSums(is.na(df))

df$Outcome<-as.factor(df$Outcome)
#Plotting Histogram
par(mfrow=c(2, 3))
hist(df$Pregnancies, breaks = 10, col = "coral2", main = "No. of Pregnancies", xlab = "Pregnancies")
hist(df$Glucose, breaks = 5, col = "gold1", main = "Glucose", xlab = "Glucose")
hist(df$BloodPressure, breaks = 5, col = "light green", main = "Blood Pressure", xlab = "Blood Pressure")
hist(df$SkinThickness, breaks = 10, col = "sky blue", main = "Skin Thickness", xlab = "Skin Thickness")
hist(df$Insulin, breaks = 10, col = "orange", main = "Insulin", xlab = "Insulin")
hist(df$Age, breaks = 10, col = "pink", main = "Age", xlab = "Age")


#Plotting Correlation Matrix
cor(df)

require(ggcorrplot)
corr <- round(cor(df), 1) 
ggcorrplot(corr,
           type = "lower",
           lab = TRUE, 
           lab_size = 5,  
           colors = c("red", "white", "cyan4"),
           title="Correlogram of Housing Dataset", 
           ggtheme=theme_bw)

par(mfrow=c(2,3))
boxplot(df$Pregnancies~df$Outcome, main="No. of Pregnancies vs. Diabetes", xlab="Outcome", ylab="Pregnancies")
boxplot(df$Glucose~df$Outcome, main="Glucose vs. Diabetes", xlab="Outcome", ylab="Glucose")
boxplot(df$SkinThickness~df$Outcome, main="Skin Thickness vs. Diabetes", xlab="Outcome", ylab="Skin Thickness")
boxplot(df$BMI~df$Outcome, main="BMI vs. Diabetes", xlab="Outcome", ylab="BMI")
boxplot(df$DiabetesPedigreeFunction~df$Outcome, main="Diabetes Pedigree Function vs. Diabetes", xlab="Outcome", ylab="DiabetesPedigreeFunction")
boxplot(df$Age~df$Outcome, main="Age vs. Diabetes", xlab="Outcome", ylab="Age")


outlier_treat <- function(x){
        UC = quantile(x, p=0.99,na.rm=T)
        LC = quantile(x, p=0.01,na.rm=T)
        x=ifelse(x>UC,UC, x)
        x=ifelse(x<LC,LC, x)
        return(x)
}
df = data.frame(apply(df, 2, FUN=outlier_treat))

df$Outcome<-as.factor(df$Outcome)

# Splitting the dataset into the Training set and Test set
require(caTools)
sample.split(df$Outcome, SplitRatio = 0.75)->split_index
training_set <- subset(df,split_index == TRUE)
test_set <- subset(df, split_index == FALSE)
nrow(test_set)
nrow(training_set)


model <- glm(Outcome~.-SkinThickness-Insulin, data = training_set,family = "binomial")
summary(model)

### Coming Up with the Predicted Probabilities

train<- cbind(training_set, Prob=predict(model, type="response")) 
View(train)


####################################
#  MODEL EVALUATION and VALIDATION #
####################################


require(Metrics)
require(InformationValue)

#find optimal cutoff probability to use to maximize accuracy
optimal <- optimalCutoff(train$Outcome, train$Prob)
optimal



# Concordance
Concordance(train$Outcome,train$Prob)


# Classification table - train dataset
traintable <- table(Predicted = train$Prob>0.4961266, Actual = train$Outcome)
traintable

# Accuracy of the model - train dataset 
accuracy.train <- round(sum(diag(traintable))/sum(traintable),2)
sprintf("Accuracy is %s",accuracy.train)

# Precision and Recall of the model - train dataset 
precision.train <- traintable [2,2]/sum(traintable [2,])
recall.train <- traintable [2,2]/sum(traintable [,2])
sprintf("Precision is %s",accuracy.train)
sprintf("Recall is %s",accuracy.train)


#4. How to find the threshold value


library(ROCR)

ROCRPred=prediction(train$Prob,train$Outcome)

ROCRPerf<-performance(ROCRPred,"tpr","fpr")

plot(ROCRPerf,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))

############################
#  VALIDATION OF THE MODEL #
############################

test <- cbind(test_set, Prob=predict(model, test_set, type = "response"))
View(test)
test$Probability<- ifelse(test$Prob>0.5994366,"1","0")

# Coming up with predcited class using threshold @0.5

# Classification table - train dataset
testtable <- table(Predicted = test$Prob>0.5994366, Actual = test$Outcome)
testtable

# Accuracy of the model - train dataset 
accuracy.test <- round(sum(diag(testtable))/sum(testtable),2)
sprintf("Accuracy is %s",accuracy.test)

# Precision and Recall of the model - train dataset 
precision.test <- testtable [2,2]/sum(testtable [2,])
recall.test <- testtable [2,2]/sum(testtable [,2])
sprintf("Precision is %s",accuracy.test)
sprintf("Recall is %s",accuracy.test)


test$Probability<- ifelse(test$Prob>0.5994366,"1","0")
library(ggplot2)
ggplot(test, aes(x=Outcome, y=Probability)) + geom_point() + 
        stat_smooth(method="glm", method.args=list(family="binomial"), se=FALSE)

par(mar = c(4, 4, 1, 1)) # Reduce some of the margins so that the plot fits better
plot(test$Outcome, test$Probability)
curve(predict(model, data.frame(Probability=x), type="response"), add=TRUE) 


ggplot(test, aes(x=Outcome, y=Prob)) + 
        geom_point(alpha=.5) +
        stat_smooth(method="glm", se=FALSE, method.args = list(family=binomial))


predicted.data <- data.frame(prob.of.diabetes=test, Outcome = test$Outcome)
view
predicted.data <- predicted.data[order(predicted.data$prob.of.diabetes, decreasing = F),]
predicted.data$rank <- 1:nrow(predicted.data) 

ggplot(predicted.data, aes(x=rank, y=prob.of.diabetes)) + 
        geom_point(aes(color=Outcome), alpha=1, shape=3, stroke=2) +
        xlab("Index") +
        ylab("Predicted probablity of diabetes")


library(rpart)
library(rpart.plot)
par(mfrow=c(1, 1))
model2 <- rpart(Outcome ~ Pregnancies + Glucose + BMI + DiabetesPedigreeFunction, data=training_set,method="class")

plot(model2, uniform=TRUE, 
     main="Classification Tree for Diabetes")
text(model2, use.n=TRUE, all=TRUE, cex=.8)

rpart.plot(x = model2, yesno = 2, type = 0, extra = 0)


# class prediction
class_predicted <- predict(object = model2,  
                           newdata = test_set,   
                           type = "class")

# Generate a confusion matrix for the test data
confusionMatrix(class_predicted,       
                test_set$Outcome)

Accuracy <- accuracy(test_Y,class_predicted)
Accuracy

# Senstivity
Senstivity = InformationValue::sensitivity(test_Y,pred_mod_log3)
Senstivity

# specificity
specificity = InformationValue::specificity(test_Y,pred_mod_log3)
specificity