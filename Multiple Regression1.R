#===========================================
#Calling Packages 
#===========================================
library(readr)
library(caret)
library(corrplot)
#===========================================
#Importing Training Data Set
#===========================================

existingproduct <- read_csv("C:/Users/admin/Desktop/Task 3 - Multiple Regression/existingproductattributes2017.csv")
View(existingproduct)

#===========================================
#UNDERSTANDING THE DATA
#===========================================

#Look into data set:
summary(existingproduct)

#Data have NA'S?
sum(is.na(existingproduct)) #<- Yes, 15

#Structure of Data 
str(existingproduct) #<- all NUM but Product Type which is Chr
## Cannot do regression analysis on non numeric variables## 
##When a researcher wishes to include a categorical variable, 
##Supplementary steps are required to make the results interpretable

#===========================================
#PREPROCESSING THE DATA
#===========================================

# dummify the data <- converts categorical variables (factor and character)
# to binary variables using the below process
newDataFrame <- dummyVars(" ~ .", data = existingproduct)
readyData <- data.frame(predict(newDataFrame, newdata = existingproduct))
readyData ##<- Remove the column Product Type and create as much columns as types of
#data their are, and put 1 if the product corresponds to these type of data, and 0 if
##not.In General, a catergorical variable with n levels will be transformed into n-1
## variables with two levels each. These n-1 new variables contain the same information
##than the single variable. This recording creates a table called the contrast matrix.

##Export data just to verify
write.table(readyData, file = "Dummy File.csv", sep=",")
##Notice product type is merged with each level 

##Column Names 
names(readyData)

#structure of new data frame
str(readyData) #<- all

##summary of data 
summary(readyData)

#delete all columns with missing data
readyData$BestSellersRank <- NULL

#Names your attributes within your new data set.
names(readyData) 

#Data have NA'S?
sum(is.na(readyData)) #<- No, column was dropped! 

#===========================================
#Feature Selection: 
#===========================================

#correlation Data <- Find the correlation between relevant independant 
##variables and the dependant variable. 
corrData <- cor(readyData)
corrData

##correlation values fall within -1 and 1. Variables with strong positive correlation 
##are closer to 1. Conversely variables with strong negative correlation are closer 
##-1. Variables close to 0 have no correlation. 

#Install corrplot package
install.packages("corrplot")

#visualize correlation
corrplot(corrData)

##Verifying correlation 
round(cor(readyData$Volume, readyData$x5StarReviews),2) #<- perfectly correlated. 
round(cor(readyData$Volume, readyData$x4StarReviews),2) #<- .88
round(cor(readyData$Volume, readyData$x3StarReviews),2) #<- .76
round(cor(readyData$Volume, readyData$PositiveServiceReview),2) #<- .62
round(cor(readyData$Volume, readyData$x2StarReviews),2) #<- .49
round(cor(readyData$Volume, readyData$ProductTypeGameConsole),2) #<- .39
round(cor(readyData$Volume, readyData$NegativeServiceReview),2) #<- .31
round(cor(readyData$Volume, readyData$x1StarReviews),2) #<- .26
round(cor(readyData$Volume, readyData$Price),2) #<- -.18
###Need to not include 5 star reviews to avoid overfitting

#check for Colinearity - moderate to high intercorrelations among independant 
#variables. 
#problems with colinearity - if 2 or more independant variables contain essentially
#the same informaion to a large exten, one gains little by using both in the 
#regression model. Leads to unstable estimated as it tens to increase the variances 
#of regression coefficients. Solution is to keep only one of the two independant 
#variables that are highly correlated in the regression model.

Related <- (c("x4StarReviews", 
              "x3StarReviews", 
              "x2StarReviews", 
              "x1StarReviews"))
Collinearity <- readyData[,Related]
corrData1 <- cor(Collinearity)
corrData1
##x4 and x3 are highly correlated. So we only need to keep one.Keep x4 and drop
#x3 since x4 is more correlated to Volume.
##x2 and x1 are highly correlated. So we only need to keep one.Keep x2 and drop
#x1 since x2 is more correlated to Volume. 

##important variables are: 1) x4StarReviews
##                         2) PositiveServiceReview
##                         3) x2StarReviews
##                         4) ProductTypeGameConsole
##                         5) NegativeServiceReview

##drop below data (due to collinearity and overfit [5 Star, 3 Star and 1 star]): 
length(readyData) #<- number of features (28)
names(readyData) #<- names of feature
write.table(readyData, "readyData.csv", sep = ',')
seedata <- readyData[, -c(15,17,19)] #<- removing collinearity and overfit features
length(seedata) #<- verifying (25)
names(seedata) #<- verifying 
write.table(seedata, "seedata.csv", sep = ',')

#check outliers (can get actual outlier values with $out)
boxplot(seedata$x4StarReviews)$out
boxplot(seedata$PositiveServiceReview)$out
boxplot(seedata$x2StarReviews)$out
boxplot(seedata$ProductTypeGameConsole)$out
boxplot(seedata$NegativeServiceReview)$out

#===========================================
#====Develop Multiple Regression Models=====
#===========================================

set.seed(123)

#Split the data into training and test set 
trainSize<-round(nrow(seedata)*0.7) 
testSize<-nrow(seedata)-trainSize

training_indices<-sample(seq_len(nrow(seedata)),size =trainSize)
trainSet<-seedata[training_indices,]
testSet<-seedata[-training_indices,] 

#Linear Regression Model
LRModel <-lm(Volume~ x4StarReviews
             +PositiveServiceReview
             +x2StarReviews
             +ProductTypeGameConsole
             +NegativeServiceReview, trainSet)

#Summary of Model
#Multiple R-squared- How well the regression line fits the data (1 means it’s a perfect fit).
#p-value - Tells you how much the Independent Variable/Predictor affects the Dependent 
#Variable/Response/. A p-value of more than 0.05 means the Independent Variable has no effect 
#on the Dependent Variable; less than 0.05 means the relationship is statistically significant.

summary(LRModel) #<- Multiple R-Squared: .8426, R-Squared: .8095
#p-value: 1.461X10^-10

###Non-Parametric Machine Learning Models###
##Load Caret 
library(caret)

#set seed
set.seed(123)

# define an 75%/25% train/test split of the dataset
inTraining <- createDataPartition(seedata$Volume, p = .70, list = FALSE)
training <- readyData[inTraining,]
testing <- readyData[-inTraining,]

#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

##Support Vector Machine##
##Support Vector Machine in Caret
ModelSVM <- train(Volume ~x4StarReviews
                  +PositiveServiceReview
                  +x2StarReviews
                  +ProductTypeGameConsole
                  +NegativeServiceReview, data = training, method = "svmLinear"
                  ,trControl = fitControl)
ModelSVM#<- RMSE: 353.3065, RSquared: .8646, MAE: 261.722
PredictSVM <- predict(ModelSVM, data = training)
PredictSVM #<- Negative Values 

##Random Forest##
#set seed
set.seed(123)
##Random Forest in Caret
ModelRF <- train(Volume ~x4StarReviews
                 +PositiveServiceReview
                 +x2StarReviews
                 +ProductTypeGameConsole
                 +NegativeServiceReview, data = training, method = "rf"
                 ,trControl = fitControl)
ModelRF #<- RMSE: 178.233, RSquared: .9863, MAE: 126.20, Mtry = 2
PredictRF <- predict(ModelRF, data = training)
PredictRF #<- no negative values! 

##Gradient Boosting Machine##

#set seed
set.seed(123)

##Gradient Boosting Machin in Caret
ModelGBM <- train(Volume ~x4StarReviews
                  +PositiveServiceReview
                  +x2StarReviews
                  +ProductTypeGameConsole
                  +NegativeServiceReview, data = training, method = "gbm"
                  ,distribution = "gaussian"
                  ,trControl = fitControl)
ModelGBM
PredictGBM <- predict(ModelGBM, data = training)
PredictGBM #<- negative values!

#============================
#====Import New Data Set=====
#============================

library(readr)
newproducts <- read_csv("C:/Users/admin/Desktop/Task 3 - Multiple Regression/newproductattributes2017.csv")
View(newproducts)

#===========================================
#UNDERSTANDING THE DATA
#===========================================

#Look into data set:
summary(newproducts)

#Data have NA'S?
sum(is.na(newproducts)) #<- No

#Structure of Data 
str(newproducts) #<- all NUM but Product Type which is Chr
## Cannot do regression analysis on non numeric variables## 
##When a researcher wishes to include a categorical variable, 
##Supplementary steps are required to make the results interpretable

#===========================================
#PREPROCESSING THE DATA
#===========================================

# dummify the data
newDataFrame1 <- dummyVars(" ~ .", data = newproducts)
readyData1 <- data.frame(predict(newDataFrame1, newdata = newproducts))

# correlation Data <- Find the correlation between relevant independant 
# variables and the dependant variable. 
corrData2 <- cor(readyData1)
corrData2

#Install corrplot package
install.packages("corrplot")

#call package
library(corrplot)

#visualize correlation
corrplot(corrData1)

readyData1$BestSellersRank <- NULL
readyData1$x5StarReviews <- NULL
names(readyData1)

##Verifying correlation 
related1 <- c("x4StarReviews"
                         ,"x3StarReviews"
                         ,"x2StarReviews"
                         ,"x1StarReviews")
Collinearity1 <- readyData1[,related1]
corrData2 <- cor(Collinearity1)
corrData2
names(readyData1)

readyData1 <- readyData1[-c(16,18)]
names(readyData1)

#=============================================
#====Predict Volume on New Data with Model====
#=============================================

Predictions<-predict(ModelRF, newdata=readyData1)
Predictions

#====================================================
#====Add predictions to the new products data set====
#====================================================
output <- newproducts 
output$predictions <- Predictions

write.csv(output, file="C2.T3output.csv", row.names = TRUE)



plot(output$predictions, output$x5StarReviews, xlab ="Predictions", ylab = "5 Star Reviews")
plot(output$predictions, output$x4StarReviews, xlab ="Predictions", ylab = "4 Star Reviews")





