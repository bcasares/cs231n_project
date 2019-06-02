rm(list=ls())
library(magrittr)
library(readr)
# library(tidyverse)

# Reading train, val, test files

train = read.csv("data/train.csv")
val = read.csv("data/val.csv")
test = read.csv("data/test.csv")

train_regress = train[,-c(1,12,14,9,10)]
val_regress = val[,-c(1,12,14,9,10)]
test_regress = test[,-c(1,12,14,9,10)]

# # Linear Regression for Baseline
LinearModel <- lm(log(TotalValue) ~ ., train_regress)
# LinearModel <- lm(TotalValue ~ ., train_regress)
Val_Predict <- predict(LinearModel, val_regress)
Test_Predict <- predict(LinearModel, test_regress)

sm <- summary(LinearModel)

rmse <- function(sm){
  (mean(sm$residuals^2))
}
# plot(train_regress$Bathrooms, train_regress$TotalValue)
# plot(train_regress$Bathrooms, log(train_regress$TotalValue))
test_mse = rmse(sm)
test_rmse = sqrt(test_mse)

val_mse = mean((Val_Predict - log(val$TotalValue))^2)
# val_mse = mean((Val_Predict - val$TotalValue)^2)
val_rmse = sqrt(val_mse)