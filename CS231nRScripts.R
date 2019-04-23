library(magrittr)
library(readr)
library(tidyverse)
SFProperty <- readr::read_csv("SFProperty.csv")
names(SFProperty)[1]<-"CRY"

names(SFProperty)[2]<-"Property_Location"
names(SFProperty)[3]<-"Parcel_Number"
names(SFProperty)[6]<-"Volume_Number"
names(SFProperty)[7]<-"Use_Code"
names(SFProperty)[8]<-"Use_Definition"
names(SFProperty)[9]<-"Property_Class_Code"
names(SFProperty)[10]<-"Property_Class_Code_Definition"
names(SFProperty)[11]<-"Year_Property_Built"
names(SFProperty)[12]<-"Number_Bathrooms"
names(SFProperty)[13]<-"Number_Bedrooms"
names(SFProperty)[14]<-"Number_Rooms"
names(SFProperty)[15]<-"Number_Stories"
names(SFProperty)[16]<-"Number_Units"
names(SFProperty)[17]<-"Zoning_Code"
names(SFProperty)[18]<-"Construction_Type"
names(SFProperty)[19]<-"Lot_Depth"
names(SFProperty)[20]<-"Lot_Frontage"
names(SFProperty)[21]<-"Property_Area"
names(SFProperty)[22]<-"Basement_Area"
names(SFProperty)[23]<-"Lot_Area"
names(SFProperty)[24]<-"Lot_Code"

names(SFProperty)[25]<-"Tax_Rate_Area_Code"
names(SFProperty)[26]<-"Percent_of_Ownership"
names(SFProperty)[27]<-"Exemption_Code"
names(SFProperty)[28]<-"Exemption_Code_Definition"

names(SFProperty)[29]<-"Status_Code"
names(SFProperty)[30]<-"Misc_Exemption_Value"
names(SFProperty)[31]<-"Homeowner_Exemption_Value"
names(SFProperty)[32]<-"Current_Sales_Date"

names(SFProperty)[33]<-"Assessed_Fixtures_Value"
names(SFProperty)[34]<-"Assessed_Improvement_Value"
names(SFProperty)[35]<-"Assessed_Land_Value"
names(SFProperty)[36]<-"Assessed Personal_Property_Value"

names(SFProperty)[37]<-"Assessor_Neighborhood_District"
names(SFProperty)[38]<-"Assessor_Neighborhood_Code"
names(SFProperty)[39]<-"Assessor_Neighborhood"
names(SFProperty)[40]<-"Supervisor_District"

names(SFProperty)[41]<-"Analysis_Neighborhood"
names(SFProperty)[43]<-"Row_ID"

library(tidyverse)
dim(SFProperty)
SFProperty2017 <- SFProperty %>%  #Filtering only 2017 assessments and creating a new dataframe(tibble) name SFProperty2017
  filter(CRY==2017)

dim(SFProperty2017)


SFProperty2017 %>%
  filter(Property_Class_Code_Definition=="Dwelling")

SFProperty2017 %>%
  filter(Property_Class_Code=="D")

SFProperty2017 %>%
  filter(Use_Definition=="Single Family Residential")

unique(SFProperty2017$Analysis_Neighborhood)

SFProperty2017_Dwelling <- SFProperty2017 %>%
  filter(Property_Class_Code_Definition=="Dwelling")

SFProperty2017_Dwelling_Full_Ownership <- SFProperty2017_Dwelling %>%
  filter(Percent_of_Ownership==1)

SFProperty2017_Dwelling_Full_Ownership_Single_Unit <- SFProperty2017_Dwelling_Full_Ownership %>%
  filter(Number_Units==1)

dim(SFProperty2017_Dwelling_Full_Ownership_Single_Unit)
View(SFProperty2017_Dwelling_Full_Ownership_Single_Unit)



LAProp <- readr::read_csv("LAProp.csv")

View(LAProp)
LAProp_Residential <- LAProp %>%
  filter(SpecificUseType=="Single Family Residence")

Santa_Monica_residential <- LAProp_Residential %>%
  filter(City== "SANTA MONICA CA")

dim(Santa_Monica_residential)

Santa_Monica_residential_Single_Family <- Santa_Monica_residential %>%
  filter(PropertyType=="SFR")

Santa_Monica_residential_Single_Family

write_csv(Santa_Monica_residential_Single_Family, "Santa_Monica_residential_Single_Family.csv")
write_csv(SFProperty2017_Dwelling_Full_Ownership_Single_Unit, "SFProperty2017_Dwelling_Full_Ownership_Single_Unit.csv")

LAProp_Residential_2010 <- LAProp_Residential %>%
  filter(ImpBaseYear>2010)

LAProp_Residential_2012 <- LAProp_Residential %>%
  filter(ImpBaseYear>2012)

LAProp_Residential_2016 <- LAProp_Residential %>%
  filter(ImpBaseYear>2016)

LAProp_Residential_2017 <- LAProp_Residential %>%
  filter(ImpBaseYear>2017)

LAProp_Residential_2017_Only_Houses <- LAProp_Residential_2017 %>%  # Removed empty lands and houses with more than 2 units (more than 1 unit assessment is tricky)
  filter(ImprovementValue>0 & Units==1 & PropertyType=="SFR") 

dim(LAProp_Residential_2017_Only_Houses)  # [1] 35456    51 Dimension of our new dataset after removing empty lands

# Now, I see zip codes like 90222-1823, but we need to standardize it to 5 digit to feed into regression models. Otherwise, it will treat the same zip code
# as different zip codes. Below function removes anyting after -

LAProp_Residential_2017_Only_Houses$ZIPcode <- gsub(LAProp_Residential_2017_Only_Houses$ZIPcode, pattern="-.*", replacement = "")


range(LAProp_Residential_2017_Only_Houses$TotalValue)


d <- density(LAProp_Residential_2017_Only_Houses$TotalValue)
plot(d)

hist(log(log(LAProp_Residential_2017_Only_Houses$TotalValue)), 
     main="LA House Values", 
     xlab="House Values", 
     border="red", 
     col="green",
     xlim=c(2.3,2.85),
     las=1, 
     breaks=50)


# Around  80% of house values are less or equal to 1M



LAProp_Residential_2018 <- LAProp_Residential %>%
  filter(ImpBaseYear>2018)


dim(LAProp_Residential )


SantaMonica_Residential_2010 <- Santa_Monica_residential_Single_Family %>%
  filter(ImpBaseYear>2010)

readr::write_csv(LAProp_Residential_2017, "LAProp_Residential_2017.csv")
readr::write_csv(LAProp_Residential_2016, "LAProp_Residential_2016.csv")

names(LAProp_Residential_2017)[51] <-"Location1"

Check <- LAProp_Residential_2017 %>%
  group_by(AIN, PropertyLocation) %>%
  filter(row_number() == 1)

dim(Check)

LAProp_Residential_2017_Only_Houses %>%
  filter(TotalValue<1000000)

dim(LAProp_Residential_2017)
LAProp_Residential_2017 %>%
  filter(TotalValue<1000000)

# Getting variables from the LAProp_Residential_2017_Only_Houses data
LAProp_Residential_2017_Only_Houses_Regres <- LAProp_Residential_2017_Only_Houses[, c(1,5, 12, 13, 14, 16, 17, 18, 19, 34)] 
dim(LAProp_Residential_2017_Only_Houses_Regres)
LAProp_Residential_2017_Only_Houses_Regres$ZIPcode <- as.factor(LAProp_Residential_2017_Only_Houses_Regres$ZIPcode)
LAProp_Residential_2017_Only_Houses_Regres$TaxRateArea <- as.numeric(LAProp_Residential_2017_Only_Houses_Regres$TaxRateArea)
LAProp_Residential_2017_Only_Houses_Regres$SpecificUseDetail1 <- as.factor(LAProp_Residential_2017_Only_Houses_Regres$SpecificUseDetail1)
LAProp_Residential_2017_Only_Houses_Regres$SpecificUseDetail2 <- as.factor(LAProp_Residential_2017_Only_Houses_Regres$SpecificUseDetail2)


write_csv(LAProp_Residential_2017_Only_Houses_Regres, "LAProp_Residential_2017_Only_Houses_Regres.csv") 
write_csv(LAProp_Residential_2017_Only_Houses, "LAProp_Residential_2017_Only_Houses.csv") 

# LAProp_Residential_2017_Only_Houses

library(gbm)

set.seed(1)
trainid <- sample(1:nrow(LAProp_Residential_2017_Only_Houses_Regres), nrow(LAProp_Residential_2017_Only_Houses_Regres)*0.8)
train <- LAProp_Residential_2017_Only_Houses_Regres[trainid,]
test <- LAProp_Residential_2017_Only_Houses_Regres[-trainid,]

boosting_LA <- gbm(log(TotalValue)~.,data = train, distribution = "gaussian",n.trees =
                       3000, interaction.depth = 4, shrinkage=0.1)

summary(boosting_LA)

training_prediction=predict(boosting_LA, train, n.trees = 3000)
test_prediction=predict(boosting_LA, test, n.trees = 3000)
training_error = mean((training_prediction-log(train$TotalValue))^2)
test_error = mean((test_prediction-log(test$TotalValue))^2)

#Gradient Boosted Regression gave below mse results after log-transformation of house assessment values, I have some suspicion in the results
# training_error
#[1] 0.03468971
# test_error
#[1] 0.04150031

experiment <- LAProp_Residential_2017_Only_Houses_Regres %>%
  filter(TotalValue > 150000 & TotalValue < 1500000)

set.seed(1)
trainid <- sample(1:nrow(experiment), nrow(experiment)*0.8)
train <- experiment[trainid,]
test <- experiment[-trainid,]

boosting_experiment <- gbm(log(TotalValue)~.,data = train, distribution = "gaussian",n.trees =
                     3000, interaction.depth = 4, shrinkage=0.1)

summary(boosting_experiment)

training_prediction_experiment = predict(boosting_experiment, train, n.trees = 3000)
test_prediction_experiment = predict(boosting_experiment, test, n.trees = 3000)
training_error_experiment = mean((training_prediction_experiment-log(train$TotalValue))^2)
test_error_experiment = mean((test_prediction_experiment-log(test$TotalValue))^2)
                                                                                
