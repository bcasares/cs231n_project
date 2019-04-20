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



LAProp <- read_csv("LAProp.csv")

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

