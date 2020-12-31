#----- importing the Dataset -----

library(readr)

urlfile = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv"
credit <- read.csv(url(urlfile))

#----- Data exploration -----
str(credit)

#freq table of related categorical features
table(credit$checking_balance)
table(credit$savings_balance)

#statistical value of related numerical features
summary(credit$months_loan_duration)
summary(credit$amount)

#default is condition of unmeet payment agreement for loan
credit$default <- factor(credit$default, levels = c(1, 2), labels = c("No", "Yes"))
str(credit$default)
table(credit$default)

#----- Preparing training and test data -----
set.seed(12345)
credit_rand <- credit[order(runif(1000)), ]

summary(credit$amount)
summary(credit_rand$amount)

head(credit$amount)
head(credit_rand$amount)

credit_train <- credit_rand[1:900, ]
credit_test <- credit_rand[901:1000, ]

#----- Training the Model -----
#install.packages("C50")
library(C50)

#build the model excluding col 17 (default feature) and use it as target feature
credit_model <- C5.0(credit_train[-17], credit_train$default)
credit_model
summary(credit_model)
