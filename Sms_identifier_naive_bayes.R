#-----importing data-----

library(readr)

urlfile = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/sms_spam.csv"

sms_raw <- read.csv(url(urlfile), stringsAsFactors = F)

str(sms_raw)

sms_raw$type <- factor(sms_raw$type)

str(sms_raw$type)
table(sms_raw$type)

#----- data preparation -----

#install.packages("tm")
library(tm)

sms_corpus <- Corpus(VectorSource(sms_raw$text))

print(sms_corpus)
inspect(sms_corpus[1:3])

#cleaning the text data
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
inspect(corpus_clean[1:10])

#convert itu document term matrix to tokenize each sms word
sms_dtm <- DocumentTermMatrix(corpus_clean)

#splitting data frame
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test <- sms_raw[4170:5574, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5574, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5574]

prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

#visualizing words with wordcloud
#install.packages("wordcloud")
library(wordcloud)

wordcloud(sms_corpus_train, min.freq = 40, random.order = F)

##subsetting and comparing spam and ham for visualization
spam <- subset(sms_raw_train, type == "spam")
ham <- subset(sms_raw_train, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.2), random.order = F)
wordcloud(ham$text, max.words = 40, scale = c(3, 0.2), random.order = F)

#saving frequent words only
sms_dict <- c(findFreqTerms(sms_dtm_train, 5))

sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))

#converting count of times of words appearing in sparse matrix to factor type matrix
convert_counts <- function(x){
  x = ifelse(x>0, 1, 0)
  x = factor(x, levels = c(1, 0), labels = c("Yes", "No"))
  return(x)
}

sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_test, MARGIN = 2, convert_counts)

#-----Training the Data-----
#install.packages("e1071")
library(e1071)

sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)

#----- Testing the classifier and evaluating the model -----
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type, prop.chisq = F, prop.t = F, dnn = c('predicted', 'actual'))

#improving the model by adding laplace estimator = 1
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_test_pred2, sms_raw_test$type, prop.chisq = F, prop.t = F, prop.r = F, dnn = c("predicted", "actual"))
