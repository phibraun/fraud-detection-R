# Data Loading and Preprocessing

library(caret)
library(dplyr)
library(MASS)
library(pROC)
library(ggplot2)
library(corrplot)
library(glmnet)
library(randomForest)
set.seed(1234)
df <- read.csv("Transactions_Data.csv")

#One-hot encoding 'type' column
dummy.model<- dummyVars(~ type, data = df, fullRank = TRUE)
encoded_type<- predict(dummy.model, newdata = df)
encoded_type<- as.data.frame(encoded_type)
dataset<- cbind(df, encoded_type)
dataset$type<- NULL

#Identifier Redundancy Check
# Checking for unique and repeated nameOrig/nameDest
# These names are often unique and only used once so they may not be
# helpful for prediction.
unique_nameOrig <- n_distinct(dataset$nameOrig)
num_repeats_nameOrig <- sum(duplicated(dataset$nameOrig))

unique_nameDest <- n_distinct(dataset$nameDest)
num_repeats_nameDest <- sum(duplicated(dataset$nameDest))

cat("Repeated nameOrig values:", num_repeats_nameOrig, "\n")
cat("Repeated nameDest values:", num_repeats_nameDest, "\n")
# Since the number of fraud cases only occur at most once per account, 
# the name of origin is not very useful for predicting fraud.
# So we will drop this predictor.

#Dropping nameOrig, oldbalanceOrg, oldbalanceDest, and isFlaggedFraud

#Correlation Plot

png("correlation_matrix.png", width = 800, height = 600)
numeric_vars <- dataset[, sapply(dataset, is.numeric)]
corrplot(cor(numeric_vars, use = "complete.obs"), method = "color")
dev.off()

dataset<- dataset[, !(names(dataset) %in% 
                         c("nameOrig", "oldbalanceOrg", 
                           "oldbalanceDest", "isFlaggedFraud"))]

# Encoding nameDest frequency
nameDest_freq <- dataset %>% count(nameDest, name = "nameDest_freq")
dataset <- dataset %>% 
  left_join(nameDest_freq, by = "nameDest") %>% dplyr::select(-nameDest)

# Exploratory Data Analysis

#Distribution of Amount
p1 <- ggplot(dataset %>% filter(amount > 0), aes(x = amount)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  scale_x_log10() +
  labs(title = "Distribution of Amount (Log-Scaled)",
       x = "Log10(Amount)", y = "Count")
ggsave("amount_distribution.png", plot = p1, width = 6, height = 4, dpi = 300)

#Class Proportions
fraud_counts <- table(dataset$isFraud)
fraud_proportions <- prop.table(fraud_counts)
print(fraud_counts)
print(fraud_proportions)

#Amount by Fraud Status
p2 <- ggplot(dataset, aes(x = amount + 1, fill = factor(isFraud))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 50) +
  scale_x_log10() +
  facet_wrap(~ isFraud, scales = "free_y") +
  labs(title = "Transaction Amount by Fraud Status",
       fill = "isFraud")
ggsave("amount_by_fraud_status.png", plot = p2, width = 6, height = 4, dpi = 300)

# LASSO Logistic Regression
# Creating training and test sets

split <- createDataPartition(dataset$isFraud, p= 0.7, list= FALSE)
train<- dataset[split, ]
test<- dataset[-split, ]
train_down<- downSample(
  x = train[, -which(names(train) == "isFraud")],
  y = as.factor(train$isFraud),
  yname= "isFraud"
)

# Prepare matrix format for glmnet
x_train<- model.matrix(isFraud ~ ., data = train_down)[,-1]
y_train<- train_down$isFraud

cv_model<-cv.glmnet(x_train, y_train, family= "binomial", alpha=1)

# Finding best lambda
best_lambda<- cv_model$lambda.min
cat("Optimal lambda:", best_lambda, "\n")
lasso_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)

# Predicting on test set
x_test<- model.matrix(isFraud ~ ., data = test)[, -1]
pred_probs<- predict(lasso_model, newx = x_test, type = "response")
pred_labels<- ifelse(pred_probs > 0.5, 1, 0)

# Evaluating performance
confusionMat <- confusionMatrix(factor(pred_labels), factor(test$isFraud), positive = "1")
print(confusionMat)

# ROC Curve and AUC

roc_obj <- roc(test$isFraud, as.numeric(pred_probs))
png("lasso_roc_curve.png", width = 800, height = 600)
plot(roc_obj, main = "ROC Curve - LASSO Logistic Regression", col = "blue", lwd = 2)
dev.off()
cat("AUC:", auc(roc_obj), "\n")

# Deviance residual analysis
# Convert predicted probabilities to numeric vector
fitted_probs <- as.numeric(pred_probs)
y_obs <- as.numeric(as.character(test$isFraud))  # Ensure it's 0/1

# Bounding predicted values away from 0 and 1 to avoid log(0)
fitted_probs <- pmin(pmax(fitted_probs, 1e-15), 1 - 1e-15)

# Computing deviance residuals
dev_residuals <- sign(y_obs - fitted_probs)* sqrt(-2*(
  y_obs*log(fitted_probs) + (1- y_obs)* log(1 - fitted_probs)
))

# Plot histogram of deviance residuals
png("lasso_deviance_residuals.png", width = 800, height = 600)
hist(dev_residuals, breaks = 50,
     main = "Deviance Residuals (LASSO Logistic Regression)",
     xlab = "Deviance Residuals", col = "lightblue")
dev.off()

# Random Forest

rf_model <- randomForest(isFraud ~ .,
                         data = train_down, ntree = 100, importance = TRUE)
rf_probs<- predict(rf_model, newdata = test, type = "prob")[, 2]
rf_preds<- ifelse(rf_probs > 0.5, 1, 0)
rf_cm<- confusionMatrix(factor(rf_preds), factor(test$isFraud), positive = "1")
print(rf_cm)

rf_roc <- roc(test$isFraud, rf_probs)
png("random_forest_roc_curve.png", width = 800, height = 600)
plot(rf_roc, main = "ROC Curve - Random Forest", col = "darkgreen", lwd = 2)
dev.off()
cat("Random Forest AUC:", auc(rf_roc), "\n")
varImpPlot(rf_model, main = "Random Forest Feature Importance")

# Variable importance from Random Forest
importance_vals <- importance(rf_model)

# Plot variable importance
varImpPlot(rf_model,
           main = "Random Forest Variable Importance (Gini Index)",
           n.var = min(10, ncol(importance_vals)))

