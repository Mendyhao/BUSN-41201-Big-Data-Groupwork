library(glmnet)
library(corrplot)

fx <- read.csv("FXmonthly.csv")
sorted_col <- sort(colnames(fx))
sorted_data <- fx[, sorted_col]
fx <- (sorted_data[2:120,]-sorted_data[1:119,])/(sorted_data[1:119,])
sp <- read.csv("sp500.csv")
currency <- read.table(file = "currency_codes.txt", header = TRUE)
colnames(currency) <- c("code", "country")
new_row <- data.frame(code = "al", country = "australia")
currency <- rbind(currency, new_row)
currency <- currency[order(currency$code), ]

fx_codes <- substr(colnames(fx), 3, 4)
colnames(fx) <- currency$code[match(fx_codes, currency$code)]

#Question 1:
correlation_matrix <- cor(fx)
correlation_matrix
corrplot(correlation_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)
#The correlation matrix of the FX dimensions reveals varying degrees of correlation among the different currency pairs. Some pairs exhibit strong positive correlations, indicating that they tend to move together. For example, pairs like (exnzus, exdnus), (exnous, exdnus), and (exnzus, exalus) demonstrate correlations above 0.8, suggesting significant co-movements.
#These strong correlations imply the presence of common underlying factors influencing the exchange rates of these currency pairs. In the context of factor modeling, this indicates that a smaller set of latent factors could potentially explain the variation in these correlated currency pairs. Factor models, such as principal component analysis (PCA) or factor analysis, could effectively capture the common variation among these currency pairs and provide insights into the underlying factors driving their movements.
#Conversely, some pairs exhibit weaker correlations or even negative correlations. For example, pairs like (excaus, exchus) and (exhkus, exjpus) have correlations close to zero, indicating little to no linear relationship between them. In such cases, factor models may not be as effective in capturing the common variation among these pairs, and other modeling approaches may be necessary to understand their dynamics.

#Question 2:
fx_pca <- prcomp(fx, scale. = TRUE)
summary(fx_pca)
plot(fx_pca)
#Principal component analysis (PCA) is a powerful technique used for dimensionality reduction and identifying patterns in high-dimensional data. In the provided output, we have the results of PCA applied to a dataset containing information about currency pairs. The output includes the standard deviations of the principal components, which indicate the amount of variance captured by each component. Additionally, the rotation matrix shows the loadings of the original variables (currency pairs) on each principal component. These loadings help interpret the meaning of each principal component in the context of the original data.
#Interpreting principal components involves analyzing the loadings to understand which variables contribute most strongly to each component and the nature of their relationships. By examining the magnitude and sign of the loadings, we can identify which currency pairs have the greatest influence on each principal component and whether they move together or in opposite directions. For instance, if certain currency pairs consistently have high positive loadings on a principal component, it suggests that they tend to move in tandem, while negative loadings indicate an inverse relationship.
#Moreover, identifying patterns among the variables with high loadings allows us to give meaningful interpretations to the principal components. For example, if a principal component has high loadings on currency pairs from a particular region or with similar economic characteristics, it may represent movements driven by factors specific to that region or economic sector. Naming the principal components based on these patterns helps communicate their meaning and facilitates further analysis and interpretation of the data.
#Furthermore, considering the proportion of variance explained by each principal component provides insight into how much information each component captures. Higher proportions indicate that the component accounts for a larger portion of the variability in the data, making it more important for understanding the underlying structure or patterns. This information is valuable for selecting which principal components to retain for further analysis or visualization, as well as assessing the overall effectiveness of the PCA in capturing the variability present in the dataset.

#Question 3:
set.seed(123)
K <- 3
currency_factors <- predict(fx_pca, newdata = fx)

# Fit GLM model
sp_model_glm <- glm(sp500 ~ currency_factors[, 1:K], data = sp, family = gaussian)
summary(sp_model_glm)

#The GLM model results indicate that the intercept term is approximately 0.0004431. Additionally, the coefficients for the first three principal components (PC1, PC2, and PC3) of the currency factors are 0.0059741, -0.0111795, and 0.0082100, respectively. These coefficients represent the estimated effect of each principal component on the SP500 returns. For instance, a one-unit increase in PC1 is associated with an increase of approximately 0.0059741 in SP500 returns, holding other variables constant. Conversely, a one-unit increase in PC2 is associated with a decrease of approximately 0.0111795 in SP500 returns, holding other variables constant. PC3 has a positive coefficient, indicating that an increase in PC3 is associated with an increase in SP500 returns.

# Fit Lasso model using cross-validation
sp_model_lasso <- cv.glmnet(x = currency_factors, y = sp$sp500, nfolds = 20)
coef(sp_model_lasso)

#The lasso model results show that the minimum mean-squared error (MSE) is achieved with a lambda value of approximately 0.001539, corresponding to 3 non-zero coefficients. The 1 standard error rule (1se) selects a lambda value of approximately 0.009896, resulting in 2 non-zero coefficients. This suggests that the lasso model is penalizing some coefficients to zero, leading to a sparse model with fewer predictors. The coefficients associated with the selected predictors provide insights into the relationship between the currency factors and SP500 returns, considering the regularization effect of the lasso technique. The lower MSE values indicate better predictive performance of the model.

#Question 4:
fx_matrix <- as.matrix(fx)

lasso_model <- cv.glmnet(x = fx_matrix, y = sp$sp500, nfolds = 20)
coef(lasso_model)

par(mfrow = c(1,2))
plot(sp_model_lasso, main = "LASSO with Cross-validation")
plot(lasso_model, main = "LASSO with Original Covariates")

#Lasso: 
#Lasso performs feature selection by penalizing the absolute size of the coefficients using an L1 penalty. This penalty encourages sparsity in the coefficient vector, effectively shrinking some coefficients to zero and selecting a subset of features that contribute most to the prediction.
# Lasso tends to produce simpler models with fewer nonzero coefficients. By penalizing the absolute size of the coefficients, lasso effectively shrinks some coefficients to zero, leading to a sparse model. This can help prevent overfitting and improve interpretability.

#PCR: 
#PCR, on the other hand, does not perform explicit feature selection. Instead, it creates new orthogonal features (principal components) that are linear combinations of the original features. PCR does not inherently eliminate features; rather, it reduces the dimensionality of the feature space by representing the data with fewer principal components.
# PCR reduces the dimensionality of the feature space by capturing the variability in the data using a smaller number of principal components. However, PCR does not necessarily reduce the number of features in the model, as all principal components are retained. As a result, PCR models can be more complex and may suffer from overfitting if too many principal components are retained.

#In summary, while both lasso and PCR are techniques for regression modeling, they differ in their approach to feature selection and model complexity. Lasso explicitly selects a subset of features by shrinking coefficients, leading to sparser models, while PCR reduces dimensionality by creating new features based on linear combinations of the original features, potentially resulting in more complex models with all principal components retained.


