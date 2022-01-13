![image](https://github.com/sunil-20/Text-classification-project/blob/main/Images/cfpb.png)
---
# Table of Contents
1. [Background & Motivation](#background)
2. [Data Source](#data)
3. [EDA & Feature Engineering](#eda)
   1. [Product type and number of complaints](#product)
   2. [Text length of the complaint](#text)
   3. [Wordcloud of whole dataset](#wcw)
   4. [Wordcloud of Credit or Prepaid card class](#credit)
   5. [Wordcloud of Checking or Savings account](#check)
4. [Model used](#model)
   1. [Random Forest](#rf)
   2. [Logistic Regression](#lg)
   3. [Multinomial Naive bayes](#nb)
   4. [Linear SVC](#svc)
   5. [XGBoost](#xgb)
5. [Conclusions](#result)
6. [Citations](#ref)

## 1. Background & Project Motivation <a name="background"></a>
Considering the importance of consumer complaints which provides a significant opportunity to improve the business, this project aimed at using the complaint dataset to classify the complaints according to specified categories they fall on. Complaints that have been filed/given in a complaint filing institution/financial marketplace by a consumer are valuable resources to address many business problems. We can leverage the consumer complaint by identifying the legitimate issues shared by the client which might indicate if the business process has been some issues for smooth client services.<br><br>
When there is a complaint from a client with a product or service of the company, that helps in modifying or improving the service/product which might be related to client relationships or other product features or aspects. As these complaints might be accessible publicly, that may allow the public to have an overview of the business or the organization. Consumers are always willing to adhere to the business with less complaint. So, many complaints of the business in the public domain could hamper the company’s reputation. So, as a final take, companies need to take the complaint as a learning resource for upgrading their product or services. <br><br>
Financial services like Chase bank are a critical sector that receives many complaints regarding the service provided by the institution. So, in the first place, we must segregate those complaints into a specific category/class before handling them for further business improvement. To have an overview of the consumer complaint, this project has utilized the dataset from Consumer Financial Protection Bureau (CFPB). CFPB is a U.S. government agency that helps the consumer to be fairly treated by banks, lenders, and other financial companies.

## 2. Data & Model <a name="data"></a>
The data consists of 1,048,575 observations and 18 features which includes all financial institutions complaints provided on CFPB. I have narrowed down the institution to one which allows for an overview of one institution. Using one institution also provides a strong model outcome and interpretability. Additionally, the model doesn’t exhaust available computing resources while running the algorithm. As JPMorgan Chase & Co. is one of the leading financial institutions with a great track record and relatively old institution, I have chosen to use the complaint received by this institution to use as a multi-text classification. <br><br>
* __Website__: https://www.consumerfinance.gov/data-research/consumer-complaints/

## 3. EDA & Feature Engineering <a name="eda"></a><br>
This section provides some useful information regarding the data and visualisations of important text used in the complaints.<br><br>

### 3.1. Product type and number of complaints<a name="product"></a><br>

The following table highlights the product or complaint type and the number of complaints in each categories.<br>
| Product    | Number of complaints |
| :---        |    :----:   |
|1. Credit card or prepaid card|                                                     4023|
|2. Checking or savings account|                                                     3364|
|3. Credit reporting, credit repair services, or other personal consumer reports|    1631|
|4. Mortgage|                                                                        1022|
|5. Money transfer, virtual currency, or money service|                               752|
|6. Debt collection|                                                                  517|
|7. Vehicle loan or lease|                                                            316|
|8. Credit card|                                                                       88|
|9. Bank account or service|                                                           58|
|10. Payday loan, title loan, or personal loan|                                         38|
|11. Student loan|                                                                      11|
|12. Credit reporting|                                                                   5|
|13. Consumer Loan|                                                                      3|
|14. Money transfers|                                                                    2|
|15. Other financial service|                                                            1|

<br>
The following figure shows the number of complaints in each categories. Some of the categories which are relevant to combine has been merged together for better interpretation and optimization. Figure shows that Credit or prepaid card and Checking or savings account complaints are higher compared to other complaints categories. <br>

![image](https://github.com/sunil-20/Text-classification-project/blob/main/Images/Complaint_cases.png)<br><br>

### 3.2. Text length of the complaint<a name="text"></a><br>

The following table shows the text length and the respective product type.It shows that Debt collection has the longest text length with 29239 characters. <br>

| Product    | Text length|
| :---        |    :----:   |
|Debt collection|29239|
|Debt collection|25450|
|Credit or prepaid card|23778|
|Debt collection|21175|
|Money transfer, virtual currency, or money service|20189|
|Payday, title, or personal loan|19547|
|Payday, title, or personal loan|19161|
|Payday, title, or personal loan|17676|
|Payday, title, or personal loan|17117|
|Credit or prepaid card|16755|<br>


The following image shows how the text length is distributed across different product classes. We can see from the figure that, the product ID Number 5 has almost 30000 characters which is the class "Debt collection". We can see lot of outliers across all the classes. That means, text lenght is highly variant and it might depend on the person and the context of the complaint. We can confirm that almost 75% of the text length in all classes are less than 4000 characters. <br>

![image](https://github.com/sunil-20/Text-classification-project/blob/main/Images/Text_length.png)<br><br>

### 3.3. Wordcloud of the whole dataset<a name="wcw"></a><br>
The following figure shows wordcloud of the whole dataset. We can see that most frequent words used in the complaints are: credit card, account, charge, called, asked payment, time, amount etc.This provides an initial understanding of how they frame complaint about the product or services. <br>

![image](https://github.com/sunil-20/Text-classification-project/blob/main/Images/Wordcloud_whole.png)<br><br>

### 3.4. Wordcloud of Credit or Prepaid card class<a name="credit"></a><br>
The following figure has been narrowed down to only Credit or Prepaid card class. Here most frequent words are: credit card, payment, charge, account, merchant, dispute, information, purchase etc.Here many of the word are overlapped with the previous wordcloud.

![image](https://github.com/sunil-20/Text-classification-project/blob/main/Images/Wordcloud_prepaid_card.png)<br><br>

### 3.5. Wordcloud of Checking or Savings account<a name="check"></a><br>
The following figure provides an overview of frequent words used in Checking or Savings account class. This class has also received many complaints compared to other classes. Here the most frequent words are: account, check, money, time, charge, transaction, called, fund, checking account, one etc.<br>

![image](https://github.com/sunil-20/Text-classification-project/blob/main/Images/Wordcloud_Checking_Savings.png)<br><br>

## 4. Models used <a name="model"></a><br>
When we develop a model, we can’t blindly trust the model right away because it fits the training data well. So, we need to validate the model performance with different models and compare them to how they perform independently. Additionally, we can use the nested model where the output of the preceding trained model is used as part of the input for the succeeding model during the process of model training. Because of resource limitations, I stayed with a separate model with cross-validation steps, train-test split, and model parameter tuning. In the end, I have compared them with different model performance scoring criteria. Specifically, I have used F1-score, ROC_AUC score, and Accuracy for comparing the model performance. <br>   
### 4.1. Random Forest<a name="rf"></a><br>
A random forest is a machine learning model used to solve both classification and regression problems. It uses many decisions trees to reach the solutions and the algorithm is trained through bagging(ensemble model) or bootstrap aggregating so that each tree is different from the other and there is no correlation issue. For classification tasks, the output of random forest is the class most trees have selected. Each tree in the forest got a vote on the outcome for a given observation. <br>
| Model     | Accuracy| ROC_AUC score | F1 score(1) | F1 score(2) | F1 score(3)|
| :---      |   :----:|   :---:       | :---:       |:---:        |:---:       |
| __Base__  | 0.78    | 0.92          |  0.84       |  0.81       |0.82        |
| __Tuned__ | 0.74    | 0.94          | 0.79        | 0.81        | 0.77       |

### 4.2. Logistic Regression <a name="lg"></a><br>
Logistic model is used to model the probability of a certain class or events. Logistic regression uses a logistic function to maximize the entropy of the labels or classes conditioned on the features.<br>
The Logistic function takes the following form:<br>
<img src="https://github.com/sunil-20/Text-classification-project/blob/main/Images/logistic_fx.png" alt="F1-score" width="350" height="60"> <br>

| Model     | Accuracy| ROC_AUC score | F1 score(1) | F1 score(2) | F1 score(3)|
| :---      |   :----:|   :---:       | :---:       |:---:        |:---:       |
| __Base__  | 0.82    | 0.95          |  0.88       |  0.86       |0.86        |
| __Tuned__ | 0.82    | 0.95          | 0.88        | 0. 86       | 0.85       |

### 4.3. Multinomial Naive bayes<a name="nb"></a><br>

| Model     | Accuracy| ROC_AUC score | F1 score(1) | F1 score(2) | F1 score(3)|
| :---      |   :----:|   :---:       | :---:       |:---:        |:---:       |
| __Base__  | 0.71    | 0.93          |  0.78       |  0.81       |0.62        |
| __Tuned__ | 0.71    | 0.93          | 0.78        | 0.81        | 0.62       |

### 4.4. Linear SVC <a name="svc"></a><br>

| Model     | Accuracy| ROC_AUC score | F1 score(1) | F1 score(2) | F1 score(3)|
| :---      |   :----:|   :---:       | :---:       |:---:        |:---:       |
| __Base__  | 0.83    | 0.95          |  0.88       |  0.86       |0.86        |
| __Tuned__ | 0.82    | 0.95          | 0.88        | 0.85        | 0.86       |

### 4.5. XGBoost <a name="xgb"></a><br>

| Model     | Accuracy| ROC_AUC score | F1 score(1) | F1 score(2) | F1 score(3)|
| :---      |   :----:|   :---:       | :---:       |:---:        |:---:       |
| __Base__  | 0.80    | 0.94          |  0.85       |  0.83       |0.84        |
| __Tuned__ | 0.77    | 0.91          | 0.83        | 0.80        | 0.81       |

## 5. Conclusion <a name="result"></a><br>
The consumer complaint dataset has been trained using five different classification models: Random Forest, Logistic Regression, Naïve Bayes, Linear Support Vector Machine and XGBoost. As all of them are classification models, we can use the ROC_AUC, Accuracy and F1 score of each class to determine the model performance. 
<br>The ROC curve provides information on trade-off between the true positive rate and the false positive rate for given threshold. The bigger the total area under the curve (AUC), the better the model performance.<br>
The F1 Score is also an important model performance measure. F1 score is harmonic mean of precision and recall.<br> 
<img src="https://github.com/sunil-20/Text-classification-project/blob/main/Images/FScore.png" alt="F1-score" width="350" height="60">

Accuracy is a metrics used in classification models which measures the number of correct predictions. <br>
<br>
<img src="https://github.com/sunil-20/Text-classification-project/blob/main/Images/Accuracy.png" alt="F1-score" width="350" height="60"> <br>
| Model     | Accuracy| ROC_AUC score | F1 score(1) | F1 score(2) | F1 score(3)|
| :---      |   :----:|   :---:       | :---:       |:---:        |:---:       |
| __Random Forest(Base)__  | 0.78    | 0.92          |  0.84       |  0.81       |0.82     |
| __Random Forest(Tuned)__ | 0.74    | 0.94          | 0.79        | 0.81        | 0.77    |
| __Logistic Regression(Base)__  | 0.82    | 0.95          |  0.88       |  0.86     |0.86  |
| __Logistic Regression (Tuned)__ | 0.82    | 0.95          | 0.88        | 0. 86   | 0.85 |
| __Naive Bayes(Base)__  | 0.71    | 0.93          |  0.78       |  0.81       |0.62    |
| __Naive Bayes (Tuned)__| 0.71    | 0.93          | 0.78        | 0.81        | 0.62 |
| __Linear SVC(Base)__  | 0.83    | 0.95          |  0.88       |  0.86       |0.86|
| __Linear SVC (Tuned)__ | 0.82    | 0.95          | 0.88        | 0.85        | 0.86|
| __XGBoost(Base)__  | 0.80    | 0.94          |  0.85       |  0.83       |0.84|
| __XGBoost(Tuned)__ | 0.77    | 0.91          | 0.83        | 0.80        | 0.81|

## 6. Citation <a name="ref"></a>
