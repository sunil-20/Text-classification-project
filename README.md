![image](https://github.com/sunil-20/Text-classification-project/blob/main/Images/cfpb.png)
---
# Table of Contents
1. [Background & Motivation](#background)
2. [Data Source](#data)
3. [EDA & Feature Engineering](#eda)
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

## 3. EDA & Feature Engineering <a name="eda"></a>
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
## 4. Models used <a name="model"></a>
### 4.1. Random Forest<a name="rf"></a>
### 4.2. Logistic Regression <a name="lg"></a>
### 4.3. Multinomial Naive bayes<a name="nb"></a>
### 4.4. Linear SVC <a name="svc"></a>
### 4.5. XGBoost <a name="xgb"></a>

## 5. Conclusion <a name="result"></a>

## 6. Citation <a name="ref"></a>
