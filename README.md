# PORTOFOLIO

## 04. [Survival Analysis - Driving Customer Lifetime Value Growth Through a 29% Reduction in Churn Risk](survival_analysis/survival_analysis.md)

### Introduction

A fitness center offers comprehensive facilities and training programs to help customers achieve their fitness goals. Currently, they are running an attractive promotional program by offering a 99% discount on the first month's membership fee for new sign-ups. The aims of this coupon program is to:
- increase member growth
- increase long term profit
- build a loyal customer

The marketing team wants to know customer lifetime value by understanding how long customers will stop visiting the fitness center using the survival analysis methodology. Churn is a tricky case. Although we say the customer is a churn already, there is still a chance to win back even old customers. Our marketing team definition for churn is if customers don't visit the fitness center in last six months, they assume that the customers are churn. It means, churn is for customers that don't visit for more than six months and not churn is for customers that still visit in last six months.

Information we need to know:
- subscription price: IDR 199.000
- gross profit: 55%
- first month discount: 99%


```python
info = {'price': 199000,
        'gross_profit': 0.55,
        'first_month_discount': 0.99}
```

### Objectives

The aim of this project is to understand:
- the customers behaviour after signup and using their 99% discount on the first month membership,
- how long customer will survive from churn,
- what is the customer lifetime value,
- what is the CLTV:CAC ratio, and
- does age influence the duration of survival.

### Methodology

The survival analysis has some steps to do. All variables we need are time to event where this variable explains how long until the event occurs, target variable where this variable explains whether or not the event occurs until the study ends (it's called right cencored), and other variables that has correlation with the target.

We'll make churn variable by using if logic. 1 if last_visit < (query_time - six months) else 0 where 1 is churn and 0 is not churn. The CLTV formula we adjust for this case:

```
CLTV = {Price * Gross \ Profit * Average \ Lifetime}
```

Where:
- Transaction means customers pays the bill (counted per each month)
- Price is the subscription price
- Gross Profit is a profit after all expenses
- Average Lifetime is the average duration before they do churn.

## 03. [A/B Testing - Strategic Recommendations for Customer Growth](ab_testing/ab_testing.md)

### Introduction

The marketing team makes a loyalty program strategy to increase sales. Member customers will get 13% discount if they purchase above IDR130. They try making conversion by triggering casual customers into member customers. The marketing team believe that if every customers become a member, the purchasing power and retention rate will increase. This firm has five branches. The marketing team implements exact same advertisement for each city.

The marketing team wants to know the effectiveness of the program using A/B testing that runs for one year. They want to know if there is a different for total repeat purchases between member and casual customers, for total purchases between member and casual customers, for average purchases between member and casual customers, and association between city and customers that decide to become members.

### Objectives

The aim of this project is to understand whether or not:
- There is a significant difference in total repeat purchases between member and casual customers.
- There is a significant difference in total purchases between member and casual customers.
- There is a significant difference in average purchases between member and casual customers.
- There is a significant association between city and customers that decide to become members.

### Methodology

Because we will do two types of testing, we will seperate them into two.

#### T-test

Assumption:
- The sample must be independent.
- The sample must normally distributed. If not, use Mann Whitney U test.
- The sample must be clean from outliers.

Null Hypothesis:
- Conceptual: There are no significant differences between group A and group B.
- Mathematical: The mean/median score for group A is equal to the mean/median score for group B.

Alternative Hypothesis:
- Conceptual: There is a significant difference between group A and group B.
- Mathematical: The mean/median score for group A is not equal to the mean/median score for group B.

#### Chi-square test

Assumption:
- There are two categorical variables.
- The sample are independent.
- The amount of expected value must be greater than 5.

Null Hypothesis:
- Conceptual: There are no significant association between variable X and variable Y.
- Mathematical: The observed frequencies is equal to the expected frequencies.

Alternative Hypothesis:
- Conceptual: There are significant associations between variable X and variable Y.
- Mathematical: The observed frequencies are not equal to the expected frequencies.


## 02. [Cohort Analysis - Optimize Customer Retention with Acquisition Time-Based Segmentation in Python](cohort_analysis/cohort_analysis.md)

### Introduction

Cohort analysis is one of the low complexity but high impact to the business decision. Like RFM analysis, cohort analysis also tries to segment customers based on their characteristics such as channel acquisition or their first time purchase. The group is called cohort. One of the most used variable is transaction date.

### Objectives

This analysis aims to find trend and pattern of customer behaviour based on the month they first time purchased. These objectives include:
- which cohort has the most new customers.
- how much percentage of retention and churn rate month to month.

### Methodology

For this analysis, we only need customer_id and transaction date where later we will do some data preprocessing to find their first time purchase and other metrics. Here are the steps to do cohort analysis:
- find the first time purchase for each customer. name it as cohort month.
- use like distinct function in order to make one customer if purchases many times at the same month, count as one time for transaction month.
- substract the transaction month with cohort month to get cohort index.
- make pivot table so we can see which cohort month has how many customers for each cohort index.

## 01. [RFM Analysis - Optimize Business Strategy using Customer Data Analysis in Python](rfm_analysis/rfm_analysis.md)

### Introduction

RFM analysis is a method used to analyze customer behaviour using recency, frequency, and monetary where recency answers when is the latest purchase each customer, frequency answers how many times each customer buy products, and monetary answers how much money each customer spend to buy products. 

RFM analysis helps us to increase customer retention, optimize marketing campigns, and identify which customers has high value.

### Objectives

The Marketing team needs a customer segmentation so they can optimize business marketing strategy for each customer. We need to figure which customers:
- Need up-sell or cross-sell promotion to gain more purchase.
- Need loyalty program.
- Need educational content to build trust and knowledge or time-limited promotion.

### The Methodology

So far, we knew that we need are id_customer, transaction_date, and money_spent. Let's see the metrics we need and how to calculate it:
- To find the recency score, we need to calculate today date minus the newest date the customers bought. For example, customer 03 bought something in '03-04-2023' and today is '06-04-2024'. It means '06-04-2024' - '03-04-2024' = 3 days. The smaller day you get, the better you get score. that's recency.
- To find the frequency score, you just need to count how many times the customers buy. This must be the easiest calculation in RFM analysis.
- To find the monetary is sometimes tricky. Monetary means how much money they spent to buy our products. In this case, we use 'jumlah' variable (which means amount) multiple by 'harga_jual_satuan' (which means price per item).

After that, we will separate them into three categories for each metrics. Recency will get low, medium, and high category. Also the same with Frequency and Monetary. To make it easier to calculate, we will use 1 for low, 2 for medium, and 3 for high.

Finally, we will give weight for each metrics. In this example, we will give weight 20% for recency, 35% for frequency, and 45% for monetary. And then sum them up.



