import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

#import the file on which association rules are to be applied
# df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
dataset = pd.read_excel('online_retail_data.xlsx')
print(dataset.head())



# Preprocessing of dataset - 

# remove unnecessary spaces from description
# remove unull invoice numbers
# remove 'C' from invoice number (invoice numbers with credit)
dataset['Description'] = dataset['Description'].str.strip()
dataset.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
dataset['InvoiceNo'] = dataset['InvoiceNo'].astype('str')
dataset = dataset[~dataset['InvoiceNo'].str.contains('C')]

# Add extra fields 
dataset['TotalAmount'] = dataset['Quantity'] * dataset['UnitPrice']
dataset['InvoiceYear'] = dataset['InvoiceDate'].dt.year
dataset['InvoiceMonth'] = dataset['InvoiceDate'].dt.month
dataset['InvoiceYearMonth'] = dataset['InvoiceYear'].map(str) + "-" + dataset['InvoiceMonth'].map(str)

print(dataset.head())

#basic computations on dataset
print(dataset.describe())


#Ques Total number of sales incurred by the company
print("\n\n-----------Total number of sales incurred by the company: -----------")
print(len(dataset['InvoiceNo'].unique()))


#Ques Total profit earned by the company
print("\n\n-----------Total profit earned by the company: -----------")
print(sum(dataset['TotalAmount']))


#Ques Top 20 customers based on the company
customers_amounts = dataset.groupby('CustomerID')['TotalAmount'].agg(np.sum).sort_values(ascending=False)
print("\n\n-----------Top 20 customers based on the shopping amount that they spent: -----------")
print(customers_amounts.head(20))

customers_amounts.head(20).plot.bar()
plt.show()


#Ques Frequently sold items by quantitiy
gp_stockcode = dataset.groupby('Description')
gp_stockcode_frq_quantitiy = gp_stockcode['Quantity'].agg(np.sum).sort_values(ascending=False)
print("\n\n-----------Frequently sold items by quantitiy: -----------")
print(gp_stockcode_frq_quantitiy.head(20))

gp_stockcode_frq_quantitiy.head(20).plot.bar()
plt.show()


#Ques Frequently sold items by total amount
gp_stockcode_frq_amount = gp_stockcode['TotalAmount'].agg(np.sum).sort_values(ascending=False)
print("\n\n-----------Frequently sold items by total amount: -----------")
print(gp_stockcode_frq_amount.head(20))

gp_stockcode_frq_amount.head(20).plot.bar()
plt.show()


# Ques Number of sales each month
gp_month = dataset.sort_values('InvoiceDate').groupby(['InvoiceYear', 'InvoiceMonth'])
gp_month_invoices = gp_month['InvoiceNo'].unique().agg(np.size)
print("\n\n-----------Number of sales each month: -----------")
print(gp_month_invoices)

gp_month_invoices.plot.bar()
plt.show()


# Ques Monthly earnings
gp_month_frq_amount= gp_month['TotalAmount'].agg(np.sum)
print("\n\n-----------Monthly earnings: -----------")
print(gp_month_frq_amount)

gp_month_frq_amount.plot.bar()
plt.show()


#Que Region(Country) wise sales
gp_country = dataset.groupby('Country')
print("\n\n-----------Regional sales: -----------")
print(gp_country['TotalAmount'].agg(np.sum).sort_values(ascending=False))


#Ques Number of active customers in each country
print("\n\n-----------Number of active customers in each country: -----------")
print(gp_country['CustomerID'].unique().agg(np.size).sort_values(ascending=False))


#Ques Get United Kingdom top 20 customers based on the total amount
uk_customers_amounts = dataset[dataset['Country']=='United Kingdom'].groupby('CustomerID')['TotalAmount'].agg(np.sum).sort_values(ascending=False)
print("\n\n-----------United Kingdom's top 20 customers based on the total amount: -----------")
print(uk_customers_amounts.head(20))

uk_customers_amounts.head(20).plot.bar()
plt.show()


#Ques United Kingdom frequently sold items by quantitiy
uk_gp_stockcode = dataset[dataset['Country']=='United Kingdom'].groupby('Description')
uk_gp_stockcode_frq_quantitiy = uk_gp_stockcode['Quantity'].agg(np.sum).sort_values(ascending=False)
print("\n\n-----------United Kingdom's frequently sold items by quantitiy: -----------")
print(uk_gp_stockcode_frq_quantitiy.head(20))

uk_gp_stockcode_frq_quantitiy.head(20).plot.bar()
plt.show()



#Ques United Kingdom frequently sold items by total amount
uk_gp_stockcode_frq_amount = uk_gp_stockcode['TotalAmount'].agg(np.sum).sort_values(ascending=False)
print("\n\n-----------United Kingdom's frequently sold items by total amount: -----------")
print(uk_gp_stockcode_frq_amount.head(20))

uk_gp_stockcode_frq_amount.head(20).plot.bar()
plt.show()



#----------------------APRIORI------------------------


#FRANCE
#consolidate the items into 1 transaction per row with each product 1 hot encoded: convert each category value into a new column and assigns a 1 or 0 (True/False)

basket_france = (dataset[dataset['Country'] =="France"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
print(basket_france.head())

#any positive values are converted to a 1 and anything less the 0 is set to 0
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

#remove the postage column
basket_france_sets = basket_france.applymap(encode_units)
basket_france_sets.drop('POSTAGE', inplace=True, axis=1)

#frequent item sets that have a support of at least 7%
frequent_itemsets_france = apriori(basket_france_sets, min_support=0.07, use_colnames=True)

#generate the rules with their corresponding support, confidence and lift
rules_france = association_rules(frequent_itemsets_france, metric="lift", min_threshold=1)
print(rules_france.head())



#----------------------CONCLUSION------------------------

#filtering the dataframe with large lift (6) and high confidence (.8):
print(rules_france[ (rules_france['lift'] >= 6) &
      (rules_france['confidence'] >= 0.8) ])

#how much opportunity there is to use the popularity of one product to drive sales of another
#(we can see that we sell 340 Green Alarm clocks but only 316 Red Alarm Clocks so maybe we can drive more Red Alarm Clock sales through recommendations)
print(basket_france['ALARM CLOCK BAKELIKE GREEN'].sum())
print(basket_france['ALARM CLOCK BAKELIKE RED'].sum())

#GERMANY
basket_germany = (dataset[dataset['Country'] =="Germany"]
           .groupby(['InvoiceNo', 'Description'])['Quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('InvoiceNo'))

basket_germany_sets = basket_germany.applymap(encode_units)
basket_germany_sets.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets_germany = apriori(basket_germany_sets, min_support=0.05, use_colnames=True)
rules_germany = association_rules(frequent_itemsets_germany, metric="lift", min_threshold=1)

print(rules_germany[ (rules_germany['lift'] >= 4) &
       (rules_germany['confidence'] >= 0.5)])










