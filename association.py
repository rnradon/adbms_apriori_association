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












