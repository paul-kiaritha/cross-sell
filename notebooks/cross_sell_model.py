#!/usr/bin/env python
# coding: utf-8

# ### Cross Sell Model for Consumer and High Networth Clients

# Based on bio  and transactional data currently available from the core banking system, the following exploratory analysis has been carried out:

# In[1]:


# Initialization cells
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

np.set_printoptions(precision=2, suppress=True)

import warnings
warnings.filterwarnings('ignore')


# ### Product table as run using the prod_age_rec.sql query (fetch dat from the Enterprise Data Warehouse)
# The aim is to create a table that has product age, age of customer and a rating derived from dividing the two. 
# This helps in the ranking used in collaborative filtering which is the algorithm used in this case
# * cosine_similarity is used (i.e. how similar people are based on length of time products have been held)
# * 30 most similar customers are then picked and products preference is ranked from highest to lowest
# * Products held are then removed and the highest ranking products remaining, recommended to the customer

# In[2]:


# Data from prod_age_rec.sql
# other_table = pd.read_csv('RecData.txt',delimiter='|')
# prod_table = pd.read_csv('RecData.txt',delimiter='|')
prod_table = pd.read_csv('prod_age_chnw_jun22.csv')


# In[3]:


prod_table.head()


# Get the unique products held by the customers in Consumer and High Net Worth business unit

# In[4]:


prod_table['PROD_GRP'].unique(), prod_table['PROD_GRP'].nunique()


# In[6]:


#Drop the null values on the product column
prod_table.dropna(subset=['PROD_GRP'], inplace=True)
prod_table.head()


# In[7]:


prod_table.isna().sum()


# Visualize the distribution of product holdings in the segment

# In[8]:


sns.barplot(prod_table['PROD_GRP'].value_counts().index,prod_table['PROD_GRP'].value_counts().values)
plt.xticks(rotation=60)


# In[9]:


prod_table['PROD_GRP'].value_counts()


# Get the product rating by dividing the number of active years of the product with the age of customer
# * This allows for smoothing of the rating based on actual age

# In[10]:


# Drop the non eligible products fro CHNW 
prod_table = prod_table[prod_table['PROD_GRP'] != 'BUSINESS.LOAN']
prod_table = prod_table[prod_table['PROD_GRP'] != 'CORPORATE.LOAN']
prod_table = prod_table[prod_table['PROD_GRP'] != 'SAFRCOM.LN']


# In[11]:


# Get the product rating by dividing the number of product active years with the age of customer
prod_table['RATING_NM'] = prod_table['RATING'] 

# Map the products with numeric codes to aid with creation of the matrix (note the prod_id column should not have .0, \
# if it does the mapping has not been done correctly)
di = {'HOME.LOAN':1000,'BUSINESS.LOAN': 1001,'VAF.LOAN':1002,'IPF':1003, 'CURRENT.ACCOUNTS':1004, 'SAVINGS.ACCOUNTS':1005, 
      'FIXED.DEPOSIT':1006, 'MOBILE.DIGITAL.LENDING':1007, 'PERSONAL.LN':1008,'CALL.DEPOSIT':10010, 'TRADE':10011, 'SAFRCOM.LN':10012 }  

prod_table['PROD_ID'] = prod_table['PROD_GRP'].map(di)
prod_table.head()          


# #### Standardizing the output to cater for difference in product ages
# * To take care of difference in customer age and holding, the rating has been standardized by subtracting the mean product holding age
# * This allows for comparison of new customers and old existing customers as the average product holding age would be subtracted

# In[12]:


Mean = prod_table.groupby(by="CUSTOMER",as_index=False)['RATING_NM'].mean()
Prod_avg = pd.merge(prod_table,Mean,on='CUSTOMER')
Prod_avg['ADG_RATING']=Prod_avg['RATING_NM_x']-Prod_avg['RATING_NM_y']


# In[13]:


Prod_avg['PROD_ID'].unique(), Prod_avg.shape


# In[14]:


Prod_avg.head()


# ##### Create a table to be used to check existing product holdings of the customer

# In[15]:


check = pd.pivot_table(Prod_avg,values='RATING_NM_x',index='CUSTOMER',columns='PROD_ID')
check.head()


# ##### Create the final table with standardized ratings

# In[16]:


final = pd.pivot_table(Prod_avg,values='ADG_RATING',index='CUSTOMER',columns='PROD_ID')
final.head()


# To fill in the Nan values two approaches can be used
# * Product average (column-wise)
# * Customer average (row-wise)
# 
# Customer average is preferred as the user to user collaborative filtering will be used (user centric approach)

# In[17]:


# Replacing NaN by Prod Average - product similarity approach
# final_prod = final.fillna(final.mean(axis=0))

# Replacing NaN by user Average to fill in the matrix - user centric approach
final_customer = final.apply(lambda row: row.fillna(row.mean()), axis=1)


# In[18]:


# Drop rows that have zero values
final_customer = final_customer.loc[(final_customer !=0).any(axis=1)]


# In[19]:


final_customer.shape


# In[20]:


final_customer = final_customer[~final_customer.index.duplicated(keep='last')]
final_customer.query('CUSTOMER == 1124001')


# ### Cosine similarity

# **Cosine Similarity** measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction. In this case, the similarity based on product holding ratings.

# In[21]:


# user similarity on replacing NAN by user avg
b = cosine_similarity(final_customer)
np.fill_diagonal(b, 0 )
similarity_with_customer = pd.DataFrame(b,index=final_customer.index)
similarity_with_customer.columns=final_customer.index
similarity_with_customer.head()


# ### Getting similar users for product recommendation (kNN, k=30)
# The kNN algorithm will be used to get the 30 most similar customers. These will then be used to recommend products to the customer

# In[22]:


## Function to get similar customers
def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df


# In[24]:


# top 30 neighbours for each user
sim_customer_30_u = find_n_neighbours(similarity_with_customer,30)
sim_customer_30_u.head()


# ##### Get the customer product score

# In[25]:


def Customer_prod_score(cus,item):
    a = sim_customer_30_u[sim_customer_30_u.index==cus].values
    b = a.squeeze().tolist()
    c = final_customer.loc[:,item]
    d = c[c.index.isin(b)]
    f = d[d.notnull()]
    avg_user = Mean.loc[Mean['CUSTOMER'] == cus,'RATING_NM'].values[0]
    index = f.index.values.squeeze().tolist()
    corr = similarity_with_user.loc[cus,index]
    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['adg_score','correlation']
    fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
    nume = fin['score'].sum()
    deno = fin['correlation'].sum()
    final_score = avg_user + (nume/deno)
    return final_score


# Create a table of product holding

# In[26]:


Prod_avg = Prod_avg.astype({"PROD_ID": str})
Prod_user = Prod_avg.groupby(by = 'CUSTOMER')['PROD_ID'].apply(lambda x:','.join(x))
user = list(map(int,final_customer.index.values.tolist()))


# ##### Predicting the customer products

# In[ ]:


# Get products used by a specific user from the check DataFrame
def get_products_used_by_user(user, check):
    return check.columns[check.loc[user].notna()].tolist()

# Get products used by similar users based on sim_customer_30_u and Prod_user DataFrames
def get_similar_users_products(user, sim_customer_30_u, Prod_user):
    similar_users = sim_customer_30_u.loc[user].squeeze().tolist()
    products_used_by_similar_users = Prod_user[Prod_user.index.isin(similar_users)]
    products_used_by_similar_users = products_used_by_similar_users.values.flatten()
    return products_used_by_similar_users.tolist()

# Calculate recommendation scores for products
def calculate_recommendation_score(user, Prods_under_consideration, final_customer, Mean, similarity_with_customer):
    scores = []
    avg_user = Mean.loc[Mean['CUSTOMER'] == user, 'RATING_NM'].values[0]
    
    for item in Prods_under_consideration:
        user_ratings = final_customer[item]
        similar_users_ratings = user_ratings[user_ratings.index.isin(similar_users)]
        valid_ratings = similar_users_ratings.dropna()
        
        user_corr = similarity_with_customer.loc[user, valid_ratings.index]
        merged_data = pd.concat([valid_ratings, user_corr], axis=1)
        merged_data.columns = ['adg_score', 'correlation']
        
        merged_data['score'] = merged_data.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
        nume = merged_data['score'].sum()
        deno = merged_data['correlation'].sum()
        final_score = avg_user + (nume / deno)
        scores.append(final_score)
    
    return scores

# Get top recommended product names
def get_top_recommendations(Prods_under_consideration, scores, prod_table):
    data = pd.DataFrame({'PROD_ID': Prods_under_consideration, 'score': scores})
    top_recommendations = data.sort_values(by='score', ascending=False).head(5)
    recommendations_with_names = top_recommendations.merge(prod_table, how='inner', on='PROD_ID')
    recommended_product_names = recommendations_with_names['PROD_GRP'].values.tolist()
    return recommended_product_names

# Main function for calculating user-item scores and generating recommendations
def user_item_score(user, check, sim_customer_30_u, Prod_user, final_customer, Mean, similarity_with_customer, prod_table):
    Prod_used_by_user = get_products_used_by_user(user, check)
    similar_users = sim_customer_30_u.loc[user].squeeze().tolist()
    Prod_used_by_similar_users = get_similar_users_products(user, sim_customer_30_u, Prod_user)
    
    Prods_under_consideration = list(set(Prod_used_by_similar_users) - set(map(str, Prod_used_by_user)))
    Prods_under_consideration = list(map(int, Prods_under_consideration))
    
    scores = calculate_recommendation_score(user, Prods_under_consideration, final_customer, Mean, similarity_with_customer)
    
    recommended_product_names = get_top_recommendations(Prods_under_consideration, scores, prod_table)
    
    return recommended_product_names


# #### Function that checks on customers product holding and compares it to similar customer product holdings
# * It returns a list of products that are ranked highly per customer but that are not currently held
# * Refers to the table named check above

# In[ ]:


def Recommender(users_to_predict, check, sim_customer_30_u, Prod_user, final_customer, Mean, similarity_with_customer):
    """
    Generate product recommendations for a list of customers.

    Args:
        users_to_predict (list): List of customer numbers (CIF) to be predicted.
        check (DataFrame): DataFrame containing product usage information.
        sim_customer_30_u (DataFrame): DataFrame containing similarity scores between customers.
        Prod_user (DataFrame): DataFrame containing products used by customers.
        final_customer (DataFrame): DataFrame containing customer-product ratings.
        Mean (DataFrame): DataFrame containing customer average ratings.
        similarity_with_customer (DataFrame): DataFrame containing similarity scores between customers.

    Returns:
        DataFrame: Recommendations containing 'PROD_ID', 'score', and 'CUSTOMER'.
    """
    users_to_predict = list(map(int,users_to_predict))
    customer = []
    prod = []
    score = []

    for user in users_to_predict:
        # Get products used by the user and by similar users
        Prod_used_by_user = check.columns[check.loc[user].notna().any()].tolist()
        similar_users = sim_customer_30_u.loc[user].squeeze().tolist()
        Prod_used_by_similar_users = Prod_user[Prod_user.index.isin(similar_users)]
        Prods_under_consideration = list(set(Prod_used_by_similar_users.values.flatten()) - set(map(str, Prod_used_by_user)))
        Prods_under_consideration = list(map(int, Prods_under_consideration))

        for item in Prods_under_consideration:
            customer.append(user)
            prod.append(item)

            # Calculate recommendation scores for products
            user_ratings = final_customer[item]
            similar_users_ratings = user_ratings[user_ratings.index.isin(similar_users)]
            valid_ratings = similar_users_ratings.dropna()
            user_corr = similarity_with_customer.loc[user, valid_ratings.index]
            merged_data = pd.concat([valid_ratings, user_corr], axis=1)
            merged_data.columns = ['adg_score', 'correlation']

            # Calculate the score as the product of adg_score and correlation
            merged_data['score'] = merged_data.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
            nume = merged_data['score'].sum()
            deno = merged_data['correlation'].sum()

            # Calculate the final score for the item recommendation
            avg_user = Mean.loc[Mean['CUSTOMER'] == user, 'RATING_NM'].values[0]
            final_score = avg_user + (nume / deno)
            score.append(final_score)

    df = pd.DataFrame({'PROD_ID': prod, 'score': list(np.around(np.array(score), 8)), 'CUSTOMER': customer})
    return df


# In[29]:


# Get a list of all customer numbers
customers_pred = final_customer.index.values.tolist()


# In[30]:


# Use the recommender function above to return recommended products
data = Recommender(customers_pred, check, sim_customer_30_u, Prod_user, final_customer, Mean, similarity_with_customer)


# In[31]:


data.query("CUSTOMER==1124001")


# In[32]:


# Mapping of product IDs to product names
product_id_to_name = {
    1000: 'HOME.LOAN', 1001: 'BUSINESS.LOAN', 1002: 'VAF.LOAN', 1003: 'IPF',
    1004: 'CURRENT.ACCOUNTS', 1005: 'SAVINGS.ACCOUNTS', 1006: 'FIXED.DEPOSIT',
    1007: 'MOBILE.DIGITAL.LENDING', 1008: 'PERSONAL.LN', 10010: 'CALL.DEPOSIT',
    10011: 'TRADE', 10012: 'SAFRCOM.LN'
}

# Create a pivot table from the data DataFrame
pivot_table = pd.pivot_table(data, values='score', index='CUSTOMER', columns='PROD_ID')

# Map column indices (product IDs) to product names
pivot_table.columns = pivot_table.columns.map(product_id_to_name)

# Fill NaN values with 0
pivot_table.fillna(0, inplace=True)

# Print the pivot table
print(pivot_table.head())


# ##### A function that goes through the dataframe and returns the recommended products names (maps the prod_id to prod_name)

# In[33]:


def sort_df(df):
    """
    Sort the columns of a DataFrame in descending order and create a new DataFrame with sorted column labels.

    Args:
        df (DataFrame): Input DataFrame to be sorted.

    Returns:
        DataFrame: New DataFrame with sorted column labels.
    """
    sorted_tags = pd.DataFrame(index=df.index, columns=['prod_{}'.format(i) for i in range(df.shape[1])])

    for i in range(df.shape[0]):
        # Sort the values of the current row in descending order and get the corresponding column labels
        sorted_column_labels = df.iloc[i, :].sort_values(ascending=False).index
        sorted_tags.iloc[i, :] = list(sorted_column_labels)

    return sorted_tags


# In[34]:


final_pred = sort_df(pivot_table)


# In[35]:


final_pred.head()


# In[36]:


# Pick the highest ranked products
two_prod = final_pred[['prod_0']]
two_prod.head()


# In[37]:


print("Number of predictions are: {}".format(two_prod.shape[0]))


# In[38]:


two_prod['prod_0'].value_counts()


# In[39]:


two_prod.query("CUSTOMER==1124001")


# ### Save the data to the database table

# In[41]:


two_prod.reset_index(inplace=True)
two_prod = two_prod[['CUSTOMER', 'prod_0']]
import cx_Oracle
from sqlalchemy import types, create_engine 
for i in two_prod.select_dtypes(include=['O','int64']).columns.to_list():
    two_prod[i]=two_prod[i].astype(str)
dtyp = {c:types.VARCHAR(two_prod[c].str.len().max()+20)
        
        for c in two_prod.select_dtypes(include=['O','int64']).columns.to_list()}
print("write table to database")
oracle_sqlalchemy_engine = oracle_sqlalchemy_engine()
two_prod.to_sql(name='xxxxx', con=oracle_sqlalchemy_engine, if_exists='replace', index=False,dtype=dtyp)
print("finished writing")


# In[1]:


get_ipython().system(' pip install nbconvert')


# In[ ]:




