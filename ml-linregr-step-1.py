import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from scipy.stats import linregress

def desired_marketing_expenditure(marketing_expenditure, units_sold, desired_units_sold):
    """
    :param marketing_expenditure: (list) A list of integers with the expenditure for each previous campaign.
    :param units_sold: (list) A list of integers with the number of units sold for each previous campaign.
    :param desired_units_sold: (integer) Target number of units to sell in the new campaign.
    :returns: (float) Required amount of money to be invested.
    """
    m_e = np.array(marketing_expenditure).reshape(-1,1)
    print(m_e)
    u_s = np.array(units_sold)
    regr = lm.LinearRegression()
    regr.fit(m_e, u_s)
    predict = (desired_units_sold - regr.intercept_)/regr.coef_[0]
    # print(regr.predict(np.array(250000).reshape(-1,1)), regr.coef_, regr.intercept_)
  
    """ simple option - using SciPy linear regression """
    # res = linregress(m_e, u_s)
    # predict = res.intercept + res.slope * 250000
    # res = linregress(marketing_expenditure, units_sold)
    # predict = (desired_units_sold - res.intercept)/res.slope
    return predict


# print(desired_marketing_expenditure(
#     [300000, 200000, 400000, 300000, 100000],
#     [60000, 50000, 90000, 80000, 30000],
#     60000))

def most_corr(prices):
    """
    :param prices: (pandas.DataFrame) A dataframe containing each ticker's 
                   daily closing prices.
    :returns: (container of strings) A container, containing the two tickers that 
              are the most highly (linearly) correlated by daily percentage change.
    """
    changes = prices.pct_change()[1:]
    tickets = prices.columns.to_list()
    size = len(tickets)
    corr = np.zeros((size,size))
    corr_max = 0
    row_max = 0
    col_max = 0
    for row in range(size):
        for col in range(row,size):
            if col != row:
                corr[row,col] = changes[tickets[row]].corr(changes[tickets[col]])
                if corr[row,col] > corr_max:
                    corr_max = corr[row,col]
                    row_max = row
                    col_max = col
    return (tickets[row_max], tickets[col_max])

"""
print(most_corr(pd.DataFrame.from_dict({
    'GOOG' : [
        742.66, 738.40, 738.22, 741.16,
        739.98, 747.28, 746.22, 741.80,
        745.33, 741.29, 742.83, 750.50
    ],
    'FB' : [
        108.40, 107.92, 109.64, 112.22,
        109.57, 113.82, 114.03, 112.24,
        114.68, 112.92, 113.28, 115.40
    ],
    'MSFT' : [
        55.40, 54.63, 54.98, 55.88,
        54.12, 59.16, 58.14, 55.97,
        61.20, 57.14, 56.62, 59.25
    ],
    'AAPL' : [
        106.00, 104.66, 104.87, 105.69,
        104.22, 110.16, 109.84, 108.86,
        110.14, 107.66, 108.08, 109.90
    ]
})))
"""

def login_table(id_name_verified, id_password):
    """
    :param id_name_verified: (DataFrame) DataFrame with columns: Id, Login, Verified.   
    :param id_password: (numpy.array) Two-dimensional NumPy array where each element
                        is an array that contains: Id and Password
    :returns: (None) The function should modify id_name_verified DataFrame in-place. 
              It should not return anything.
    """   
    id_name_verified.drop(columns=['Verified'], inplace=True)
    df_pass = pd.DataFrame(id_password,columns=['Id','Password'])
    # df=id_name_verified.merge(df_pass[['Id','Password']],left_on='Id',right_on='Id')
    df=id_name_verified.join(df_pass.set_index("Id"),on="Id")
    id_name_verified['Password'] = df['Password']
    pass

# id_name_verified = pd.DataFrame([[1, "JohnDoe", True], [2, "AnnFranklin", False]], columns=["Id", "Login", "Verified"])
# id_password = np.array([[1, 987340123], [2, 187031122]], np.int32)
# login_table(id_name_verified, id_password)
# print(id_name_verified)

