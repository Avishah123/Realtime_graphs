from django.shortcuts import render
import pandas as pd 
import numpy as np
from scipy.stats import norm
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
import plotly.graph_objs as go
from django.shortcuts import render
import pandas as pd

import plotly.express as px
from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objs as go
from django.shortcuts import render
import pandas as pd
import numpy as np
# Create your views here.


option_type1='p'
option_type2='c'

def index(request):
    return render(request, 'index.html')

def blackScholes(r, S, K, T, sigma, option_type="c"):
    """Calculate BS price of call/put for a list of strike prices (K)"""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "c":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "p":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'c' for Call or 'p' for Put.")

    return price

def delta_calc(r, S, K, T, sigma, option_type="c"):
    """Calculate delta of an option for a list of strike prices (K)"""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

    if option_type == "c":
        delta_values = norm.cdf(d1)
    elif option_type == "p":
        delta_values = -norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'c' for Call or 'p' for Put.")

    return delta_values

def gamma_calc(r, S, K, T, sigma, option_type="c"):
    """Calculate gamma of an option for a list of strike prices (K)"""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

    gamma_values = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return gamma_values

def vega_calc(r, S, K, T, sigma, option_type="c"):
    """Calculate vega of an option for a list of strike prices (K)"""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

    vega_values = S * norm.pdf(d1) * np.sqrt(T)

    return vega_values * 0.01

def theta_calc(r, S, K, T, sigma, option_type="c"):
    """Calculate theta of an option for a list of strike prices (K)"""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "c":
        theta_values = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "p":
        theta_values = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'c' for Call or 'p' for Put.")

    return theta_values / 365


def plotly_chart(request):
    # Generate random data for demonstration
    num_points = 100  # Number of data points
    random_data = {
        'X': np.random.rand(num_points),  # Replace with your desired data generation logic
        'Y': np.random.rand(num_points)   # Replace with your desired data generation logic
    }
    
    df_filtered1 = pd.DataFrame(random_data)

    trace = go.Scatter(x=df_filtered1['X'].values,
                       y=df_filtered1['Y'].values,
                       mode='markers',
                       name='CE BS delta',
                       marker=dict(color='blue', size=10))

    layout = go.Layout(title='Plotly Scatter Plot',
                       xaxis=dict(title='X-Axis Label'),
                       yaxis=dict(title='Y-Axis Label'))

    figure = go.Figure(data=[trace], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'graph1.html', context)    

def plotly_prices_chart(request):
    path = './option_chain.csv'
    df = pd.read_csv(path)

    # Assuming df_filtered1 and df_filtered2 are your DataFrames
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13
    option_type1 = 'p'  # Define your option types (call or put)
    option_type2 = 'c'

    combined_array1 = np.array([selected_df1['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    combined_array2 = np.array([selected_df2['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']

    trace_pe_prices = go.Scatter(x=df_filtered1['Strike_Price'],
                                y=df_filtered1['Price'],
                                mode='lines+markers',
                                name='PE BS prices',
                                marker=dict(symbol='circle', size=8, color='blue'))

    trace_ce_prices = go.Scatter(x=df_filtered2['Strike_Price'],
                                y=df_filtered2['Price'],
                                mode='lines+markers',
                                name='CE BS prices',
                                marker=dict(symbol='x', size=10, color='orange'))

    vertical_line = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                               y=[min(df_filtered1['Price'].min(), df_filtered2['Price'].min()),
                                  max(df_filtered1['Price'].max(), df_filtered2['Price'].max())],
                               mode='lines',
                               name='ATM',
                               line=dict(color='red', dash='dash'))

    layout = go.Layout(title='Plotly Line Plot with Vertical Line',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='Prices'))

    figure = go.Figure(data=[trace_pe_prices, trace_ce_prices, vertical_line], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'prices.html', context)

def plotly_delta_chart(request):
    path = './option_chain.csv'
    df = pd.read_csv(path)

    # Assuming df_filtered1 and df_filtered2 are your DataFrames
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13
    option_type1 = 'c'  # Define your option types (call or put)
    option_type2 = 'p'

    combined_array1 = np.array([selected_df1['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])

    combined_array2 = np.array([selected_df2['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 2] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 2] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']

    trace_ce_delta = go.Scatter(x=df_filtered1['Strike_Price'],
                                y=df_filtered1['Delta'],
                                mode='lines+markers',
                                name='CE BS delta',
                                marker=dict(symbol='circle', size=8, color='blue'))

    trace_pe_delta = go.Scatter(x=df_filtered2['Strike_Price'],
                                y=df_filtered2['Delta'],
                                mode='lines+markers',
                                name='PE BS delta',
                                marker=dict(symbol='x', size=10, color='orange'))

    vertical_line = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                               y=[min(df_filtered1['Delta'].min(), df_filtered2['Delta'].min()),
                                  max(df_filtered1['Delta'].max(), df_filtered2['Delta'].max())],
                               mode='lines',
                               name='ATM',
                               line=dict(color='red', dash='dash'))

    layout = go.Layout(title='Delta Plot',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='Delta'))

    figure = go.Figure(data=[trace_ce_delta, trace_pe_delta, vertical_line], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'delta.html', context)

def plotly_gamma_plot(request):
    path = './option_chain.csv'
    df = pd.read_csv(path)

    # Assuming df_filtered1 and df_filtered2 are your DataFrames
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13
    option_type1 = 'c'  # Define your option types (call or put)
    option_type2 = 'p'

    combined_array1 = np.array([selected_df1['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    
    combined_array2 = np.array([selected_df2['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']

    trace_ce_gamma = go.Scatter(x=df_filtered1['Strike_Price'],
                                y=df_filtered1['Gamma'],
                                mode='lines+markers',
                                name='CE BS gamma',
                                marker=dict(symbol='circle', size=8, color='blue'))

    trace_pe_gamma = go.Scatter(x=df_filtered2['Strike_Price'],
                                y=df_filtered2['Gamma'],
                                mode='lines+markers',
                                name='PE BS gamma',
                                marker=dict(symbol='x', size=10, color='orange'))

    vertical_line = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                               y=[min(df_filtered1['Gamma'].min(), df_filtered2['Gamma'].min()),
                                  max(df_filtered1['Gamma'].max(), df_filtered2['Gamma'].max())],
                               mode='lines',
                               name='ATM',
                               line=dict(color='red', dash='dash'))

    layout = go.Layout(title='Gamma Plot',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='Gamma'))

    figure = go.Figure(data=[trace_ce_gamma, trace_pe_gamma, vertical_line], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'gamma.html', context)

def plotly_vega_chart(request):
    path ='./option_chain.csv'
    df = pd.read_csv(path)
    # print(df)
    
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    print(df_filtered1)
    print(df)

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']

    trace_ce_vega = go.Scatter(x=df_filtered1.iloc[:, 0].values,
                               y=df_filtered1.iloc[:, 4].values,
                               mode='lines+markers',
                               name='CE BS vega',
                               marker=dict(symbol='circle', size=8, color='blue'))

    trace_pe_vega = go.Scatter(x=df_filtered2.iloc[:, 0].values,
                               y=df_filtered2.iloc[:, 4].values,
                               mode='lines+markers',
                               name='PE BS vega',
                               marker=dict(symbol='x', size=10, color='orange'))

    vertical_line = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                               y=[min(df_filtered1.iloc[:, 4].values.min(), df_filtered2.iloc[:, 4].values.min()),
                                  max(df_filtered1.iloc[:, 4].values.max(), df_filtered2.iloc[:, 4].values.max())],
                               mode='lines',
                               name='ATM',
                               line=dict(color='red', dash='dash'))

    layout = go.Layout(title='Vega Plot',
                       xaxis=dict(title='X-Axis Label'),
                       yaxis=dict(title='Y-Axis Label'))

    figure = go.Figure(data=[trace_ce_vega, trace_pe_vega, vertical_line], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'vega.html', context)


def plotly_CE_gxoi_chart(request):
    # Assuming result_df1, atm_strike, and max_strike_price1 are your data
    path ='./option_chain.csv'
    df = pd.read_csv(path)
       
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    
    OIdf1 = selected_df1[selected_df1['strikeprice'].isin(df_filtered1.iloc[:, 0].values)]
    OIdf2 = selected_df2[selected_df2['strikeprice'].isin(df_filtered2.iloc[:, 0].values)]

    result_df1 = pd.merge(OIdf1, df_filtered1, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df1['GxOI'] = result_df1['openinterest'] * result_df1['Gamma']
    result_df1['VxOI'] = result_df1['openinterest'] * result_df1['Vega']

    result_df2 = pd.merge(OIdf2, df_filtered2, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df2['GxOI'] = result_df2['openinterest'] * result_df2['Gamma']
    result_df2['VxOI'] = result_df2['openinterest'] * result_df2['Vega']


    max_strike_price1 = result_df1.loc[result_df1['GxOI'].idxmax(), 'strikeprice']
    max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']
    
    trace_ce_gxoi = go.Scatter(x=result_df1['strikeprice'],
                               y=result_df1['GxOI'],
                               mode='lines+markers',
                               name='CE GxOI',
                               marker=dict(symbol='circle', size=8, color='blue'))

    vertical_line_atm = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                                   y=[result_df1['GxOI'].min(), result_df1['GxOI'].max()],
                                   mode='lines',
                                   name='ATM',
                                   line=dict(color='red', dash='dash'))

    vertical_line_max_strike = go.Scatter(x=[max_strike_price1, max_strike_price1],
                                         y=[result_df1['GxOI'].min(), result_df1['GxOI'].max()],
                                         mode='lines',
                                         name='Max Strike Price',
                                         line=dict(color='green', dash='dash'))

    layout = go.Layout(title='CE GXOI',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='GxOI'))

    figure = go.Figure(data=[trace_ce_gxoi, vertical_line_atm, vertical_line_max_strike], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'ce_gxoi.html', context)


def plotly_pe_gxoi_chart(request):
    path ='./option_chain.csv'
    df = pd.read_csv(path)
       
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    
    OIdf1 = selected_df1[selected_df1['strikeprice'].isin(df_filtered1.iloc[:, 0].values)]
    OIdf2 = selected_df2[selected_df2['strikeprice'].isin(df_filtered2.iloc[:, 0].values)]

    result_df1 = pd.merge(OIdf1, df_filtered1, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df1['GxOI'] = result_df1['openinterest'] * result_df1['Gamma']
    result_df1['VxOI'] = result_df1['openinterest'] * result_df1['Vega']

    result_df2 = pd.merge(OIdf2, df_filtered2, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df2['GxOI'] = result_df2['openinterest'] * result_df2['Gamma']
    result_df2['VxOI'] = result_df2['openinterest'] * result_df2['Vega']


    max_strike_price1 = result_df1.loc[result_df1['GxOI'].idxmax(), 'strikeprice']
    max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']

    trace_pe_gxoi = go.Scatter(x=result_df2['strikeprice'],
                               y=result_df2['GxOI'],
                               mode='lines+markers',
                               name='PE GxOI',
                               marker=dict(symbol='circle', size=8))

    vertical_line_atm = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                                   y=[result_df2['GxOI'].min(), result_df2['GxOI'].max()],
                                   mode='lines',
                                   name='ATM',
                                   line=dict(color='red', dash='dash'))

    max_strike_price2 = result_df2['strikeprice'].max()
    vertical_line_max_strike = go.Scatter(x=[max_strike_price2, max_strike_price2],
                                          y=[result_df2['GxOI'].min(), result_df2['GxOI'].max()],
                                          mode='lines',
                                          name='Max Strike',
                                          line=dict(color='green', dash='dash'))

    layout = go.Layout(title='PE GXOI',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='GxOI'))

    figure = go.Figure(data=[trace_pe_gxoi, vertical_line_atm, vertical_line_max_strike], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'pe_gxoi.html', context)

def plotly_ce_vxoi_chart(request):
    path ='./option_chain.csv'
    df = pd.read_csv(path)
       
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    
    OIdf1 = selected_df1[selected_df1['strikeprice'].isin(df_filtered1.iloc[:, 0].values)]
    OIdf2 = selected_df2[selected_df2['strikeprice'].isin(df_filtered2.iloc[:, 0].values)]

    result_df1 = pd.merge(OIdf1, df_filtered1, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df1['GxOI'] = result_df1['openinterest'] * result_df1['Gamma']
    result_df1['VxOI'] = result_df1['openinterest'] * result_df1['Vega']

    result_df2 = pd.merge(OIdf2, df_filtered2, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df2['GxOI'] = result_df2['openinterest'] * result_df2['Gamma']
    result_df2['VxOI'] = result_df2['openinterest'] * result_df2['Vega']


    max_strike_price1 = result_df1.loc[result_df1['GxOI'].idxmax(), 'strikeprice']
    max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']

    trace_ce_vxoi = go.Scatter(x=result_df1['strikeprice'],
                               y=result_df1['VxOI'],
                               mode='lines+markers',
                               name='CE VxOI',
                               marker=dict(symbol='circle', size=8))

    vertical_line_atm = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                                   y=[result_df1['VxOI'].min(), result_df1['VxOI'].max()],
                                   mode='lines',
                                   name='ATM',
                                   line=dict(color='red', dash='dash'))

    max_strike_price1 = result_df1['strikeprice'].max()
    vertical_line_max_strike = go.Scatter(x=[max_strike_price1, max_strike_price1],
                                          y=[result_df1['VxOI'].min(), result_df1['VxOI'].max()],
                                          mode='lines',
                                          name='Max Strike',
                                          line=dict(color='green', dash='dash'))

    layout = go.Layout(title='CE VXOI',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='VxOI'))

    figure = go.Figure(data=[trace_ce_vxoi, vertical_line_atm, vertical_line_max_strike], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'ce_vxoi.html', context)

def plotly_pe_vxoi_chart(request):
    path ='./option_chain.csv'
    df = pd.read_csv(path)
       
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    
    OIdf1 = selected_df1[selected_df1['strikeprice'].isin(df_filtered1.iloc[:, 0].values)]
    OIdf2 = selected_df2[selected_df2['strikeprice'].isin(df_filtered2.iloc[:, 0].values)]

    result_df1 = pd.merge(OIdf1, df_filtered1, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df1['GxOI'] = result_df1['openinterest'] * result_df1['Gamma']
    result_df1['VxOI'] = result_df1['openinterest'] * result_df1['Vega']

    result_df2 = pd.merge(OIdf2, df_filtered2, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df2['GxOI'] = result_df2['openinterest'] * result_df2['Gamma']
    result_df2['VxOI'] = result_df2['openinterest'] * result_df2['Vega']


    max_strike_price1 = result_df1.loc[result_df1['GxOI'].idxmax(), 'strikeprice']
    max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']

    trace_pe_vxoi = go.Scatter(x=result_df2['strikeprice'],
                               y=result_df2['VxOI'],
                               mode='lines+markers',
                               name='CE VxOI',
                               marker=dict(symbol='circle', size=8))

    vertical_line_atm = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                                   y=[result_df2['VxOI'].min(), result_df2['VxOI'].max()],
                                   mode='lines',
                                   name='ATM',
                                   line=dict(color='red', dash='dash'))

    max_strike_price1 = result_df1['strikeprice'].max()
    vertical_line_max_strike = go.Scatter(x=[max_strike_price2, max_strike_price2],
                                          y=[result_df2['VxOI'].min(), result_df2['VxOI'].max()],
                                          mode='lines',
                                          name='Max Strike',
                                          line=dict(color='green', dash='dash'))

    layout = go.Layout(title='PE VXOI',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='VxOI'))

    figure = go.Figure(data=[trace_pe_vxoi, vertical_line_atm, vertical_line_max_strike], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return render(request, 'pe_vxoi.html', context)


# ------------------------------------------------------ x -----------------------------------------------------------------------
from .models import *

def prices_plot():
    option_chain_data = OptionGreeks.objects.all()
    df = pd.DataFrame(option_chain_data.values())
 
    # Assuming df_filtered1 and df_filtered2 are your DataFrames
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13
    option_type1 = 'p'  # Define your option types (call or put)
    option_type2 = 'c'

    combined_array1 = np.array([selected_df1['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    combined_array2 = np.array([selected_df2['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']

    trace_pe_prices = go.Scatter(x=df_filtered1['Strike_Price'],
                                y=df_filtered1['Price'],
                                mode='lines+markers',
                                name='PE BS prices',
                                marker=dict(symbol='circle', size=8, color='blue'))

    trace_ce_prices = go.Scatter(x=df_filtered2['Strike_Price'],
                                y=df_filtered2['Price'],
                                mode='lines+markers',
                                name='CE BS prices',
                                marker=dict(symbol='x', size=10, color='orange'))

    vertical_line = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                               y=[min(df_filtered1['Price'].min(), df_filtered2['Price'].min()),
                                  max(df_filtered1['Price'].max(), df_filtered2['Price'].max())],
                               mode='lines',
                               name='ATM',
                               line=dict(color='red', dash='dash'))

    layout = go.Layout(title='Plotly Line Plot with Vertical Line',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='Prices'))

    figure = go.Figure(data=[trace_pe_prices, trace_ce_prices, vertical_line], layout=layout)

    context = {
        'plot_div': figure.to_html(full_html=False)
    }

    return figure.to_html(full_html=False)

def delta_plot():
    option_chain_data = OptionGreeks.objects.all()
    df = pd.DataFrame(option_chain_data.values())

    # Assuming df_filtered1 and df_filtered2 are your DataFrames
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13
    option_type1 = 'c'  # Define your option types (call or put)
    option_type2 = 'p'

    combined_array1 = np.array([selected_df1['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])

    combined_array2 = np.array([selected_df2['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 2] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 2] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']

    trace_ce_delta = go.Scatter(x=df_filtered1['Strike_Price'],
                                y=df_filtered1['Delta'],
                                mode='lines+markers',
                                name='CE BS delta',
                                marker=dict(symbol='circle', size=8, color='blue'))

    trace_pe_delta = go.Scatter(x=df_filtered2['Strike_Price'],
                                y=df_filtered2['Delta'],
                                mode='lines+markers',
                                name='PE BS delta',
                                marker=dict(symbol='x', size=10, color='orange'))

    vertical_line = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                               y=[min(df_filtered1['Delta'].min(), df_filtered2['Delta'].min()),
                                  max(df_filtered1['Delta'].max(), df_filtered2['Delta'].max())],
                               mode='lines',
                               name='ATM',
                               line=dict(color='red', dash='dash'))

    layout = go.Layout(title='Delta Plot',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='Delta'))

    figure = go.Figure(data=[trace_ce_delta, trace_pe_delta, vertical_line], layout=layout)

    return figure.to_html(full_html=False)

def gamma_plot():
    option_chain_data = OptionGreeks.objects.all()
    df = pd.DataFrame(option_chain_data.values())

    # Assuming df_filtered1 and df_filtered2 are your DataFrames
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13
    option_type1 = 'c'  # Define your option types (call or put)
    option_type2 = 'p'

    combined_array1 = np.array([selected_df1['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type2)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    
    combined_array2 = np.array([selected_df2['strikeprice'].values,
                                [round(x, 3) for x in blackScholes(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in delta_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in gamma_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 4) for x in vega_calc(r, S, K, T, sigma, option_type1)],
                                [round(x, 3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']

    trace_ce_gamma = go.Scatter(x=df_filtered1['Strike_Price'],
                                y=df_filtered1['Gamma'],
                                mode='lines+markers',
                                name='CE BS gamma',
                                marker=dict(symbol='circle', size=8, color='blue'))

    trace_pe_gamma = go.Scatter(x=df_filtered2['Strike_Price'],
                                y=df_filtered2['Gamma'],
                                mode='lines+markers',
                                name='PE BS gamma',
                                marker=dict(symbol='x', size=10, color='orange'))

    vertical_line = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                               y=[min(df_filtered1['Gamma'].min(), df_filtered2['Gamma'].min()),
                                  max(df_filtered1['Gamma'].max(), df_filtered2['Gamma'].max())],
                               mode='lines',
                               name='ATM',
                               line=dict(color='red', dash='dash'))

    layout = go.Layout(title='Gamma Plot',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='Gamma'))

    figure = go.Figure(data=[trace_ce_gamma, trace_pe_gamma, vertical_line], layout=layout)
   
    return figure.to_html(full_html=False)

def vega_plot():
    option_chain_data = OptionGreeks.objects.all()
    df = pd.DataFrame(option_chain_data.values())
        
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]
   
    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']

    trace_ce_vega = go.Scatter(x=df_filtered1.iloc[:, 0].values,
                               y=df_filtered1.iloc[:, 4].values,
                               mode='lines+markers',
                               name='CE BS vega',
                               marker=dict(symbol='circle', size=8, color='blue'))

    trace_pe_vega = go.Scatter(x=df_filtered2.iloc[:, 0].values,
                               y=df_filtered2.iloc[:, 4].values,
                               mode='lines+markers',
                               name='PE BS vega',
                               marker=dict(symbol='x', size=10, color='orange'))

    vertical_line = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                               y=[min(df_filtered1.iloc[:, 4].values.min(), df_filtered2.iloc[:, 4].values.min()),
                                  max(df_filtered1.iloc[:, 4].values.max(), df_filtered2.iloc[:, 4].values.max())],
                               mode='lines',
                               name='ATM',
                               line=dict(color='red', dash='dash'))

    layout = go.Layout(title='Vega Plot',
                       xaxis=dict(title='X-Axis Label'),
                       yaxis=dict(title='Y-Axis Label'))

    figure = go.Figure(data=[trace_ce_vega, trace_pe_vega, vertical_line], layout=layout)

   
    return figure.to_html(full_html=False)

def ce_gxoi_plot():
    # Assuming result_df1, atm_strike, and max_strike_price1 are your data
    option_chain_data = OptionGreeks.objects.all()
    df = pd.DataFrame(option_chain_data.values())
       
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    
    OIdf1 = selected_df1[selected_df1['strikeprice'].isin(df_filtered1.iloc[:, 0].values)]
    OIdf2 = selected_df2[selected_df2['strikeprice'].isin(df_filtered2.iloc[:, 0].values)]

    result_df1 = pd.merge(OIdf1, df_filtered1, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df1['GxOI'] = result_df1['openinterest'] * result_df1['Gamma']
    result_df1['VxOI'] = result_df1['openinterest'] * result_df1['Vega']

    result_df2 = pd.merge(OIdf2, df_filtered2, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df2['GxOI'] = result_df2['openinterest'] * result_df2['Gamma']
    result_df2['VxOI'] = result_df2['openinterest'] * result_df2['Vega']


    max_strike_price1 = result_df1.loc[result_df1['GxOI'].idxmax(), 'strikeprice']
    max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']
    
    trace_ce_gxoi = go.Scatter(x=result_df1['strikeprice'],
                               y=result_df1['GxOI'],
                               mode='lines+markers',
                               name='CE GxOI',
                               marker=dict(symbol='circle', size=8, color='blue'))

    vertical_line_atm = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                                   y=[result_df1['GxOI'].min(), result_df1['GxOI'].max()],
                                   mode='lines',
                                   name='ATM',
                                   line=dict(color='red', dash='dash'))

    vertical_line_max_strike = go.Scatter(x=[max_strike_price1, max_strike_price1],
                                         y=[result_df1['GxOI'].min(), result_df1['GxOI'].max()],
                                         mode='lines',
                                         name='Max Strike Price',
                                         line=dict(color='green', dash='dash'))

    layout = go.Layout(title='CE GXOI',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='GxOI'))

    figure = go.Figure(data=[trace_ce_gxoi, vertical_line_atm, vertical_line_max_strike], layout=layout)

    return figure.to_html(full_html=False)

def pe_gxoi_plot():
    option_chain_data = OptionGreeks.objects.all()
    df = pd.DataFrame(option_chain_data.values())
       
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    
    OIdf1 = selected_df1[selected_df1['strikeprice'].isin(df_filtered1.iloc[:, 0].values)]
    OIdf2 = selected_df2[selected_df2['strikeprice'].isin(df_filtered2.iloc[:, 0].values)]

    result_df1 = pd.merge(OIdf1, df_filtered1, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df1['GxOI'] = result_df1['openinterest'] * result_df1['Gamma']
    result_df1['VxOI'] = result_df1['openinterest'] * result_df1['Vega']

    result_df2 = pd.merge(OIdf2, df_filtered2, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df2['GxOI'] = result_df2['openinterest'] * result_df2['Gamma']
    result_df2['VxOI'] = result_df2['openinterest'] * result_df2['Vega']


    max_strike_price1 = result_df1.loc[result_df1['GxOI'].idxmax(), 'strikeprice']
    max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']

    trace_pe_gxoi = go.Scatter(x=result_df2['strikeprice'],
                               y=result_df2['GxOI'],
                               mode='lines+markers',
                               name='PE GxOI',
                               marker=dict(symbol='circle', size=8))

    vertical_line_atm = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                                   y=[result_df2['GxOI'].min(), result_df2['GxOI'].max()],
                                   mode='lines',
                                   name='ATM',
                                   line=dict(color='red', dash='dash'))

    # max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikePrice']

    max_strike_price2 =result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']
    vertical_line_max_strike = go.Scatter(x=[max_strike_price2, max_strike_price2],
                                          y=[result_df2['GxOI'].min(), result_df2['GxOI'].max()],
                                          mode='lines',
                                          name='Max Strike',
                                          line=dict(color='green', dash='dash'))

    layout = go.Layout(title='PE GXOI',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='GxOI'))

    figure = go.Figure(data=[trace_pe_gxoi, vertical_line_atm, vertical_line_max_strike], layout=layout)

    
    return figure.to_html(full_html=False)

def ce_vxoi_plot():
    option_chain_data = OptionGreeks.objects.all()
    df = pd.DataFrame(option_chain_data.values())
       
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    
    OIdf1 = selected_df1[selected_df1['strikeprice'].isin(df_filtered1.iloc[:, 0].values)]
    OIdf2 = selected_df2[selected_df2['strikeprice'].isin(df_filtered2.iloc[:, 0].values)]

    result_df1 = pd.merge(OIdf1, df_filtered1, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df1['GxOI'] = result_df1['openinterest'] * result_df1['Gamma']
    result_df1['VxOI'] = result_df1['openinterest'] * result_df1['Vega']

    result_df2 = pd.merge(OIdf2, df_filtered2, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df2['GxOI'] = result_df2['openinterest'] * result_df2['Gamma']
    result_df2['VxOI'] = result_df2['openinterest'] * result_df2['Vega']


    max_strike_price1 = result_df1.loc[result_df1['GxOI'].idxmax(), 'strikeprice']
    max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']

    trace_ce_vxoi = go.Scatter(x=result_df1['strikeprice'],
                               y=result_df1['VxOI'],
                               mode='lines+markers',
                               name='CE VxOI',
                               marker=dict(symbol='circle', size=8))

    vertical_line_atm = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                                   y=[result_df1['VxOI'].min(), result_df1['VxOI'].max()],
                                   mode='lines',
                                   name='ATM',
                                   line=dict(color='red', dash='dash'))

  
    vertical_line_max_strike = go.Scatter(x=[max_strike_price1, max_strike_price1],
                                          y=[result_df1['VxOI'].min(), result_df1['VxOI'].max()],
                                          mode='lines',
                                          name='Max Strike',
                                          line=dict(color='green', dash='dash'))

    layout = go.Layout(title='CE VXOI',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='VxOI'))

    figure = go.Figure(data=[trace_ce_vxoi, vertical_line_atm, vertical_line_max_strike], layout=layout)

    return figure.to_html(full_html=False)

def pe_vxoi_plot():
    option_chain_data = OptionGreeks.objects.all()
    df = pd.DataFrame(option_chain_data.values())
       
    selected_df1 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'CE')]
    selected_df2 = df[(df['expirydate'] == '25-Jan-2024') & (df['instrumenttype'] == 'PE')]

    atm_strike = selected_df1.loc[(selected_df1['strikeprice'] - selected_df1['underlyingvalue']).abs().idxmin()]

    r = 0.073
    S = atm_strike['strikeprice']
    K = selected_df1['strikeprice'].values
    T = 6 / 365
    sigma = 0.13

    combined_array1 = np.array([selected_df1['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type2)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type2)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type2)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type2)]])
    combined_array2 = np.array([selected_df2['strikeprice'].values, [round(x,3) for x in blackScholes(r, S, K, T, sigma, option_type1)], [round(x,3) for x in delta_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in gamma_calc(r, S, K, T, sigma, option_type1)], [round(x,4) for x in vega_calc(r, S, K, T, sigma, option_type1)], [round(x,3) for x in theta_calc(r, S, K, T, sigma, option_type1)]])

    df_greeks1 = pd.DataFrame(combined_array1.T)
    df_greeks2 = pd.DataFrame(combined_array2.T)

    # Drop rows where any entry is 0
    df_filtered1 = df_greeks1[df_greeks1.iloc[:, 1] != 0]
    df_filtered2 = df_greeks2[df_greeks2.iloc[:, 1] != 0]

    df_filtered1.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    df_filtered2.columns = ['Strike_Price', 'Price', 'Delta', 'Gamma', 'Vega', 'Theta']
    
    OIdf1 = selected_df1[selected_df1['strikeprice'].isin(df_filtered1.iloc[:, 0].values)]
    OIdf2 = selected_df2[selected_df2['strikeprice'].isin(df_filtered2.iloc[:, 0].values)]

    result_df1 = pd.merge(OIdf1, df_filtered1, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df1['GxOI'] = result_df1['openinterest'] * result_df1['Gamma']
    result_df1['VxOI'] = result_df1['openinterest'] * result_df1['Vega']

    result_df2 = pd.merge(OIdf2, df_filtered2, left_on='strikeprice', right_on='Strike_Price', how='left')
    result_df2['GxOI'] = result_df2['openinterest'] * result_df2['Gamma']
    result_df2['VxOI'] = result_df2['openinterest'] * result_df2['Vega']


    max_strike_price1 = result_df1.loc[result_df1['GxOI'].idxmax(), 'strikeprice']
    max_strike_price2 = result_df2.loc[result_df2['GxOI'].idxmax(), 'strikeprice']

    trace_pe_vxoi = go.Scatter(x=result_df2['strikeprice'],
                               y=result_df2['VxOI'],
                               mode='lines+markers',
                               name='CE VxOI',
                               marker=dict(symbol='circle', size=8))

    vertical_line_atm = go.Scatter(x=[atm_strike['strikeprice'], atm_strike['strikeprice']],
                                   y=[result_df2['VxOI'].min(), result_df2['VxOI'].max()],
                                   mode='lines',
                                   name='ATM',
                                   line=dict(color='red', dash='dash'))

    
    vertical_line_max_strike = go.Scatter(x=[max_strike_price2, max_strike_price2],
                                          y=[result_df2['VxOI'].min(), result_df2['VxOI'].max()],
                                          mode='lines',
                                          name='Max Strike',
                                          line=dict(color='green', dash='dash'))

    layout = go.Layout(title='PE VXOI',
                       xaxis=dict(title='Strike Price'),
                       yaxis=dict(title='VxOI'))

    figure = go.Figure(data=[trace_pe_vxoi, vertical_line_atm, vertical_line_max_strike], layout=layout)

    return figure.to_html(full_html=False)


def plotly_all_charts(request):
  
    plot_div1 = prices_plot()  
    plot_div2 = delta_plot()  
    plot_div3 = gamma_plot() 
    plot_div4 = vega_plot()  
    plot_div5 = ce_gxoi_plot() 
    plot_div6 = pe_gxoi_plot()  
    plot_div7 = ce_vxoi_plot()  
    plot_div8 = pe_vxoi_plot()

    context = {
        'plot_div1': plot_div1,
        'plot_div2': plot_div2,
        'plot_div3': plot_div3,
        'plot_div4': plot_div4,
        'plot_div5': plot_div5,
        'plot_div6': plot_div6,
        'plot_div7': plot_div7,
        'plot_div8': plot_div8,
    }

    return render(request, 'all_plots.html', context)



