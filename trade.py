import numpy as np
import pandas as pd
import hvplot.pandas
from dotenv import load_dotenv
import os
import alpaca_trade_api as tradeapi
import math

class Trade:
    def __init__(self, start_amount=10000, number_of_tickers=1):
        self.start_amount = start_amount
        self.ticker_portion = round(start_amount/number_of_tickers,2)
    """ If to buy, we can only use start_amount/number_of_tickers at a time, or spend all
        If to sell, sell all shares
    """
    def spend_how_much( self, bal=0, price=0 ):
        if bal > self.ticker_portion:
            return self.ticker_portion
        elif bal/price >= 1: 
            return bal
        else:
            return 0
        
    def trade( self, on=pd.to_datetime('today').normalize(), do=0, ticker="", b_bal=0, shares=0 ):
        self.last_action_ticker = ticker
        status = 0
        a_bal = b_bal
        msg = 'success'
        action = 'none'
    
        if len(ticker) == 0:
            status = -1
            msg = 'no ticker'
        else:
            price = market_price(on, [ticker])
            spend = self.spend_how_much( b_bal, price )
            if price < 0:
                status = -1
                msg = 'market closed'
            elif do == 1 and spend == 0:
                state = -1
                msg = 'no money to buy 1'
            elif do == 1 :
                status = 0
                action = 'buy'
                shares = math.floor(spend/price)
                a_bal = round(a_bal - shares * price,2)
            elif do == -1 and shares == 0:
                status = -1
                msg = 'no share to sell'
            elif do == -1:
                status = 0
                action = 'sell'
                a_bal = round(a_bal + shares * price,2)
            else:
                status = -1
                msg = 'invalid action (1/-1 only)'
        return { 'action' : action,
                 'price' : price,
                 'bal' : a_bal,
                 'share' : shares,
                 'status' : status,
                 'msg' : msg }

def market_price( on, tickers ):
    load_dotenv()
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')

    # Create the Alpaca API object
    alpaca_api = tradeapi.REST(
       alpaca_api_key,
       alpaca_secret_key,
       api_version = 'v2'
    )
    
    # Set timeframe to "1Day" for Alpaca API
    timeframe = "1Day"
    start_end = pd.Timestamp(on,tz='America/New_York')
            
    # Get number_of_years' worth of historical data for tickers
    data_df = alpaca_api.get_bars(
        tickers,
        timeframe,
        start = start_end.isoformat(),
        end = start_end.isoformat()
    ).df
    if len(data_df) == 0:
        return -1
    return round( data_df.iloc[-1][['high','low','open','close']].sum()/4,2 )

def trade_action( instr, start_amount=10000.00 ):
    # 0: date, 1: ticker, 2: action
    trade = Trade(start_amount, len(set(instr.iloc[:,1])))
    bal = start_amount
    shares = {}      
    perf = pd.DataFrame()

    for row in instr.iterrows():
        one_trade = row[1]
        try:
            share = shares[one_trade[1]]
        except KeyError:
            share = 0
        
        traded = trade.trade(one_trade[0], 
                             do=one_trade[2], 
                             ticker=one_trade[1], 
                             b_bal=bal,
                             shares=share
                            ) 
        print(f"On {one_trade[0]} trade {one_trade[1]}")
        print(traded)
        if traded['status'] == 0:
            bal = traded['bal']
            try:
                if traded['action'] == 'buy':
                    shares[one_trade[1]] += traded['share']
                if traded['action'] == 'sell':
                    shares[one_trade[1]] -= traded['share']
            except KeyError:
                shares[one_trade[1]] = traded['share']
    
        share_worth = 0
        perf_one_row = pd.DataFrame([[one_trade[0], bal]])
        for key in shares:
            share_worth += round(shares[key] * traded['price'],2)
            perf_one_row[key] = shares[key]
        perf_one_row['networth'] = traded['bal'] + share_worth
        perf = pd.concat([perf, perf_one_row], join='outer')
        
    print(f"""networth: Cash: {traded['bal']}, 
              shares: {shares}, 
              total: {round(traded['bal'] + share_worth,2)}""")
    
    perf = perf.rename(columns = { 0:'date', 1:'cash' })
    perf = perf.set_index('date')
    share_plot = (perf.drop(['cash','networth'], axis=1)).hvplot(shared_axes=False, 
                                                                 frame_width=325,
                                                                 ylabel='shares',
                                                                 title='shares in trading'
                                                                )
    money_plot = perf[['cash','networth']].hvplot(shared_axes=False, 
                                                  frame_width=325,
                                                  ylabel='dollars',
                                                  title='networth and cash'
                                                 )
    return share_plot + money_plot
