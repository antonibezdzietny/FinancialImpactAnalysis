import numpy as np
import pandas as pd
import yfinance as yf

def castToDiffQuasiLog(report_value : pd.DataFrame) -> pd.DataFrame: 
    QUASI_LOG_VALUE = 1e-6
    new_index = report_value.index.to_numpy()[1:] # Without first row
    reports_np =  report_value.to_numpy()
    log_value = np.vectorize(lambda x : np.log10(np.max([x, QUASI_LOG_VALUE])))(reports_np[:,2:])
    log_diff = log_value[1:] - log_value[:-1]
    reports_np = reports_np[1:]
    reports_np[:, 2:] = log_diff
    return pd.DataFrame(data=reports_np, 
                        columns=report_value.columns, 
                        index=new_index)

def castDatabaseToQuasiLog(database):
    #Prepare data frame 
    log_diff_database = pd.DataFrame(columns=database.columns)

    # For all tickers 
    tickers = database["Ticker"].unique()
    for ticker in tickers:
        company_data = database[database["Ticker"] == ticker]
        company_log_diff = castToDiffQuasiLog(company_data)
        log_diff_database = pd.concat([log_diff_database, company_log_diff], ignore_index=False)

    return log_diff_database

class IndexReturn:
    PATH_2_HISTORICAL = "../database/indexHistorical/"
    
    def __init__(self) -> None:
        self.historical_data : pd.DataFrame = None
        self.price_type :str = "Close"

    def readHistorical(self, wig_ticker : str) -> None:
        self.historical_data = pd.read_csv(f"{IndexReturn.PATH_2_HISTORICAL}{wig_ticker}.csv")
        # Cast to yahoo finance format 
        self.historical_data["Data"] = self.historical_data["Data"].apply(lambda x : pd.to_datetime(np.datetime64(x)))
        self.historical_data.set_index("Data", inplace=True, drop=True)
        self.historical_data.index.names = ['Date']
        self.historical_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def getDailyReturn(self, t0 : np.datetime64, pre_offset : int, post_offset) -> pd.DataFrame: 
        if self.historical_data[(self.historical_data.index == t0)].empty:
            return  pd.DataFrame({"CompanyReturn": []})   
        
        historical = self.getHistoricalData(t0, pre_offset-1, post_offset)

        historical_price = historical[self.price_type].to_numpy()

        historical_date   = historical.index[1:]
        historical_return = (historical_price[1:] - historical_price[:-1]) / historical_price[:-1]

        return pd.DataFrame({"IndexReturn": historical_return},
                            index=historical_date)
        
    def getHistoricalData(self, t0 : np.datetime64, pre_offset : int, post_offset) -> pd.DataFrame:
        if self.historical_data[(self.historical_data.index == t0)].empty:
            return  pd.DataFrame({"CompanyReturn": []})
        
        post_historical = self.historical_data[(self.historical_data.index < t0)][-pre_offset:]
        pre_historical = self.historical_data[(self.historical_data.index >= t0)][:post_offset]
        return pd.concat([post_historical, pre_historical])
    

class CompanyReturn:
    POSTFIX = ".WA"
    PATH_2_WIGS = "../database/gpwCompaniesLists/WIGs.csv"

    def __init__(self):
        self.historical_data : pd.DataFrame = None
        self.price_type :str = "Close"
        self.wigs = pd.read_csv(CompanyReturn.PATH_2_WIGS)

    def readHistorical(self, company_ticker : str) -> None:
        yf_handler = yf.Ticker(company_ticker+CompanyReturn.POSTFIX)
        self.historical_data = yf_handler.history(period="max", interval="1d")
        self.historical_data.index = list(map(lambda x : np.datetime64(str(x)[:10]), 
                                              self.historical_data.index))

    def getDailyReturn(self, t0 : np.datetime64, pre_offset : int, post_offset) -> pd.DataFrame:
        if self.historical_data[(self.historical_data.index == t0)].empty:
            return  pd.DataFrame({"CompanyReturn": []})

        historical = self.getHistoricalData(t0, pre_offset-1, post_offset)

        historical_price = historical[self.price_type].to_numpy()

        historical_date   = historical.index[1:]
        historical_return = (historical_price[1:] - historical_price[:-1]) / historical_price[:-1]

        return pd.DataFrame({"CompanyReturn": historical_return},
                            index=historical_date)
    
    def getHistoricalData(self, t0 : np.datetime64, pre_offset : int, post_offset) -> pd.DataFrame:
        if self.historical_data[(self.historical_data.index == t0)].empty:
            return  pd.DataFrame({"CompanyReturn": []})
        
        post_historical = self.historical_data[(self.historical_data.index < t0)][-pre_offset:]
        pre_historical = self.historical_data[(self.historical_data.index >= t0)][:post_offset]
        return pd.concat([post_historical, pre_historical])
    
    def getIndexTick(self, company_ticker : str) -> str:
        return self.wigs.columns[self.wigs.isin([company_ticker]).any()].to_list()[0]