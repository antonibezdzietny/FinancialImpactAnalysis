from io import StringIO
import urllib.request
import pandas as pd
import numpy as np
import os.path


class StockPriceDatabase:
    def __init__(self, 
                 database_file_dir_: str = "../database/stockPrices/" ) -> None:
        self.database = dict()
        self.database_file_dir: str = database_file_dir_

    def __prepare_tickers(self, tickers: str | list[str]) -> list[str]:
        if type(tickers) is str:
            tickers = [tickers]
        return tickers

    def __is_stock_file_exist(self, ticker: str) -> bool:
        return os.path.isfile(f"{self.database_file_dir}{ticker}.csv")
    
    def __save_stock_file(self, ticker: str, stock_price: pd.DataFrame):
        stock_price.to_csv(f"{self.database_file_dir}{ticker}.csv")         
        
    def download_stock_price(self, ticker: str) -> pd.DataFrame:
        download_url = f'https://stooq.com/q/d/l/?s={ticker}&i=d'
        with urllib.request.urlopen(download_url) as f:
            html = f.read().decode('utf-8')

        if html == "No data":
            raise f"For {ticker} no data"
        
        csvStringIO = StringIO(html)
        return pd.read_csv(csvStringIO, sep=",", index_col=[0])    

    def update_locally_database(self, 
                                tickers: str | list[str], 
                                overwrite: bool = False) -> None:
        tickers = self.__prepare_tickers(tickers)  
        for ticker in tickers:
            if self.__is_stock_file_exist(ticker):
                if overwrite == False:
                    continue # If file exist and overwrite == False -> continue
            # Else download and save data
            stock_price = self.download_stock_price(ticker)
            self.__save_stock_file(ticker, stock_price)  

    def load_stock_price(self, 
                         tickers: str | list[str], 
                         first_read_locally: bool = True) -> None:
        
        tickers = self.__prepare_tickers(tickers)  
        for ticker in tickers: 
            if first_read_locally and self.__is_stock_file_exist(ticker):
                stock_price = pd.read_csv(f"{self.database_file_dir}{ticker}.csv", index_col=[0])
            else:
                stock_price = self.download_stock_price(ticker)
            self.database[ticker] = stock_price

    def get_stock_price(self, ticker: str) -> pd.DataFrame:
        if ticker not in self.database:
            self.load_stock_price(ticker)
        
        return self.database[ticker]
            
