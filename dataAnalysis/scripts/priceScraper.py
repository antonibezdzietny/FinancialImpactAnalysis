from abc import ABC, abstractmethod
from yfinance import Ticker 
from enum import Enum
import pandas as pd
import numpy as np

class StockPriceScraper(ABC):
    class _PeriodType(Enum):
        AFTER = 1
        BEFORE = 2

    def __init__(self, is_data_stored :bool = True) -> None:
        super().__init__()
        self._current_ticker : str = None
        self._historical_price : pd.DataFrame = None 
        # Used when is_data_stored == True 
        self._is_data_stored : bool = is_data_stored
        self._database = dict()

    def __setTicker(self, ticker : str) -> None:
        self._current_ticker = self.__prepareTicker(ticker)

    def __loadHistoricalPrice(self) -> None: 
        if self._current_ticker in self._database:
            self._historical_price = self._database[self._current_ticker]
            return
        self.__downloadAndPrepareHistoricalData()
        if self._is_data_stored:
            self.__saveToDatabase()            

    def __saveToDatabase(self) -> None:
        self._database[self._current_ticker] = self._historical_price

    def __downloadAndPrepareHistoricalData(self) -> None:
        self.__downloadHistoricalPrice()
        self.__preprocessingHistoricalPrice()

    @abstractmethod
    def __downloadHistoricalPrice(self) -> None:
        pass

    @abstractmethod
    def __preprocessingHistoricalPrice(self) -> None:
        pass

    @abstractmethod
    def __prepareTicker(self, ticker : str) -> str:
        pass

    def __getDateIndex(self, date : np.datetime64) -> int:
        if self._historical_price.empty:
            return None
        try:
            return self._historical_price.index.get_loc(date)
        except:
            return None

    def __getNHistoricalPrice(self, date: np.datetime64, 
                              offset: int, period: int, period_type: _PeriodType) -> pd.DataFrame:
        index = self.__getDateIndex(date)

        # Return empty data frame if no date in data frame
        if index == None:
            return pd.DataFrame([])
        # Select data
        if period_type == StockPriceScraper._PeriodType.AFTER:
            selected_pd = self._historical_price.iloc[index+offset : index+offset+period]
        else:
            selected_pd = self._historical_price.iloc[index-offset-period+1 : index-offset+1]
        # Return empty when database not have all days
        if selected_pd.shape[0] != period:
            return pd.DataFrame([])
        # Return selected data
        return selected_pd    

    def getNHistoricalPriceAfter(self, ticker: str, date: np.datetime64, 
                                 offset: int, period: int) -> pd.DataFrame:
        self.__setTicker(ticker)
        self.__loadHistoricalPrice()
        return self.__getNHistoricalPrice(date, offset, period, 
                                   StockPriceScraper._PeriodType.AFTER)

    def getNHistoricalPriceBefore(self, ticker: str, date: np.datetime64, 
                                  offset: int, period: int) -> pd.DataFrame:
        self.__setTicker(ticker)
        self.__loadHistoricalPrice()
        return self.__getNHistoricalPrice(date, offset, period, 
                                   StockPriceScraper._PeriodType.BEFORE)

    def loadDatabase(self, tickers: list[str]) -> None:
        for ticker in tickers:
            self.__setTicker(ticker)
            self.__loadHistoricalPrice()            


class CompanyStockPriceScraper(StockPriceScraper):
    def __init__(self, is_data_stored: bool = True) -> None:
        super().__init__(is_data_stored)
        self._yahoo_postfix : str = ".WA"
        self._yahoo_handler : Ticker = None

    def _StockPriceScraper__downloadHistoricalPrice(self) -> None:
        self._yahoo_handler = Ticker(self._current_ticker)
        self._historical_price = self._yahoo_handler.history(period="max", interval="1d")

    def _StockPriceScraper__preprocessingHistoricalPrice(self) -> None:
        if self._historical_price.empty:
            return
        # Remove UTC format
        self._historical_price.index = list(map(lambda x : np.datetime64(str(x)[:10]), 
                                               self._historical_price.index))

    def _StockPriceScraper__prepareTicker(self, ticker: str) -> str:
        return f"{ticker}{self._yahoo_postfix}"
        
        
class IndexStockPriceScraper(StockPriceScraper):
    PATH_2_HISTORICAL = "../database/indexHistorical/"
    PATH_2_WIGS = "../database/gpwCompaniesLists/WIGs.csv"

    def __init__(self, is_data_stored: bool = True) -> None:
        super().__init__(is_data_stored)
        self._wigs = pd.read_csv(IndexStockPriceScraper.PATH_2_WIGS)

    def _StockPriceScraper__downloadHistoricalPrice(self) -> None:
        self._historical_price = pd.read_csv(f"{IndexStockPriceScraper.PATH_2_HISTORICAL}{self._current_ticker}.csv")
    
    def _StockPriceScraper__preprocessingHistoricalPrice(self) -> None:
        if self._historical_price.empty:
            return
        self._historical_price["Data"] = self._historical_price["Data"].apply(lambda x : pd.to_datetime(np.datetime64(x)))
        self._historical_price.set_index("Data", inplace=True, drop=True)
        self._historical_price.index.names = ['Date']
        self._historical_price.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def _StockPriceScraper__prepareTicker(self, ticker: str) -> str:
        return self.getIndexTicker(ticker)
    
    def getIndexTicker(self, ticker : str) -> str:
        return self._wigs.columns[self._wigs.isin([ticker]).any()].to_list()[0]