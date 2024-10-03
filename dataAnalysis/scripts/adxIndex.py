import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ADXIndex:
    def __init__(self, **kwargs) -> None:
        self.smoothing: float = 2. 
        self.periods: int = 14
        self.__dict__.update(kwargs)

        # Private field (operating fields)
        self._historical_offset: int = 1
        self._running_offset: int = 2 * self.periods
        self._stock_price: pd.DataFrame = None
        self._begin_date: np.datetime64 = None
        self._begin_index: int = None
        self._analysis_period: int = None  
        self._move_up: np.ndarray = None
        self._move_down: np.ndarray = None
        self._direction_up: np.ndarray = None
        self._direction_down: np.ndarray = None
        self._true_range: np.ndarray = None

    def __is_sufficient_period(self):
        if self._begin_index < self.periods + self._historical_offset:
            raise f"To short historical date (date - {self.periods + self._historical_offset})"

    def __find_begin_index(self) -> None:
        self._begin_index = np.sum(self._stock_price.index < self._begin_date)

    def __excision_data(self) -> None:
        self._stock_price = self._stock_price[self._begin_index - self._running_offset - self._historical_offset :
                                              self._begin_index + self._analysis_period]

    def __check_and_prepare_data(self) -> None:
        self.__find_begin_index()
        self.__is_sufficient_period()
        self.__excision_data()

    def __trueRange(self) -> None:
        self._true_range = np.zeros((3,self._stock_price.shape[0] - 1))
        self._true_range[0, :] = (self._stock_price["High"].to_numpy()-self._stock_price["Low"].to_numpy())[1:]
        self._true_range[1, :] = abs(self._stock_price["High"].to_numpy()[1:]-self._stock_price["Close"].to_numpy()[:-1])
        self._true_range[2, :] = abs(self._stock_price["Low"].to_numpy()[1:]-self._stock_price["Close"].to_numpy()[:-1])
        self._true_range = np.max(self._true_range, axis=0)

    def __calculate_move(self) -> None:
        self._move_up   = self._stock_price["High"].to_numpy()[1:] - self._stock_price["High"].to_numpy()[:-1]
        self._move_down = self._stock_price["Low"].to_numpy()[:-1] - self._stock_price["Low"].to_numpy()[1:]

    def __calculate_direction(self) -> None:
        # Use after __calculate_move
        self._direction_up   = np.all([self._move_up > self._move_down, self._move_up > 0],   axis=0) * self._move_up
        self._direction_down = np.all([self._move_down > self._move_up, self._move_down > 0], axis=0) * self._move_down

    def __simple_moving_average(self, x: np.ndarray) -> np.ndarray:
        sma = np.zeros(x.shape)
        for i in range(x.shape[0]):
            sma[i] = np.mean(x[np.max([0,i-self.periods+1]) : i+1])
        return sma
    
    def __wilders_moving_avg(self, x: np.ndarray) -> np.ndarray:
        wilders = np.zeros(x.shape)
        wilders[:self.periods+1] = self.__simple_moving_average(x[:self.periods+1])
        for i in range(self.periods, x.shape[0]):
            wilders[i] = (x[i] + wilders[i-1] * (self.periods - 1))/self.periods
        return wilders

    
    def __exponential_moving_average(self, x: np.ndarray) -> np.ndarray:
        ema = np.copy(x)
        factor = self.smoothing / (1 + self.periods)
        ema[:self.periods] = self.__simple_moving_average(x[:self.periods])
        for i in range(self.periods, x.shape[0]):
            ema[i] = ema[i] * factor + ema[i-1] * (1 - factor)
        return ema
    
    def __set_data(self, stock_price: pd.DataFrame, begin_date: np.datetime64, analysis_period: int) -> None:
        self._stock_price = stock_price
        self._begin_date = begin_date
        self._analysis_period = analysis_period

    def __calculate_indexes(self) -> dict:
        indicators = {"ADX": [], "+DI": [], "-DI": []}
        
        self.__calculate_move()
        self.__calculate_direction()
        self.__trueRange()
        
        atr = self.__wilders_moving_avg(self._true_range)[self._running_offset:]
        adu = self.__wilders_moving_avg(self._direction_up)[self._running_offset:]
        add = self.__wilders_moving_avg(self._direction_down)[self._running_offset:]

        indicators["+DI"] = (100 * adu / atr)
        indicators["-DI"] = (100 * add / atr)
        indicators["ADX"] = 100 * self.__wilders_moving_avg(np.abs((indicators["+DI"] - indicators["-DI"]) /
                                                                   (indicators["+DI"] + indicators["-DI"])))

        indicators["+DI"] = indicators["+DI"]
        indicators["-DI"] = indicators["-DI"]
        indicators["ADX"] = indicators["ADX"]
        indicators["Indexes"] = self._stock_price.index[self._running_offset+self._historical_offset:]

        return indicators
    
    def get_index_range(self, stock_price: pd.DataFrame, begin_date: np.datetime64, analysis_period: int) -> dict:
        self.__set_data(stock_price, begin_date, analysis_period)
        self.__check_and_prepare_data()
        return self.__calculate_indexes()

    def get_index(self, stock_price: pd.DataFrame) -> dict:
        self._stock_price = stock_price
        self._begin_index = self._running_offset + self._historical_offset
        return self.__calculate_indexes()

    def cast_to_data_frame(self, indexes: dict) -> pd.DataFrame:
        return pd.DataFrame(index=indexes["Indexes"], 
                            columns=["ADX", "+DI", "-DI"],
                            data=np.array([indexes["ADX"], indexes["+DI"], indexes["-DI"]]).T)
    
    def data_frame_to_dict(self, df: pd.DataFrame) -> dict:
        return {"ADX": df["ADX"].to_numpy(), 
                "+DI": df["+DI"].to_numpy(), 
                "-DI": df["-DI"].to_numpy(), 
                "Indexes": df.index.to_numpy()}

    def plot(self, indicators: dict, stock: pd.DataFrame) -> None:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle("ADX index")
        ax1.plot(indicators["Indexes"], stock[np.all([stock.index >= indicators["Indexes"][0], 
                                            stock.index <= indicators["Indexes"][-1]], axis=0)]["Close"].to_numpy(),
                                            label="Close Value")
        ax1.grid()
        ax1.legend(loc = 'upper right')

        ax2.plot(indicators["Indexes"], indicators["ADX"], linewidth=0.7, 
                 color="grey", label="ADX")
        ax2.plot(indicators["Indexes"], indicators["+DI"], linestyle="dashed", 
                 linewidth=0.7, color="green", label="+DI" )
        ax2.plot(indicators["Indexes"], indicators["-DI"], linestyle="dashed", 
                 linewidth=0.7, color="red", label=" -DI")
        ax2.grid()
        ax2.legend(loc = 'upper right')
        ax2.tick_params(axis='x', rotation=60)

        return fig