class DateIter:
    def __init__(self) -> None:
        self.__MAX_DAY = 31
        self.__MAX_MONTH = 12
        self.__current = 0

    def setLimits(self, up_day: int, up_month: int, up_year: int,
                  down_day: int, down_month: int, down_year: int):
        self.__setUpLimits(up_day, up_month, up_year)
        self.__setDownLimits(down_day, down_month, down_year)

    def reset(self):
        self.__current = 0
        self.__setStartNumericDate()
        self.__setEndNumericDate()

    def getCurrentDate(self) -> list[int]:
        date_sum = self.__startNumericDate + self.__current
        return [date_sum%self.__MAX_DAY + 1, 
                    (date_sum//self.__MAX_DAY)%self.__MAX_MONTH + 1, 
                    date_sum//(self.__MAX_DAY * self.__MAX_MONTH)]

    def __setUpLimits(self, day: int, month: int, year: int) -> None:
        self.__up_limits ={"D" : day, "M" : month, "Y" : year}
        self.__setEndNumericDate()

    def __setDownLimits(self, day: int, month: int, year: int) -> None:
        self.__down_limits ={"D" : day, "M" : month, "Y" : year}
        self.__setStartNumericDate()

    def __setStartNumericDate(self):
        self.__startNumericDate = (self.__down_limits["Y"] * self.__MAX_DAY * self.__MAX_MONTH + \
            (self.__down_limits["M"] - 1) * self.__MAX_DAY + \
            self.__down_limits["D"] - 1) - 1
        
    def __setEndNumericDate(self):
        self.__endNumericDate = self.__up_limits["Y"] * self.__MAX_DAY * self.__MAX_MONTH + \
            (self.__up_limits["M"] - 1) * self.__MAX_DAY + \
            self.__up_limits["D"] - 1

    def __iter__(self):
        return self
    
    def __next__(self):
        self.__current += 1
        date_sum = self.__startNumericDate + self.__current
        if date_sum < self.__endNumericDate:
            return self.getCurrentDate()
        raise StopIteration