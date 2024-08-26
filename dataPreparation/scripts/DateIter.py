import numpy as np

class DateIter:
    def __init__(self, 
                 begin_date : np.datetime64 = np.datetime64('nat'), 
                 end_date   : np.datetime64 = np.datetime64('nat') ) -> None:
        self.setDateLimits(begin_date, end_date)
        self.reset()

    def setDateLimits(self, 
                  begin_date : np.datetime64, 
                  end_date   : np.datetime64) -> None:
        self._begin_date = begin_date
        self._end_date   = end_date

    def reset(self) -> None:
        self._current_date = self._begin_date
            
    def getCurrentDate(self) -> list[int]:
        year = self._current_date.astype('datetime64[Y]').astype(int) + 1970
        month = self._current_date.astype('datetime64[M]').astype(int) % 12 + 1
        day = (self._current_date - self._current_date.astype('datetime64[M]')).astype(int) + 1
        return [day, month, year]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return_date = self.getCurrentDate()
        if self._end_date - self._current_date >= np.timedelta64(0,'D'):
            self._current_date += 1
            return return_date
        raise StopIteration