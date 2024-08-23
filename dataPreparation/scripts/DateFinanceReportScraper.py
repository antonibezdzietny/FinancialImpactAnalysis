"""
Data coming from https://biznes.pap.pl
"""
from .DateIter import DateIter
from .DateFinanceReportDB import DateFinanceReportDB
import requests
from bs4 import BeautifulSoup
from difflib import get_close_matches


class DateFinanceReportScraper:
    def __init__(self, 
                 di: DateIter, 
                 db: DateFinanceReportDB) -> None:
        self.setDatabase(db)
        self.setDateIter(di)
        self.__URL_RAW = "https://biznes.pap.pl/pl/reports/espi/term"


    def setDateIter(self, di: DateIter) -> None: 
        self._di = di

    def setDatabase(self, db: DateFinanceReportDB) -> None:
        self._db = db

    def scrapDates(self):
        self._di.reset()
        for _ in self._di:
            self.__parseDay()

    def __getFullURL(self, yy: int, mm: int, dd: int, pp: int) -> str:
        return f"{self.__URL_RAW},{yy},{mm},{dd},{pp}"
    
    def __getURLPage(self, url: str) -> BeautifulSoup:
        request = requests.get(url)
        html_code = BeautifulSoup(request.content, 'html.parser')
        return html_code
    
    def __insertToDatabase(self, company: str, title: str) -> None:
        date = self._di.getCurrentDate()
        report_type = ["roczny", "kwartalny"]
        close_matches = get_close_matches(title, report_type, n=1, cutoff=0.1)[0]
        if close_matches == "roczny":
            self._db.insert(company, DateFinanceReportDB.ReportType.ROCZNY, date)
        else:
            if 1 <= date[1] <= 3:
                self._db.insert(company, DateFinanceReportDB.ReportType.Q4, date)
                return
            if 4 <= date[1] <= 6:
                self._db.insert(company, DateFinanceReportDB.ReportType.Q1, date)
                return
            if 7 <= date[1] <= 9:
                self._db.insert(company, DateFinanceReportDB.ReportType.Q2, date)
                return
            if 10 <= date[1] <= 12:
                self._db.insert(company, DateFinanceReportDB.ReportType.Q3, date)
                return

    def __parseTable(self, html_code: BeautifulSoup) -> bool:
        espi_table = html_code.find("table", attrs={"class":"espi"})
        espi_rows = espi_table.find_all("tr", attrs={"class":"inf", "valign": "top"})
        for row in espi_rows:
            columns = row.find_all("td")
            company = columns[2].text.strip()
            title = columns[3].text.strip()
            self.__insertToDatabase(company, title)
        if len(espi_rows) < 20:
            return True
        return False

    def __parseDay(self):
        date = self._di.getCurrentDate()
        page = 1 
  
        while True:
            url = self.__getFullURL(date[2], date[1], date[0], page)
            html_code = self.__getURLPage(url)
            if self.__parseTable(html_code): # If parse is finish
                break
            page += 1