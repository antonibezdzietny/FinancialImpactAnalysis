from bs4 import BeautifulSoup
import requests
import numpy as np
from enum import Enum 
import pandas as pd

class FinancialReportScraper():
    """
    Class base on data coming from https://www.biznesradar.pl
    """
    class ReportType(Enum): 
        Q = "Q",
        R = "Y"

    MAIN_URL = "https://www.biznesradar.pl/"
    SUB_URL  = [
            "raporty-finansowe-rachunek-zyskow-i-strat/",
            "raporty-finansowe-bilans/",
            "raporty-finansowe-przeplywy-pieniezne/",
        ]

    ROW_TO_OMIT = [
        "Data publikacji", 
        "EBITDA", 
        "Wartość firmy", 
        "Aktywa z tytułu prawa do użytkowania",
        "Skup akcji", 
        "Płatności z tytułu umów leasingu",
        "Free Cash Flow",
    ]

    
    def __init__(self) -> None:
        self.__path = "../database/reports/"
        self.__resetTable()

    def __getURLPage(self, sub_url: str, ticker: str, type: ReportType) -> BeautifulSoup:
        return f"{self.MAIN_URL}{sub_url}{ticker},{type.value[0]}"
    
    def __getPage(self, url: str) -> BeautifulSoup:
        request = requests.get(url)
        html_code = BeautifulSoup(request.content, 'html.parser')
        return html_code
    
    def __parseHeaderDate(self, html_code: BeautifulSoup) -> None:
        report_table = html_code.find("table", attrs={"class":"report-table"})
        date_tr = report_table.find("tr") # Get first row (row with date)
        date_coll = date_tr.find_all("th", attrs={"class":"thq h"})
        self.date = {"Period": ["".join(date.text.split()).split('(')[0] for date in date_coll]}
        
    def __parseDataTable(self, html_code: BeautifulSoup) -> dict:
        report_table = html_code.find("table", attrs={"class":"report-table"})
        report_tr = report_table.find_all("tr")[1:] # Omit row with date
        for tr in report_tr:
            data = self.__parseTableRow(tr)
            if not data:
                continue

            self.value.update(data["Value"])
            self.dynamic.update(data["Dynamic"])
                     
    def __parseTableRow(self, tr):
        # Get row header
        row_header = tr.find("td", attrs={"class":"f"}).text.strip()

        if row_header in self.ROW_TO_OMIT:
            return None
        
        # Get value and dynamic
        values = []
        dynamics = []
        row_cells = tr.find_all("td", attrs={"class":"h"})[:-1]
        for cell in row_cells:
            value, dynamic = self.__parseCell(cell)
            values.append(value)
            dynamics.append(dynamic)

        return {"Value": {row_header: values}, "Dynamic": {row_header: dynamics}}
        
    def __parseCell(self, cell):
        data = cell.find_all("span", attrs={"class": "pv"})
        value_span = cell.find("span", attrs={"class": "value"})
        if value_span == None:
            value = 0
        else:
            value = float("".join(value_span.text.split()))

        dynamic_span = cell.find("div", attrs={"class", "changeyy"})
        if not dynamic_span == None:
            dynamic_span = dynamic_span.find("span", attrs={"class", "pv"})
            dynamic_str = (dynamic_span.text)[:-1]
            dynamic = float("".join(dynamic_str.split()))
        else:
            dynamic = np.nan
        return value, dynamic

    def __resetTable(self):
        self.value = {}
        self.dynamic = {}
        self.date = {}

    def __save(self, ticker: str, report_type: ReportType) -> None:
        path = f"{self.__path}{ticker}_{report_type.value[0]}_V.csv"
        self.value.to_csv(path)
        path = f"{self.__path}{ticker}_{report_type.value[0]}_D.csv"
        self.dynamic.to_csv(path)

    def __castToPD(self):
        data = {}
        data.update(self.date)
        data.update(self.value)
        self.value = pd.DataFrame(data).transpose()

        data = {}
        data.update(self.date)
        data.update(self.dynamic)
        self.dynamic = pd.DataFrame(data).transpose()

    def setSavePath(self, path: str):
        self.__path = path
    
    def parse(self, 
              ticker: str, 
              save: bool = True, 
              report_type: ReportType = ReportType.R):
        self.__resetTable()
        
        for sub_url in self.SUB_URL:
            url = self.__getURLPage(sub_url, ticker, report_type)
            html_code = self.__getPage(url)
            self.__parseHeaderDate(html_code)
            self.__parseDataTable(html_code)

        self.__castToPD()
        if save:
            self.__save(ticker, report_type)
