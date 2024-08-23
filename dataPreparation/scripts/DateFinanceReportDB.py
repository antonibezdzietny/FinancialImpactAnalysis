from enum import Enum
import json
import pickle


class DateFinanceReportDB():
    class ReportType(Enum):
        ROCZNY = "Roczny",
        Q1 = "Q1",
        Q2 = "Q2",
        Q3 = "Q3",
        Q4 = "Q4",

    def __init__(self) -> None:
        self._finance_report_db = {}

    def __str__(self) -> str:
        return json.dumps(self._finance_report_db, indent=2, ensure_ascii=False)

    def __isCompanyExist(self, company_name: str) -> bool:
        if company_name in self._finance_report_db.keys():
            return True
        return False
    
    def __addCompany(self, company_name: str) -> None:
        self._finance_report_db[company_name] = { }
        for key in DateFinanceReportDB.ReportType:
            self._finance_report_db[company_name][key.value[0]] = []

    def __isReportYearExist(self, company_name: str, 
               report_type: ReportType,
               date: list[int]) -> bool:
        for row in self._finance_report_db[company_name][report_type.value[0]]:
            if date[2] in row:
                return True
        return False
    
    def __addReport(self, company_name: str, 
               report_type: ReportType,
               date: list[int]):
        self._finance_report_db[company_name][report_type.value[0]].append(date)
        

    def insert(self, company_name: str, 
               report_type: ReportType,
               date: list[int]) -> None:
        
        if not self.__isCompanyExist(company_name):
            self.__addCompany(company_name)

        if not self.__isReportYearExist(company_name, report_type, date):
            self.__addReport(company_name, report_type, date)

    def loadDatabase(self, path: str) -> None:
        with open(path, 'rb') as f:
            self._finance_report_db = pickle.load(f)

    def saveDatabase(self, database_path: str) -> None:
        with open(database_path, 'wb') as f:
            pickle.dump(self._finance_report_db, f)



            
        


    