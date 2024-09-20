import numpy as np
import pandas as pd


class ReportIndexConversion:
    def __init__(self) -> None:
        pass

    # Wskaźniki rentowności
    @staticmethod
    def calculationReturnOnAssets(df : pd.DataFrame) -> np.ndarray:
        return (df["Zysk netto"] * 100 / (df["Aktywa trwałe"] + 
                                        df["Aktywa obrotowe"])).to_numpy()
    
    @staticmethod
    def calculationReturnOnEquity(df : pd.DataFrame) -> np.ndarray:
        return (df["Zysk netto"] * 100 / (df["Kapitał (fundusz) podstawowy"] + 
                                        df["Kapitał (fundusz) zapasowy"])).to_numpy()

    # Wskaźniki płynności
    @staticmethod
    def calculationCurrentRatio(df : pd.DataFrame) -> np.ndarray:
        return (df["Aktywa obrotowe"] / df["Zobowiązania krótkoterminowe"]).to_numpy()
    
    @staticmethod
    def calculationQuickRatio(df : pd.DataFrame) -> np.ndarray:
        return ((df["Aktywa obrotowe"] - df["Zapasy"])/df["Zobowiązania krótkoterminowe"]).to_numpy()

    @staticmethod
    def calculationCashRatio(df : pd.DataFrame) -> np.ndarray:
        return (df["Inwestycje krótkoterminowe"]/df["Zobowiązania krótkoterminowe"]).to_numpy()

    # Wskaźniki zadłużenia
    @staticmethod
    def calculationDebtRatio(df : pd.DataFrame) -> np.ndarray:
        return ((df["Zobowiązania długoterminowe"] + df["Zobowiązania krótkoterminowe"])/
                (df["Aktywa trwałe"] + df["Aktywa obrotowe"])).to_numpy()

    @staticmethod
    def calculationDebtToEquityRatio(df : pd.DataFrame) -> np.ndarray:
        return ((df["Zobowiązania długoterminowe"] + df["Zobowiązania krótkoterminowe"])/
                (df["Kapitał (fundusz) podstawowy"] + df["Kapitał (fundusz) zapasowy"])).to_numpy()

    # Wskaźniki efektywności (sprawności działania)
    @staticmethod
    def calculationEBITDA(df : pd.DataFrame) -> np.ndarray:
        return (df["Zysk operacyjny (EBIT)"] + df["Amortyzacja"]).to_numpy()

    @staticmethod
    def calculationEBIT(df : pd.DataFrame) -> np.ndarray:
        return df["Zysk operacyjny (EBIT)"].to_numpy()
    
    @staticmethod
    def castReportToIndexes(df : pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Data",
                "ReturnOnAssets", 
                "ReturnOnEquity",  
                "CurrentRatio", 
                "QuickRatio", 
                "CashRatio", 
                "DebtRatio", 
                "DebtToEquityRatio",
                "EBITDA",
                "EBIT"
                ], 
            data=np.array([
                df["Ticker"],
                df["Data"],
                ReportIndexConversion.calculationReturnOnAssets(df),
                ReportIndexConversion.calculationReturnOnEquity(df),
                ReportIndexConversion.calculationCurrentRatio(df),
                ReportIndexConversion.calculationQuickRatio(df),
                ReportIndexConversion.calculationCashRatio(df),
                ReportIndexConversion.calculationDebtRatio(df),
                ReportIndexConversion.calculationDebtToEquityRatio(df),
                ReportIndexConversion.calculationEBITDA(df),
                ReportIndexConversion.calculationEBIT(df)
                ]).T,
            index = df.index)