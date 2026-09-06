import pandas as pd
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    A strategy turns price data into TRADE SIGNALS.

    Signal of the returned Series (unchanged from your original code):
        +1  -> entry signal fired on this bar
        -1  -> exit signal fired on this bar
         0  -> nothing happened

    Note this is an EVENT, not a desired position. The engine decides what to
    do with it (e.g. ignore a +1 if already long). Moving to desired-position
    signal is a deliberate later change, not something to slip in during a
    refactor.

    Rules a Strategy must obey:
      - It only reads price/feature columns. Never cash, fills, costs or PnL.
      - It never applies execution lag. The engine owns that.
      - Row t may only use information available up to and including row t.
    """
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
    
    
    
    