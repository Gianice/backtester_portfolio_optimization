import numpy as np
import pandas as pd

from .base import Strategy

class MACrossover(Strategy):
    """
    Moving-average crossover.

    Fires +1 when the short MA crosses ABOVE the long MA, -1 when it crosses
    below. The crossover is detected as an EVENT by comparing today's
    relationship against yesterday's, not as a state.

    """
    
    def __init__(self, short_window: int = 20, long_window : int = 50):
        if short_window >= long_window:
            raise ValueError('short_window must be shorter than long_window')
        self.short_window = short_window
        self.long_window = long_window
        
    def __repr__(self):
        return f"MACrossover(short={self.short_window}, long={self.long_window})"
    
    # -- internals -----------------------------------------------------------
    
    def add_moving_average(self, df:pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.reset_index().sort_values(['ticker','real_date'])
        out['short_ma'] = out.groupby('ticker')['PX_LAST'].transform(lambda x: x.rolling(window = self.short_window).mean())
        out['long_ma'] = out.groupby('ticker')['PX_LAST'].transform(lambda x: x.rolling(window = self.long_window).mean())
        #x inside the lambda is a pandas series
            
            
        return out
    
    # -- public API ----------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        out = self._add_moving_averages(df)

        out['short_ma_ytd'] = out.groupby('ticker')['short_ma'].shift(1)
        out['long_ma_ytd'] = out.groupby('ticker')['long_ma'].shift(1)

        conditions = [
            (out['short_ma'] > out['long_ma'])
            & (out['short_ma_ytd'] <= out['long_ma_ytd']),

            (out['short_ma'] < out['long_ma'])
            & (out['short_ma_ytd'] >= out['long_ma_ytd']),
        ]
        choices = [1, -1]

        signals = np.select(conditions, choices, default=0)
        return pd.Series(signals, index=out.index, name='signal')

    def debug_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same computation, but returns the intermediate columns too.

        Handy when you want to eyeball why a crossover did or didn't fire.
        Not used by the engine.
        """
        out = self._add_moving_averages(df)
        out['short_ma_ytd'] = out.groupby('ticker')['short_ma'].shift(1)
        out['long_ma_ytd'] = out.groupby('ticker')['long_ma'].shift(1)
        out['signal'] = self.generate_signals(df)
        return out
    


    
