"""
research/signals.py — Phase 4: turn s-scores into positions (direction only).

Output is +1 (long), -1 (short), 0 (flat) per stock. Sizing happens in Phase 5.

Positions have memory: whether you hold a stock today depends on whether
you held it yesterday. Open and close thresholds differ on purpose — the
gap between them stops daily flip-flopping, and every round trip costs money.
"""

import pandas as pd

S_OPEN = 1.25          # open when |s| > 1.25
S_CLOSE_LONG = 0.50    # close a long when s rises above -0.50
S_CLOSE_SHORT = 0.75   # close a short when s falls below +0.75 (earlier: borrow cost)



def update_positions(prev_pos: pd.Series, s: pd.Series, tradeable: pd.Series) -> pd.Series:
    """
    prev_pos  : (tickers,) yesterday's positions: +1 long, -1 short, 0 flat
    s         : (tickers,) today's s-scores from fit_ou
    tradeable : (tickers,) today's filter result from fit_ou

    Returns (tickers,) today's positions.
    """
    # 1. Align prev_pos to today's tickers. A ticker can appear or disappear
    #    between windows. Missing -> 0 (flat).
    #    Hint: prev_pos.reindex(s.index).fillna(0)
    # TODO: prev = ...

    # 2. Start today's positions from yesterday's.
    # TODO: pos = prev.copy()

    # 3. OPEN — only from flat, only if tradeable.
    #    flat AND tradeable AND s < -S_OPEN  -> +1  (cheap: buy)
    #    flat AND tradeable AND s >  S_OPEN  -> -1  (rich: short)
    #    Hint: build each condition as a boolean mask with &, then pos[mask] = 1
    #    All conditions check `prev`, NOT `pos`.
    # TODO

    # 4. CLOSE — check against `prev`, not `pos`.
    #    was long  (prev == 1)  AND s > -S_CLOSE_LONG   -> 0
    #    was short (prev == -1) AND s <  S_CLOSE_SHORT  -> 0
    # TODO

    # 5. FORCE CLOSE anything no longer tradeable (lost evidence of reversion).
    #    Hint: pos[~tradeable] = 0
    # TODO

    return pos


if __name__ == "__main__":
    # Toy test — every rule fires once. Work out the expected answer by hand first.
    tickers = ['A', 'B', 'C', 'D', 'E', 'F']
    prev = pd.Series([0,    0,    1,    -1,   1,     1  ], index=tickers)
    s    = pd.Series([-1.5, 1.5, -0.3,  0.5, -1.0, -0.9], index=tickers)
    ok   = pd.Series([True, True, True, True, False, True], index=tickers)

    print(update_positions(prev, s, ok))
    # Expected:
    # A  flat,  s=-1.5             -> open long    +1
    # B  flat,  s=+1.5             -> open short   -1
    # C  long,  s=-0.3 > -0.50     -> close         0
    # D  short, s=+0.5 < +0.75     -> close         0
    # E  long,  not tradeable      -> force close   0
    # F  long,  s=-0.9 (in band)   -> hold         +1