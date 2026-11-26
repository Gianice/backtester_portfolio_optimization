import pandas as pd
from data.wide import load_wide_cached
from research.factors import extract_factors, PCA_WINDOW
from research.residuals import compute_residuals
from research.ou import fit_ou
from research.signals import update_positions
from research.sizing import size_positions



def run_one_date(returns, date, prev_pos=None):
    ''' 
    prev_pos: pd.Series -> prev position for all stock 
    '''

    # 1. window ending at T (nothing after T)
    win = returns.loc[:date].tail(PCA_WINDOW)
    # print("window:", win.index[0], "->", win.index[-1], win.shape)

    # 2. factors, then keep only the tickers it kept
    F, loadings, w, k_info = extract_factors(win, 1)
    win = win.loc[:, loadings.index]
    # print("F:", F.shape, "| tickers kept:", win.shape[1], "| k_info:", k_info)

    # 3. residuals
    alpha, betas, residual, cumres = compute_residuals(win, F)
    # print("betas:", betas.shape, "| cumres:", cumres.shape)

    # 4. OU
    ou = fit_ou(cumres)
    # print("tradeable:", ou["tradeable"].sum(), "/", len(ou))

    # 5. positions (first day, so nothing held yet)
    if prev_pos is None:
        prev_pos = pd.Series(0.0, index = cumres.columns)
    pos = update_positions(prev_pos.reindex(cumres.columns, fill_value =0.0), ou["s"], ou["tradeable"])
    # print(pos.value_counts())

    # 6. weights
    weights = size_positions(pos, betas)
    # print("sum w:", weights.sum(), "| gross:", weights.abs().sum())
    # print(weights[weights != 0])
    return pos, weights, int(ou["tradeable"].sum())