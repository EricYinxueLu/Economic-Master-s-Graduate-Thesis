import pandas as pd
import numpy as np
import os
from pathlib import Path
import statsmodels.api as sm
# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def safe_divide(a, b):
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)
def safe_lag(series, k=1):
    return series.shift(k)
# ------------------------------------------------------------------
# 1.  PULL PRICE INDICES -------------------------------------------
# ------------------------------------------------------------------
def get_ppi_index(series_path, data_path,
                  start_year=2002,
                  end_year=2022,
                  keep_blank_only=False,
                  max_value=1e4):
    # 1. Read series lookup
    lookup = pd.read_excel(series_path)

    # 2. Locate the real series‐ID, industry_code & product_code columns
    series_col = [c for c in lookup.columns
                  if 'series' in c.lower() and 'id' in c.lower()]
    if not series_col:
        raise ValueError("No 'series_id' column in Excel header")
    series_col = series_col[0]
    ic_col = [c for c in lookup.columns if c.startswith('industry_code')][0]
    pc_col = [c for c in lookup.columns if c.startswith('product_code')][0]
    # 3. Build SIC2 lookup
    lookup = (
        lookup
        .assign(
            series_id=lambda df: df[series_col],
            sic2=lambda df: pd.to_numeric(df[ic_col].astype(str).str[:2], errors='coerce'),
            product_code=lambda df: df[pc_col]
        )
        .dropna(subset=['sic2'])
    )
    if keep_blank_only:
        lookup = lookup.loc[lookup['product_code'].fillna('').str.strip() == '']
    lookup = lookup[['series_id','sic2']].drop_duplicates()
    if lookup.empty:
        raise ValueError("Lookup empty after filtering – check headers")
    # 4. Load PPI data
    if data_path.lower().endswith('.csv'):
        ppi_raw = pd.read_csv(data_path)
    else:
        ppi_raw = pd.read_csv(data_path, sep=r'\t', engine='python')
    # strip whitespace from column names
    ppi_raw.columns = ppi_raw.columns.str.strip()
    print("PPI file columns:", ppi_raw.columns.tolist())
    # 5. Restrict to your sample years & reasonable values
    ppi_raw = ppi_raw.loc[
        (ppi_raw['year'] >= start_year) &
        (ppi_raw['year'] <= end_year)
    ]
    if 'value' not in ppi_raw.columns:
        raise KeyError(f"'value' not found in PPI columns: {ppi_raw.columns.tolist()}")
    ppi_raw = ppi_raw.loc[
        (ppi_raw['value'] >= 20) &
        (ppi_raw['value'] <= max_value)
    ]
    # 6. Compute annual averages
    ppi_ann = (
        ppi_raw
        .groupby(['series_id','year'], as_index=False)['value']
        .mean()
        .rename(columns={'value':'PPI'})
    )
    # 7. Merge back to SIC2 & average
    ppi_ind = (
        ppi_ann
        .merge(lookup, on='series_id', how='inner')
        .groupby(['sic2','year'], as_index=False)['PPI']
        .mean()
    )
    if ppi_ind.empty:
        raise ValueError("No PPI left after merge – check your SIC codes")
    # rename year to fyear
    ppi_ind = ppi_ind.rename(columns={'year':'fyear'})
    print(f"Created PPI index: {len(ppi_ind)} rows across {ppi_ind['sic2'].nunique()} SIC-2s")
    return ppi_ind    
def get_cpi(cpi_path):
    cpi = pd.read_csv(cpi_path)
    cpi['fyear'] = cpi['observation_date'].str[:4].astype(int)
    return (
        cpi.groupby('fyear', as_index=False)['CPIAUCNS']
        .mean()
        .rename(columns={'CPIAUCNS': 'CPI'})
    )
def get_pk(pk_path):
    pk = pd.read_csv(pk_path)
    pk['fyear'] = pk['observation_date'].str[:4].astype(int)
    return pk.groupby('fyear', as_index=False)['Y033RD3A086NBEA']\
             .mean().rename(columns={'Y033RD3A086NBEA': 'Pk'})
def get_baa_yield(start_year, end_year):
    years = np.arange(start_year, end_year + 1)
    return pd.DataFrame({'fyear': years, 'baa_rate': 0.06})
# ------------------------------------------------------------------
def clean_compustat(df):
    numeric_cols = ['dltt','dlc','capx','ppegt','ppent','ceq','invt','sale',
                    'prcc_f','csho','dp','ib','ni','sstk','at','dvp','int_exp']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    sic = pd.to_numeric(df['sic'], errors='coerce')
    df = df[~(((sic >= 4900) & (sic <= 4999)) | ((sic >= 6000) & (sic <= 6999)))]
    df = df[(df['at'] > 2) & (df['ppegt'] > 1) & (df['sale'] > 0)]
    df['cusip'] = df['cusip'].astype(str).str.zfill(9)
    return df
# ------------------------------------------------------------------
def compute_K_rep(df_firm):
    K = [np.nan] * len(df_firm)
    if len(df_firm) == 0:
        return K
    K[0] = df_firm.iloc[0]['ppent']
    for i in range(1, len(df_firm)):
        if pd.isna(K[i-1]) or pd.isna(df_firm.iloc[i]['Pk']) or pd.isna(df_firm.iloc[i-1]['Pk']):
            K[i] = np.nan
        else:
            K[i] = K[i-1] * (df_firm.iloc[i]['Pk']/df_firm.iloc[i-1]['Pk']) + df_firm.iloc[i]['capx']
    return K
def compute_market_debt(df_firm, agg_dist):
    if len(df_firm) == 0:
        return []
 # Ensure interest expense column exists
    if 'int_exp' not in df_firm.columns:
        df_firm['int_exp'] = np.nan
    Tn = len(df_firm)
    D = np.zeros((Tn, 20))
    D[0, :] = df_firm.iloc[0]['dltt'] * agg_dist
    for t in range(1, Tn):
        LTD = df_firm.iloc[t]['dltt']
        D[t, :19] = D[t-1, 1:]
        D[t, 19] = max(LTD - D[t, :19].sum(), 0)
    NLTD = np.where(pd.isna(df_firm['int_exp']), 0.06*df_firm['dltt'], df_firm['int_exp'])
    disc = D.sum(axis=1) * (NLTD / np.maximum(df_firm['dltt'], 1e-6))
    return disc
def calculate_variables(data, ppi_ind, cpi_df, pk_df, baa_df):
    basePPI = ppi_ind['PPI'].min()
    df = data.copy()
    df['sic2'] = pd.to_numeric(df['sic'].astype(str).str[:2], errors='coerce')
    df = df.merge(ppi_ind, on=['sic2','fyear'], how='left')
    df = df.merge(cpi_df, on='fyear', how='left')
    df = df.merge(pk_df, on='fyear', how='left')
    df = df.merge(baa_df, on='fyear', how='left')
    df['cap_real'] = df['capx'] * (basePPI / df['PPI'])
    df['be_real']  = df['ceq']  * (basePPI / df['PPI'])
    df.sort_values(['gvkey','fyear'], inplace=True)
    df['inv_rep'] = df.groupby('gvkey')['invt'].transform(lambda x: x.ffill())
    df['K_rep']    = (
        df.groupby('gvkey')
          .apply(lambda g: pd.Series(compute_K_rep(g)))
          .reset_index(level=0, drop=True)
          .values
    )
    agg_dist = np.repeat(1/20, 20)
    df['MktDebt'] = df.groupby('gvkey') \
                  .apply(lambda g: pd.Series(compute_market_debt(g, agg_dist))) \
                  .reset_index(level=0, drop=True) \
                  .values
    # 0) market equity (price × shares plus a BAA-adjusted dividend liability)
    df['MktEquity'] = df['prcc_f'] * df['csho'] + df['dvp'] / df['baa_rate']

    # 1) build the lagged capital stock
    df['K_lag'] = df.groupby('gvkey')['K_rep'].shift(1)

    # 2) for each firm, fill first‐ever-year (e.g. 2003) with opening PPE
    df['K_denom'] = df['K_lag'].fillna(df['ppent'])

    # 3) compute Tobin’s Q and investment‐to‐capital _using_ K_denom
    df['Tobins_Q'] = (df['MktDebt'] + df['MktEquity'] - df['inv_rep']) / df['K_denom']
    df['invest_cap_sum'] = df['capx'] / df['K_denom']

    # 4) any other ratios you want on a PPE basis
    df['invest_cap'] = df['capx'] / df['ppent']
    df['cash_flow'] = (df['ni'] + df['dp']) / df['ppent']
    df['debt_to_asset'] = df['MktDebt'] / (df['MktDebt'] + df['MktEquity'])
    df['equity_issuance'] = df['sstk']
    df['ROA'] = df['ib'] / df['at']
    df['sale_lag'] = df.groupby('gvkey')['sale'].shift(1)
    df['sales_growth'] = (df['sale'] - df['sale_lag']) / df['sale_lag']

    # 5) build one‐year lags on everything you’ll regress
    grouped = df.groupby('gvkey')
    for col in ['Tobins_Q', 'invest_cap_sum', 'invest_cap', 'cash_flow', 'debt_to_asset', 'equity_issuance', 'ROA',
                'sales_growth']:
        df['lag_' + col] = grouped[col].shift(1)
        # ---------------------------------------------------------------
        # === Hellerstein-style robust cleaning (applied AFTER Whited) ===
        # ---------------------------------------------------------------
        from scipy import stats

        CORE_VARS = [
            'Tobins_Q', 'cash_flow', 'equity_issuance',
            'invest_cap_sum', 'debt_to_asset', 'ROA',
            'sales_growth', 'Disagreement', 'DisagreementInvFComp', 'DisagreementInvFCompDir'
        ]
        # 0) ------------- common, complete-case sample ------------------
        df = df.dropna(subset=CORE_VARS).copy()  # <- identical N for all core vars

        # 1) ------------- Hampel (median ± k·MAD) cap -------------------
        def hampel(s, k=2.9652):
            med = s.median()
            mad = (s - med).abs().median()
            lo, hi = med - k * mad, med + k * mad
            return s.clip(lo, hi)

        for v in CORE_VARS:
            df[v] = hampel(df[v])

        # 2) ------------- Yeo-Johnson on strictly-positive skewed vars ------
            from scipy.stats import yeojohnson
            POS_SKEW = ['Tobins_Q', 'equity_issuance']  # list of heavily skewed, numeric vars
            for v in POS_SKEW:
                # Yeo-Johnson handles zeros and negatives
                df[v], _ = yeojohnson(df[v].fillna(0))  # automatically picks λ to normalize
        # 3) ------------- 1 % / 99 % winsorisation ----------------------
        def winsor(s, p=0.01):
            lo, hi = s.quantile([p, 1 - p])
            return s.clip(lo, hi)

        for v in CORE_VARS:
            df[v] = winsor(df[v])
    return df


def build_panel(compu_path, disagree_path, ppi_series_path, ppi_data_path, cpi_path, pk_path):
    # Load raw Compustat data
    compu = pd.read_csv(compu_path)
    print(f"Initial data: {len(compu)} observations, year range: {compu['fyear'].min()}-{compu['fyear'].max()}")

    # Clean Compustat data
    compu = clean_compustat(compu)
    print(f"After basic cleaning: {len(compu)} observations")
    compu['sic2'] = pd.to_numeric(compu['sic'].astype(str).str[:2], errors='coerce')

    # Load and merge disagreement data
    disagree = pd.read_csv(disagree_path)
    print(
        f"Disagreement data: {len(disagree)} observations, year range: {disagree['fyear'].min()}-{disagree['fyear'].max()}")

    merged = compu.merge(disagree, on=['cusip', 'fyear'], how='inner')
    print(f"After merging: {len(merged)} observations")

    # Get price indices
    ppi_ind = get_ppi_index(ppi_series_path, ppi_data_path)
    cpi_df = get_cpi(cpi_path)
    pk_df = get_pk(pk_path)
    yrs = merged['fyear']
    baa_df = get_baa_yield(yrs.min(), yrs.max())

    # Calculate variables
    panel = calculate_variables(merged, ppi_ind, cpi_df, pk_df, baa_df)
    print(f"Final panel before complete case: {len(panel)} observations")

    return panel


def run_analysis(compu_path, disagree_path, ppi_series_path, ppi_data_path,
                 cpi_path, pk_path, out_dir='~/Downloads'):
    print("\nStarting analysis...")
    print(f"Using Compustat data from: {compu_path}")

    # 1) build the full, cleaned panel
    panel = build_panel(compu_path, disagree_path,
                        ppi_series_path, ppi_data_path,
                        cpi_path, pk_path)

    print("\nPanel Summary:")
    print(f"Total observations: {panel.shape[0]}")
    print(f"Unique firms: {panel['gvkey'].nunique()}")
    print("\nYear distribution:")
    year_dist = panel['fyear'].value_counts().sort_index()
    print(year_dist)

    # 2) drop ANY row with a missing value in core vars
    core_vars = ['Tobins_Q', 'invest_cap_sum', 'cash_flow', 'equity_issuance']
    panel_cc = panel.dropna(subset=core_vars)

    # Handle lagged variables
    lag_cols = [c for c in panel_cc.columns if c.startswith('lag_')]
    panel_cc[lag_cols] = (
        panel_cc
            .groupby('gvkey')[lag_cols]
            .transform(lambda x: x.ffill().bfill())
    )

    print("\nAfter complete-case cleaning:")
    print(f"Total observations: {panel_cc.shape[0]}")
    print(f"Unique firms: {panel_cc['gvkey'].nunique()}")
    print("\nYear distribution after cleaning:")
    year_dist_clean = panel_cc['fyear'].value_counts().sort_index()
    print(year_dist_clean)

    # Check for data quality
    print("\nData quality check:")
    for var in core_vars:
        print(f"\n{var}:")
        print(panel_cc[var].describe())

    # 3) Save the cleaned panel
    out_path = Path(out_dir).expanduser() / "clean_panel.csv"
    panel_cc.to_csv(out_path, index=False)
    print(f"\nSaved complete-case panel to {out_path}")

    return panel_cc
