"""
Panel_Data_Analysis.py  — replicates the full battery of investment/disagreement tests
exactly as in your original workflow (static OLS, single‑proxy OLS, Whited‑style OLS,
Two‑way fixed effects, and the incremental panel specifications M0–M10).

Running this file will:
1. Load the trimmed panel (clean_panel.csv).
2. Print & save summary statistics.
3. Build helper columns (COVID / Crisis dummies, lags, etc.).
4. Estimate and save:
   • Whited‑style pooled OLS (all 3 disagreement proxies).
   • Three single‑proxy OLS specs.
   • Two‑way FE with lagged vars.
   • The incremental panel models (M0–M10) exactly as in the report.
5. Write every table to disk (TXT) and echo key results to console.

Notes
* All OLS models use HC1 heteroskedastic‑robust s.e.
* PanelOLS models use ‘robust’ covariance (cluster‑robust at entity level under the hood).
* Lags are computed by entity; rows with missing lagged values are dropped only for models
  that need them.
"""

import sys
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    print("[ERROR]   The 'linearmodels' package is required – install via `pip install linearmodels`. ")
    sys.exit(1)

warnings.simplefilter('ignore', category=FutureWarning)
pd.set_option('display.float_format', lambda x: f"{x:,.4f}")

# --------------------------------------------------------------------------------------
# Global settings / constants

COVID_YEARS   = {2020, 2021}
CRISIS_YEARS  = {2008, 2009}
DEPENDENT_VAR = 'invest_cap_sum'
ENTITY_COL    = 'gvkey'
TIME_COL      = 'fyear'
OUTPUT_DIR    = Path('.').resolve()

# Now treat profitability & growth as core q‐theory fundamentals
BASE_CONTROLS = [
    'Tobins_Q',
    'cash_flow',
    'debt_to_asset',
    'equity_issuance',
    'sales_growth',
    'ROA'
]

DISAG_COLS = [
    'Disagreement',
    'DisagreementInvFComp',
    'DisagreementInvFCompDir'
]

ALL_CONTROLS = BASE_CONTROLS  # we no longer need OTHER_CONTROLS

# --------------------------------------------------------------------------------------
# Utility helpers

def save_txt(name: str, content: str):
    path = OUTPUT_DIR / f"{name}.txt"
    with open(path, 'w') as fh:
        fh.write(content)
    print(f"[saved] {path.relative_to(OUTPUT_DIR)}")

def robust_ols(formula: str, data: pd.DataFrame):
    """OLS with HC1 robust s.e. – returns fitted model."""
    model = smf.ols(formula, data=data).fit(cov_type='HC1')
    return model

# --------------------------------------------------------------------------------------
# Data preparation

def load_and_prepare(path: str = 'clean_panel.csv') -> pd.DataFrame:
    if not Path(path).exists():
        print(f"[ERROR]   '{path}' not found – place the trimmed panel in the working dir.")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[INFO]    Loaded {len(df):,} observations from '{path}'.")

    # Indicators
    df['covid']  = df[TIME_COL].isin(COVID_YEARS).astype(int)
    df['crisis'] = df[TIME_COL].isin(CRISIS_YEARS).astype(int)

    # Lagged vars for dynamic FE (by entity)
    lag_cols = [
        DEPENDENT_VAR.replace('_sum', ''),  # assumes original invest_cap
        *DISAG_COLS,
        *BASE_CONTROLS
    ]
    df = df.sort_values([ENTITY_COL, TIME_COL])
    for col in lag_cols:
        df[f"lag_{col}"] = df.groupby(ENTITY_COL)[col].shift(1)

    return df

# --------------------------------------------------------------------------------------
# Diagnostics / Descriptives

def describe(df: pd.DataFrame):
    # Only these nine:
    vars_for_summary = [
        'invest_cap',
        'Disagreement',
        'DisagreementInvFComp',
        'DisagreementInvFCompDir',
        'Tobins_Q',
        'invest_cap_sum',
        'equity_issuance',
        'sales_growth',
        'ROA'
    ]
    desc = df[vars_for_summary].describe().T
    save_txt('summary_statistics', desc.to_string())
    print("\n=== Summary statistics ===")
    print(desc)

# --------------------------------------------------------------------------------------
# VIF helper (optional)

def calc_vif(df: pd.DataFrame, cols):
    X = df[cols].dropna().copy()
    X.insert(0, 'const', 1.0)
    vif = pd.DataFrame({
        'Variable': [c for c in X.columns if c != 'const'],
        'VIF': [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    })
    return vif

# --------------------------------------------------------------------------------------
# Estimation blocks

def whited_style_ols(df):
    formula = DEPENDENT_VAR + ' ~ ' + ' + '.join(DISAG_COLS + ALL_CONTROLS + ['C(fyear)'])
    res = robust_ols(formula, df)
    print("\n=== Whited-style OLS (all three disagreements together) ===")
    print(res.summary())
    save_txt('whited_style_ols', res.summary().as_text())
    return res

def single_proxy_ols(df):
    mappings = {
        'Disagreement only': ['Disagreement'],
        'DisagreementInvFComp only': ['DisagreementInvFComp'],
        'DisagreementInvFCompDir only': ['DisagreementInvFCompDir']
    }
    for title, proxy in mappings.items():
        formula = DEPENDENT_VAR + ' ~ ' + ' + '.join(proxy + BASE_CONTROLS + ['C(fyear)'])
        res = robust_ols(formula, df)
        print(f"\n=== OLS with {title} ===")
        print(res.summary())
        save_txt(f"ols_{title.replace(' ', '_').replace('only','')}", res.summary().as_text())

def individual_proxy_ols(df):
    """One‐by‐one significance checks with all base controls + year FE."""
    for proxy in DISAG_COLS:
        formula = (
            f"{DEPENDENT_VAR} ~ {proxy}"
            + " + " + " + ".join(ALL_CONTROLS)
            + " + C(fyear)"
        )
        res = robust_ols(formula, df)
        print(f"\n=== OLS with {proxy} (all controls + year FE) ===")
        print(res.summary())
        save_txt(f"ols_individual_{proxy}", res.summary().as_text())

def two_way_fe(df):
    dyn_cols = ['lag_' + DEPENDENT_VAR.replace('_sum','')] + DISAG_COLS + [
        'lag_' + c for c in BASE_CONTROLS
    ]
    needed = [DEPENDENT_VAR, ENTITY_COL, TIME_COL, *dyn_cols]
    dfe = df[needed].dropna().set_index([ENTITY_COL, TIME_COL])
    formula = DEPENDENT_VAR + ' ~ ' + ' + '.join(dyn_cols) + ' + EntityEffects + TimeEffects'
    res = PanelOLS.from_formula(formula, data=dfe).fit(cov_type='robust')
    print("\n=== Two-way FE results ===")
    print(res.summary)
    save_txt('two_way_FE', str(res.summary))
    return res


def incremental_panel_models(df):
    """Re‑creates M0 … M10 exactly as in the report."""
    models = {}

    # Common pieces
    base  = DEPENDENT_VAR + ' ~ '
    disc  = 'Disagreement'
    disc_all = ' + '.join(DISAG_COLS)

    controls_std = ' + '.join(BASE_CONTROLS)
    year_fe = ' + C(fyear)'

    # --- M0: Disagreement only
    models['M0'] = base + disc

    # --- M1: + standard controls
    models['M1'] = base + disc + ' + ' + controls_std

    # --- M2: + year‑fixed effects
    models['M2'] = base + disc + ' + ' + controls_std + year_fe

    # --- M3: + Covid interaction (with all covariates + year FE)
    models['M3'] = models['M2'] + ' + Disagreement:covid'

    # --- M4: + Crisis interaction
    models['M4'] = models['M2'] + ' + Disagreement:crisis'

    # --- M5: All three disagreement proxies (w/ year FE & controls)
    models['M5'] = base + disc_all + ' + ' + controls_std + ' + sales_growth + ROA' + year_fe

    # --- M6: All disagreements × Covid
    covid_inters = ' + '.join([f'{d}:covid' for d in DISAG_COLS])
    models['M6'] = models['M5'] + ' + ' + covid_inters

    # --- M7: Direct proxy + Crisis interaction
    models['M7'] = base + 'DisagreementInvFCompDir' + ' + ' + controls_std + ' + sales_growth + ROA' + year_fe + ' + DisagreementInvFCompDir:crisis'

    # --- M8: Direct proxy + Covid interaction
    models['M8'] = models['M7'].replace(':crisis', ':covid')

    # --- M9: All three × Crisis
    crisis_inters = ' + '.join([f'{d}:crisis' for d in DISAG_COLS])
    models['M9'] = models['M5'] + ' + ' + crisis_inters

    # --- M10: All three × Covid (already model 6 – reorder for clarity)
    models['M10'] = models['M6']  # just alias

    # Fit & store
    out_tables = []
    for tag, formula in models.items():
        res = robust_ols(formula, df)
        models[tag] = res  # overwrite formula with results
        save_txt(f"model_{tag}", res.summary().as_text())
        out_tables.append(res)
        print(f"\n=== {tag}: {formula.split('~')[1].strip()} ===")
        print(res.summary())

    # Combine key p‑values for disagreement terms for quick glance
    def extract_p(model, term):
        try:
            return model.pvalues[term]
        except KeyError:
            return np.nan

    sig_df = pd.DataFrame({
        'pval': {
            'M0_Disagreement': extract_p(models['M0'], 'Disagreement'),
            'M1_Disagreement': extract_p(models['M1'], 'Disagreement'),
            'M2_Disagreement': extract_p(models['M2'], 'Disagreement'),
            'M3_Disagreement': extract_p(models['M3'], 'Disagreement'),
            'M3_Disagreement_x_covid': extract_p(models['M3'], 'Disagreement:covid'),
            'M4_Disagreement': extract_p(models['M4'], 'Disagreement'),
            'M4_Disagreement_x_crisis': extract_p(models['M4'], 'Disagreement:crisis'),
            'M5_Disagreement': extract_p(models['M5'], 'Disagreement'),
            'M5_DisagreementInvFComp': extract_p(models['M5'], 'DisagreementInvFComp'),
            'M5_DisagreementInvFCompDir': extract_p(models['M5'], 'DisagreementInvFCompDir')
        }
    }).T
    save_txt('disagreement_term_significance', sig_df.to_string())

    return models

# --------------------------------------------------------------------------------------
# Residual diagnostics for the Whited‑style OLS (optional)

def residual_diagnostics(model):
    resids    = model.resid
    jb_stat, jb_p, skew, kurt = jarque_bera(resids)
    diag = dedent(f"""
    Residual Diagnostics (Whited‑style OLS)
    --------------------------------------
    Mean       : {resids.mean():.4f}
    Std.Dev.   : {resids.std(ddof=1):.4f}
    Min, Max   : {resids.min():.4f}, {resids.max():.4f}
    Skew, Kurt : {skew:.4f}, {kurt:.4f}
    Jarque‑Bera: {jb_stat:.4f}   p‑value = {jb_p:.5f}
    """)
    print(diag)
    save_txt('residual_diagnostics', diag)

# --------------------------------------------------------------------------------------
# Main

def main():
    df = load_and_prepare()
    describe(df)

    # Whited-style pooled OLS
    whited_res = whited_style_ols(df)

    # Single-proxy variants
    single_proxy_ols(df)

    # **New**: each disagreement individually (all controls + FE)
    individual_proxy_ols(df)

    # Two-way FE with dynamics
    two_way_fe(df)

    # Incremental panel specs (M0–M10)
    incremental_panel_models(df)

    # Residual diagnostics
    residual_diagnostics(whited_res)

    print("\n[INFO]    Finished all estimations. All result files saved in working directory.")

if __name__ == '__main__':
    main()


