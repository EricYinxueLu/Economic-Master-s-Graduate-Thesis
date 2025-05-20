import warnings, sys
import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.system import IVSystemGMM
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# 1) Load clean panel
# ------------------------------------------------------------------
try:
    df = pd.read_csv("clean_panel.csv")
except FileNotFoundError:
    sys.exit("clean_panel.csv not found – place it in the same folder.")

panel = (
    df.assign(gvkey=df.gvkey.astype(str), fyear=df.fyear.astype(int))
        .set_index(["gvkey", "fyear"], drop=False)
        .sort_index()
)

# ------------------------------------------------------------------
# 2) Build lags and additional variables -----------------------------
# ------------------------------------------------------------------
# Extended lags (now including lag-3)
for k in (1, 2, 3):
    panel[f"inv_lag{k}"] = panel.groupby(level="gvkey").invest_cap_sum.shift(k)
    panel[f"Tobins_Q_lag{k}"] = panel.groupby(level="gvkey").Tobins_Q.shift(k)
    panel[f"cashflow_lag{k}"] = panel.groupby(level="gvkey").cash_flow.shift(k)

# Add crisis dummies and trends
panel['crisis'] = panel.fyear.isin([2008, 2009, 2020, 2021]).astype(int)
panel['trend'] = panel.fyear - panel.fyear.min()
panel['trend_sq'] = panel.trend ** 2

# ------------------------------------------------------------------
# 3) Keep vars & drop NA -------------------------------------------
# ------------------------------------------------------------------
vars_keep = [
    "invest_cap_sum", "inv_lag1", "inv_lag2",  # dep & its lags
    "Tobins_Q", "cash_flow",  # endogenous regressors
    "debt_to_asset", "sales_growth", "ROA",
    "Disagreement", "DisagreementInvFCompDir", "DisagreementInvFComp",  # governance proxies
    "Tobins_Q_lag2", "Tobins_Q_lag3",  # extended instruments
    "cashflow_lag2", "cashflow_lag3",
    "crisis", "trend", "trend_sq",
    "fyear", "gvkey"
]
panel_gmm = panel[vars_keep].dropna()
print(f"Obs after drop NA: {len(panel_gmm):,}")

# ------------------------------------------------------------------
# 4) Define different model specifications --------------------------
# ------------------------------------------------------------------
model_specs = {
    "baseline": {
        "formula": {
            "eq1": (
                "invest_cap_sum ~ 1 + inv_lag1 + C(fyear) + debt_to_asset "
                "+ sales_growth + ROA + Disagreement + DisagreementInvFCompDir + DisagreementInvFComp "
                "[Tobins_Q + cash_flow ~ Tobins_Q_lag2 + cashflow_lag2]"
            )
        },
        "description": "Baseline specification"
    },
    "extended_instruments": {
        "formula": {
            "eq1": (
                "invest_cap_sum ~ 1 + inv_lag1 + C(fyear) + debt_to_asset "
                "+ sales_growth + ROA + Disagreement + DisagreementInvFCompDir + DisagreementInvFComp "
                "[Tobins_Q + cash_flow ~ Tobins_Q_lag2 + Tobins_Q_lag3 + cashflow_lag2 + cashflow_lag3]"
            )
        },
        "description": "Extended instruments (lag-2 and lag-3)"
    },
    "two_lags": {
        "formula": {
            "eq1": (
                "invest_cap_sum ~ 1 + inv_lag1 + inv_lag2 + C(fyear) + debt_to_asset "
                "+ sales_growth + ROA + Disagreement + DisagreementInvFCompDir + DisagreementInvFComp "
                "[Tobins_Q + cash_flow ~ Tobins_Q_lag2 + cashflow_lag2]"
            )
        },
        "description": "Including second investment lag"
    },
    "no_cash_flow": {
        "formula": {
            "eq1": (
                "invest_cap_sum ~ 1 + inv_lag1 + C(fyear) + debt_to_asset "
                "+ sales_growth + ROA + Disagreement + DisagreementInvFCompDir + DisagreementInvFComp "
                "[Tobins_Q ~ Tobins_Q_lag2]"
            )
        },
        "description": "Excluding cash flow"
    },
    "crisis_dummy": {
        "formula": {
            "eq1": (
                "invest_cap_sum ~ 1 + inv_lag1 + crisis + trend + trend_sq + debt_to_asset "
                "+ sales_growth + ROA + Disagreement + DisagreementInvFCompDir + DisagreementInvFComp "
                "[Tobins_Q + cash_flow ~ Tobins_Q_lag2 + cashflow_lag2]"
            )
        },
        "description": "Using crisis dummy instead of year FE"
    }
}


def run_model(spec_name, spec_dict, data, clusters=None):
    """Run GMM model with diagnostics"""
    print(f"\n{'=' * 80}")
    print(f"Running {spec_name}: {spec_dict['description']}")
    print('=' * 80)

    # Fit model
    gmm = IVSystemGMM.from_formula(spec_dict["formula"], data)

    # Modified covariance configuration
    if clusters is not None:
        # Use entity-clustered standard errors
        res = gmm.fit(cov_type='kernel', kernel='bartlett', bandwidth=None)
    else:
        # Use robust standard errors
        res = gmm.fit(cov_type='robust')

    print(res.summary)

    # Run diagnostics
    run_diagnostics(res, data)

    return res


def run_diagnostics(res, data):
    """Comprehensive model diagnostics"""
    print("\n=== Diagnostic Tests ===")

    # Get residuals and calculate first differences by panel
    resids = pd.Series(res.resids.squeeze(), index=data.index)
    diff_resids = resids.groupby(level='gvkey').diff().dropna()

    # Arellano-Bond tests
    def ab_test(residuals, order):
        y = residuals.iloc[order:]
        x = residuals.shift(order).iloc[order:]
        valid = ~(y.isna() | x.isna())
        y, x = y[valid], x[valid]
        corr = np.corrcoef(y, x)[0, 1]
        n = len(y)
        stat = corr * np.sqrt(n)
        pval = 2 * (1 - stats.norm.cdf(abs(stat)))
        return stat, pval

    # Store AB test results
    ab_results = {}
    print("\n=== Arellano-Bond tests ===")
    for order in (1, 2):
        stat, pval = ab_test(diff_resids, order)
        ab_results[order] = {'stat': stat, 'pval': pval}
        verdict = "OK" if ((order == 1 and pval < 0.05) or (order == 2 and pval > 0.05)) else "REJECT"
        print(f"AR({order}): z = {stat:.4f}, p = {pval:.4f} → {verdict}")

    # Hansen J-test
    print("\n=== Hansen J-test ===")
    if hasattr(res, 'j_stat'):
        j = res.j_stat
        if j.df > 0:
            verdict = "OK (p>0.10)" if j.pval > 0.10 else "REJECT (p≤0.10)"
            print(f"Hansen J = {j.stat:.4f}  (df = {j.df})   p = {j.pval:.4f}  → {verdict}")
        else:
            print("Model is exactly identified (df = 0)")
            print("Note: Add more instruments to enable overidentification test")

    return ab_results


def compare_results(results):
    """Compare coefficients across specifications"""
    # Initialize storage for coefficients and standard errors
    coefficients = {}
    std_errors = {}

    # Variables of interest (including all key variables)
    key_vars = [
        'inv_lag1', 'inv_lag2', 'Tobins_Q', 'cash_flow',
        'Disagreement', 'DisagreementInvFCompDir', 'DisagreementInvFComp'
    ]

    # Collect coefficients and standard errors
    for spec_name, res in results.items():
        coefficients[spec_name] = {}
        std_errors[spec_name] = {}

        for var in key_vars:
            if var in res.params:
                coefficients[spec_name][var] = res.params[var]
                std_errors[spec_name][var] = res.std_errors[var]

    # Create DataFrames
    coef_df = pd.DataFrame(coefficients)
    se_df = pd.DataFrame(std_errors)

    # Format output
    print("\n=== Coefficient Comparison Across Specifications ===")
    print("\nPanel A: Coefficients")
    print(coef_df.round(4))

    print("\nPanel B: Standard Errors")
    print(se_df.round(4))

    # Create a formatted table with coefficients and standard errors
    formatted_table = pd.DataFrame(index=key_vars)

    for spec_name in coefficients.keys():
        col_values = []
        for var in key_vars:
            if var in coefficients[spec_name]:
                coef = coefficients[spec_name][var]
                se = std_errors[spec_name][var]
                # Format as coefficient (standard error)
                col_values.append(f"{coef:.4f}\n({se:.4f})")
            else:
                col_values.append("")
        formatted_table[spec_name] = col_values

    print("\nPanel C: Formatted Results (coefficient with standard error in parentheses)")
    print(formatted_table)

    return formatted_table


# ------------------------------------------------------------------
# 5) Run specifications --------------------------------------------
# ------------------------------------------------------------------
results = {}
for spec_name, spec_dict in model_specs.items():
    try:
        # Run with robust standard errors
        print(f"\nRunning {spec_name} with robust standard errors...")
        results[f"{spec_name}_robust"] = run_model(spec_name, spec_dict, panel_gmm)

        # Run with kernel-based clustering
        print(f"\nRunning {spec_name} with kernel-based clustering...")
        results[f"{spec_name}_clustered"] = run_model(
            f"{spec_name} (clustered)",
            spec_dict,
            panel_gmm,
            clusters=True
        )
    except Exception as e:
        print(f"Error running {spec_name}: {str(e)}")
        continue

# Run comparison
formatted_results = compare_results(results)

# Save results to CSV (optional)
formatted_results.to_csv('gmm_results_comparison.csv')