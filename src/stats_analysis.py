import pandas as pd
from scipy.stats import ttest_ind, ks_2samp

def run_statistical_tests(df):
    print("\n📊 Statistical Test Results:\n")

    features = [col for col in df.columns if col not in ["label", "label_name"]]

    for feat in features:
        urban = df[df.label == 1][feat]
        highway = df[df.label == 0][feat]

        t_stat, t_p = ttest_ind(urban, highway)
        ks_stat, ks_p = ks_2samp(urban, highway)

        print(f"{feat}:")
        print(f"  T-test p-value: {t_p:.5f}")
        print(f"  KS-test p-value: {ks_p:.5f}")
        print("-"*40)