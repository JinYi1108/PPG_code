import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr, shapiro, levene,ttest_ind,mannwhitneyu,kruskal,chi2_contingency,fisher_exact
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm 
import pingouin as pg
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import BayesianRidge   
from scipy.stats.mstats import winsorize
from dython.nominal import associations
from dython.nominal  import cramers_v
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso, LassoCV, lasso_path
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  


df_full = pd.read_excel('PPG-HVPG转换.xlsx')
print(f"\n原始数据维度: {df_full.shape}")
print("----------- 1.PPG-HVPG 转换 -----------")
df_analysis = df_full[['PPG', 'HVPG']].copy()
df_analysis.dropna(subset=['PPG', 'HVPG'], inplace=True)

df_analysis['PPG'] = pd.to_numeric(df_analysis['PPG'], errors='coerce')
df_analysis['HVPG'] = pd.to_numeric(df_analysis['HVPG'], errors='coerce')
df_analysis.dropna(subset=['PPG', 'HVPG'], inplace=True) 

ppg_values = df_analysis['PPG']
hvpg_values = df_analysis['HVPG']
print(f"\n用于PPG-HVPG关系分析的样本数: {len(df_analysis)}")

shapiro_ppg_stat, shapiro_ppg_p = shapiro(ppg_values)
shapiro_hvpg_stat, shapiro_hvpg_p = shapiro(hvpg_values)
print(f"\nPPG Shapiro-Wilk 正态性检验: Statistic={shapiro_ppg_stat:.3f}, P-value={shapiro_ppg_p:.3f}")
print(f"HVPG Shapiro-Wilk 正态性检验: Statistic={shapiro_hvpg_stat:.3f}, P-value={shapiro_hvpg_p:.3f}")

if shapiro_ppg_p > 0.05:
    print("PPG 数据不拒绝正态分布假设 (p > 0.05)")
else:
    print("PPG 数据拒绝正态分布假设 (p <= 0.05)")

if shapiro_hvpg_p > 0.05:
    print("HVPG 数据不拒绝正态分布假设 (p > 0.05)")
else:
    print("HVPG 数据拒绝正态分布假设 (p <= 0.05)")


df_icc_long = pd.DataFrame({
            'subject_id': np.arange(len(df_analysis)), 
            'PPG_value': ppg_values.values,
            'HVPG_value': hvpg_values.values
        })
df_icc_long = pd.melt(df_icc_long, id_vars=['subject_id'], 
                              value_vars=['PPG_value', 'HVPG_value'],
                              var_name='measurement_method', value_name='pressure_value')
        
print("\n--- 计算组内相关系数 (ICC) ---")

icc_results = pg.intraclass_corr(data=df_icc_long, targets='subject_id', 
                                          raters='measurement_method', ratings='pressure_value')

icc_results.set_index('Type', inplace=True)
        
print(icc_results)

icc_single = icc_results.loc['ICC2', 'ICC']
ci_single_low = icc_results.loc['ICC2', 'CI95%'][0]
ci_single_high = icc_results.loc['ICC2', 'CI95%'][1]
        

icc_average = icc_results.loc['ICC2k', 'ICC']
ci_average_low = icc_results.loc['ICC2k', 'CI95%'][0]
ci_average_high = icc_results.loc['ICC2k', 'CI95%'][1]

print(f"\n提取的 ICC 值:")
print(f"  Single Measures (ICC2): {icc_single:.3f} (95% CI: {ci_single_low:.3f} - {ci_single_high:.3f})")
print(f"  Average Measures (ICC2k): {icc_average:.3f} (95% CI: {ci_average_low:.3f} - {ci_average_high:.3f})")

print("\nPPG 描述性统计:")
print(ppg_values.describe())
print("\nHVPG 描述性统计:")
print(hvpg_values.describe())

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=hvpg_values, y=ppg_values)
# plt.title('PPG-HVPG 散点图')
# plt.xlabel('HVPG')
# plt.ylabel('PPG')
# plt.grid(True)
# #plt.savefig('PPG-HVPG散点图.png')
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.regplot(x=hvpg_values, y=ppg_values, ci=95)
# plt.title('PPG-HVPG 线性回归和95%置信区间')
# plt.xlabel('HVPG')
# plt.ylabel('PPG')
# plt.grid(True)
# #plt.savefig('PPG-HVPG 线性回归和95%置信区间.png')
# # plt.show()

X_hvpg = hvpg_values.values.reshape(-1, 1)
y_ppg = ppg_values.values

lin_reg = LinearRegression()
lin_reg.fit(X_hvpg, y_ppg)
ppg_pred_linear = lin_reg.predict(X_hvpg)
residuals_linear = y_ppg - ppg_pred_linear
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=ppg_pred_linear, y=residuals_linear)
# plt.axhline(0, color='red', linestyle='-')
# plt.title('线性回归残差')
# plt.xlabel('基于HVPG预测的PPG')
# plt.ylabel('残差')
# plt.grid(True)
# plt.show()

pearson_corr, pearson_p = pearsonr(hvpg_values, ppg_values)
spearman_corr, spearman_p = spearmanr(hvpg_values, ppg_values)
print(f"\nPearson 相关系数: {pearson_corr:.3f}, P-value: {pearson_p:.3e}")
print(f"Spearman 等级相关系数: {spearman_corr:.3f}, P-value: {spearman_p:.3e}")

r2_linear = lin_reg.score(X_hvpg, y_ppg)
mse_linear = mean_squared_error(y_ppg, ppg_pred_linear)
print(f"\n线性回归模型: PPG = {lin_reg.intercept_:.2f} + {lin_reg.coef_[0]:.2f} * HVPG")
print(f"  R-squared (线性): {r2_linear:.3f}")
print(f"  MSE (线性): {mse_linear:.3f}")

poly_model_2 = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
poly_model_2.fit(X_hvpg, y_ppg)
ppg_pred_poly2 = poly_model_2.predict(X_hvpg)
r2_poly2 = r2_score(y_ppg, ppg_pred_poly2)
mse_poly2 = mean_squared_error(y_ppg, ppg_pred_poly2)

poly_coeffs = poly_model_2.named_steps['linearregression'].coef_
poly_intercept = poly_model_2.named_steps['linearregression'].intercept_
print(f"二次多项式回归模型: Intercept={poly_intercept:.2f}, Coeffs={poly_coeffs}")
print(f"  R-squared (二次多项式): {r2_poly2:.3f}")
print(f"  MSE (二次多项式): {mse_poly2:.3f}")


poly_model_3 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
poly_model_3.fit(X_hvpg, y_ppg)
ppg_pred_poly3 = poly_model_3.predict(X_hvpg)
r2_poly3 = r2_score(y_ppg, ppg_pred_poly3)
mse_poly3 = mean_squared_error(y_ppg, ppg_pred_poly3)
poly_coeffs3 = poly_model_3.named_steps['linearregression'].coef_
poly_intercept3 = poly_model_3.named_steps['linearregression'].intercept_
print(f"三次多项式回归模型: Intercept={poly_intercept3:.2f}, Coeffs={poly_coeffs3}")
print(f"  R-squared (三次多项式): {r2_poly3:.3f}")
print(f"  MSE (三次多项式): {mse_poly3:.3f}")


lowess_frac = 0.5
lowess_smooth = sm.nonparametric.lowess(y_ppg, X_hvpg.flatten(), frac=lowess_frac)
plt.figure(figsize=(10, 7))
sns.scatterplot(x=hvpg_values, y=ppg_values, label='原始数据', alpha=0.6)
# plt.plot(X_hvpg, ppg_pred_linear, color='orange', linestyle='--', label=f'线性回归 (R2={r2_linear:.2f})')
# plt.plot(X_hvpg[np.argsort(X_hvpg.flatten())], ppg_pred_poly2[np.argsort(X_hvpg.flatten())], color='green', linestyle='-.', label=f'二次多项式 (R2={r2_poly2:.2f})')
# plt.plot(lowess_smooth[:, 0], lowess_smooth[:, 1], color='red', linestyle='-', linewidth=2, label=f'LOESS (frac={lowess_frac})')
# plt.title('不同模型拟合 PPG vs HVPG')
# plt.xlabel('HVPG')
# plt.ylabel('PPG')
# plt.legend()
# plt.grid(True)
# plt.show()



intercept_poly2 = poly_model_2.named_steps['linearregression'].intercept_
coeffs_poly2 = poly_model_2.named_steps['linearregression'].coef_


print(f"\n二次多项式模型确认:")
print(f"  Intercept (β0): {intercept_poly2:.4f}")
print(f"  Coefficient for HVPG (β1): {coeffs_poly2[0]:.4f}")
print(f"  Coefficient for HVPG^2 (β2): {coeffs_poly2[1]:.4f}")
print(f"  估算公式: PPG_est = {intercept_poly2:.2f} + ({coeffs_poly2[0]:.2f} * HVPG) + ({coeffs_poly2[1]:.2f} * HVPG^2)")









df_full_1 = pd.read_excel('original_data.xlsx')
print(f"原始 DataFrame 'df_full_1' 的维度: {df_full_1.shape}")
df_full_1.columns = df_full_1.columns.str.strip()
if 'PPG' in df_full_1.columns:
    df_full_1['PPG'] = pd.to_numeric(df_full_1['PPG'], errors='coerce')
if 'HVPG' in df_full_1.columns:
    df_full_1['HVPG'] = pd.to_numeric(df_full_1['HVPG'], errors='coerce')



condition_both_missing = df_full_1['PPG'].isnull() & df_full_1['HVPG'].isnull()
num_both_missing = condition_both_missing.sum()
if num_both_missing > 0:
    print(f"发现 {num_both_missing} 个样本的 PPG 和 HVPG 值均缺失。")
else:
    print("没有样本的 PPG 和 HVPG 值均缺失。")

condition_to_impute = df_full_1['PPG'].isnull() & df_full_1['HVPG'].notnull()
num_to_impute = condition_to_impute.sum()
print(f"\n找到 {num_to_impute} 个样本需要根据 HVPG 估算 PPG。")


if num_to_impute > 0:

    hvpg_for_imputation = df_full_1.loc[condition_to_impute, 'HVPG']

    ppg_estimated_values = intercept_poly2 + \
                           coeffs_poly2[0] * hvpg_for_imputation + \
                           coeffs_poly2[1] * (hvpg_for_imputation**2)
    

    df_full_1.loc[condition_to_impute, 'PPG'] = ppg_estimated_values
    
    print(f"已为 {len(ppg_estimated_values)} 个样本估算了 PPG 值。")

df_full_1['PPG'] = pd.to_numeric(df_full_1['PPG'], errors='coerce')

ppg_values1 = df_full_1['PPG'].dropna()
print("\nPPG 描述性统计:")
print(ppg_values1.describe())


sex_mapping = {
    1: 0,  
    2: 1  
}
df_full_1['Sex'] = df_full_1['Sex'].map(sex_mapping)
print("Sex 特征已编码为 0/1。")

identifier_and_target_cols = ['PPG', 'HVPG']
feature_columns = [col for col in df_full_1.columns if col not in identifier_and_target_cols]
missing_summary = df_full_1[feature_columns].isnull().sum()
missing_percentage = (missing_summary / len(df_full_1)) * 100

missing_df = pd.DataFrame({
    '缺失数量': missing_summary,
    '缺失百分比': missing_percentage
})
print(missing_df[missing_df['缺失数量'] > 0].sort_values(by='缺失数量', ascending=False))
child_pugh_col_name = 'Child-Pugh'
if child_pugh_col_name in df_full_1.columns and df_full_1[child_pugh_col_name].dtype == 'object':
    print(f"\n--- 编码特征: {child_pugh_col_name} ---")
    print(f"原始唯一值: {df_full_1[child_pugh_col_name].unique()}")
    child_pugh_mapping = {'A': 0, 'B': 1, 'C': 2} 
    df_full_1[child_pugh_col_name] = df_full_1[child_pugh_col_name].map(child_pugh_mapping)
    
    print(f"{child_pugh_col_name} 已有序编码。")



print("----------- 2 处理缺失值 -----------")


numerical_feature_cols = df_full_1[feature_columns].select_dtypes(include=np.number).columns.tolist()
categorical_feature_cols = df_full_1[feature_columns].select_dtypes(exclude=np.number).columns.tolist()



print(f"数值型特征: {numerical_feature_cols}")
print(f"分类型特征: {categorical_feature_cols}")

HVPGcols = [ 'HVPG']
featureMICE_columns = [col for col in df_full_1.columns if col not in HVPGcols]
featureMICE2_columns = [col for col in featureMICE_columns if col not in ['RPVF']]
if featureMICE2_columns:
    if df_full_1[featureMICE2_columns].isnull().sum().sum() > 0: 
        print(f"对以下数值型特征使用 IterativeImputer: {featureMICE2_columns}")

        mice_imputer_all = IterativeImputer(
            estimator=GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42),
            max_iter=30, 
            random_state=42,
            add_indicator=False)
        imputed_numerical_data = mice_imputer_all.fit_transform(df_full_1[featureMICE2_columns])
        

        df_imputed_numerical = pd.DataFrame(imputed_numerical_data, 
                                            columns=featureMICE2_columns, 
                                            index=df_full_1.index).round(1)
        for col in featureMICE2_columns:
            df_full_1[col] = df_imputed_numerical[col]
        print("数值型特征已使用 IterativeImputer 插补完成。")



feature_RPVF = ['RPVF']
if feature_RPVF:
    if df_full_1[feature_RPVF].isnull().sum().sum() > 0: 
        print(f"公式计算缺失的: {feature_RPVF}")

        mask = df_full_1['RPVF'].isnull()
        df_full_1.loc[mask, 'RPVF'] = ((df_full_1.loc[mask, 'RPV']/20)**2*3.1415926 *df_full_1.loc[mask, 'PV blood flow velocity']*0.5).round(6)
        print("RPVF 特征已根据公式计算完成。")

print("----------- 3 异常值处理 -----------")
numerical_feature_cols_original = [
    'Age', 'Hb', 'PLT', 'ALT', 'AST', 'TB', 'A', 'PT', 'INR', 'LSM',
    'PV', 'SV', 'SMV',  
     'RPVF']
num_cols_to_plot = len(numerical_feature_cols_original)
n_cols_subplot = min(4, num_cols_to_plot)
n_rows_subplot = (num_cols_to_plot + n_cols_subplot - 1) // n_cols_subplot

plt.figure(figsize=(5 * n_cols_subplot, 4 * n_rows_subplot))
for i, col in enumerate(numerical_feature_cols_original):
    plt.subplot(n_rows_subplot, n_cols_subplot, i + 1)
    sns.boxplot(y=df_full_1[col])
    plt.title(col)
    plt.tight_layout()
# plt.suptitle("数值型特征的箱线图分布 (异常值检查)", y=1.02, fontsize=16)
# plt.show()

winsor_limits_setting = (0.025, 0.025)

df_winsorized_processed = df_full_1.copy()
print(f"\n--- 应用 Winsorization (limits={winsor_limits_setting}) ---")
for col in numerical_feature_cols_original:
    if col in df_winsorized_processed.columns:
        original_min = df_winsorized_processed[col].min()
        original_max = df_winsorized_processed[col].max()
        data_to_winsorize = df_winsorized_processed[col].dropna().values
        if len(data_to_winsorize) > 0: 
            winsorized_data_values = winsorize(data_to_winsorize, limits=winsor_limits_setting)
            

            temp_series = pd.Series(np.nan, index=df_winsorized_processed.index, dtype=float)
            temp_series.loc[df_winsorized_processed[col].notna()] = winsorized_data_values
            df_winsorized_processed[col] = temp_series

            processed_min = df_winsorized_processed[col].min()
            processed_max = df_winsorized_processed[col].max()

            if not (np.isclose(original_min, processed_min) and np.isclose(original_max, processed_max)):
                print(f"  特征 '{col}': 已进行 Winsorization。")
                print(f"    原始范围: [{original_min:.2f} - {original_max:.2f}] -> Winsorized 范围: [{processed_min:.2f} - {processed_max:.2f}]")
print("\n--- Winsorization 处理后的箱线图 ---")
num_cols_to_plot = len(numerical_feature_cols_original)
n_cols_subplot = min(4, num_cols_to_plot)
n_rows_subplot = (num_cols_to_plot + n_cols_subplot - 1) // n_cols_subplot


plt.figure(figsize=(5 * n_cols_subplot, 4 * n_rows_subplot))
for i, col in enumerate(numerical_feature_cols_original):
    if col in df_winsorized_processed.columns:
        plt.subplot(n_rows_subplot, n_cols_subplot, i + 1)
        sns.boxplot(y=df_winsorized_processed[col])
        plt.title(f"{col} (Winsorized {winsor_limits_setting})")
        plt.tight_layout()
# plt.suptitle("Winsorization 处理后的数值特征箱线图", y=1.02, fontsize=16)
# plt.show()

df_full_1['PPG'] = np.where(
    df_full_1['PPG'].notnull(),
    np.where(df_full_1['PPG'] >= 20, 1, 0),
    np.nan
)





target_col = 'PPG'

continuous_numerical_features = [
    'Age', 'Hb', 'PLT',  'TB', 'A', 'PT',  'LSM','ALT/AST',
    'PV', 'SV', 'SMV', 'PV blood flow velocity', 
    'RPVF' 
]
encoded_categorical_features = [
    col for col in df_full_1.columns 
    if col not in [target_col]  + continuous_numerical_features
]


print(f"\n编码后的分类特征: {encoded_categorical_features}")

df_analysis_train = df_full_1.copy()
group0 = df_analysis_train[df_analysis_train[target_col] == 0]
group1 = df_analysis_train[df_analysis_train[target_col] == 1]
alpha_normality = 0.05
alpha_levene = 0.05
alpha_test = 0.05
univariate_results_refined = []

for feature in continuous_numerical_features:
    data_g0 = group0[feature].dropna()
    data_g1 = group1[feature].dropna()

    shapiro_g0_p = shapiro(data_g0)[1] 
    shapiro_g1_p = shapiro(data_g1)[1] 
    is_normal_g0 = shapiro_g0_p > alpha_normality 
    is_normal_g1 = shapiro_g1_p > alpha_normality 
    desc_g0 = f"{data_g0.mean():.2f} ± {data_g0.std():.2f}" if is_normal_g0 else f"{data_g0.median():.2f} ({data_g0.quantile(0.25):.2f}-{data_g0.quantile(0.75):.2f})"
    desc_g1 = f"{data_g1.mean():.2f} ± {data_g1.std():.2f}" if is_normal_g1 else f"{data_g1.median():.2f} ({data_g1.quantile(0.25):.2f}-{data_g1.quantile(0.75):.2f})"
    test_used = ""
    p_value = np.nan
    test_stat = np.nan
    if is_normal_g0 and is_normal_g1:
        levene_p = levene(data_g0, data_g1)[1]
        equal_var = levene_p > alpha_levene if pd.notnull(levene_p) else False
        test_used = "t-test" if equal_var else "Welch's t-test"
        test_stat, p_value = ttest_ind(data_g0, data_g1, equal_var=equal_var, nan_policy='omit')
    else:
        test_used = "Kruskal-Wallis H"

        test_stat, p_value = kruskal(data_g0, data_g1, nan_policy='omit') 
        #test_stat, p_value = mannwhitneyu(data_g0, data_g1, alternative='two-sided', nan_policy='omit')
        
            
    univariate_results_refined.append({
        'Feature': feature, 'Type': 'Continuous',
        'Desc_G0': desc_g0, 'Desc_G1': desc_g1,
        'Test_Used': test_used, 'P_value': p_value
    })
    print(f"  {feature}: G0={desc_g0}, G1={desc_g1}, Test={test_used}, P={p_value:.4f}")

for feature in encoded_categorical_features:
    
    contingency_table = pd.crosstab(df_analysis_train[target_col], df_analysis_train[feature])
    print(f"\n分析特征: {feature}")
    print("  列联表 (频数):")
    print(contingency_table)
    

    chi2_stat, p_value, dof, expected_freqs = np.nan, np.nan, np.nan, None
    test_used = "Chi-squared"

    chi2_stat, p_value, dof, expected_freqs = chi2_contingency(contingency_table)

    if np.any(expected_freqs < 5) and contingency_table.shape == (2,2): 
        test_used = "Fisher's Exact Test"
        odds_ratio, p_value_fisher = fisher_exact(contingency_table)
        p_value = p_value_fisher 
        print(f"期望频数低，使用 Fisher's Exact Test, P={p_value:.4f}")
    else:
        print(f" Chi-squared Test: Chi2={chi2_stat:.2f}, P={p_value:.4f}")

        
    univariate_results_refined.append({
        'Feature': feature, 'Type': 'Categorical (Encoded)',
        'Desc_G0': "See contingency table", 'Desc_G1': "See contingency table", 
        'Test_Used': test_used, 'P_value': p_value
    })

univariate_results_refined_df = pd.DataFrame(univariate_results_refined)
print("\n--- 单变量分析细化结果汇总 ---")

print(univariate_results_refined_df[['Feature', 'Type', 'Test_Used', 'P_value']].sort_values(by='P_value'))

significant_features_refined = univariate_results_refined_df[univariate_results_refined_df['P_value'] < alpha_test]['Feature'].tolist()
non_significant_features_refined = univariate_results_refined_df[univariate_results_refined_df['P_value'] >= alpha_test]['Feature'].tolist()

print(f"\n--- 基于细化单变量分析 (P < {alpha_test}) 的初步筛选结果 ---")
print(f"显著相关的特征 ({len(significant_features_refined)}个):")
print(significant_features_refined)
print(f"\n不显著相关的特征 ({len(non_significant_features_refined)}个)，考虑移除:")
print(non_significant_features_refined)

print("\n--- 连续数值特征之间的 Spearman 相关性矩阵 ---")
corr_spearman_numerical = df_full_1[continuous_numerical_features].corr(method='spearman')
    
plt.figure(figsize=(max(8, len(continuous_numerical_features)*0.8), 
                        max(6, len(continuous_numerical_features)*0.7)))
sns.heatmap(corr_spearman_numerical, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix (Continuous Numerical Features)')
plt.tight_layout()
plt.show()
upper_spearman = corr_spearman_numerical.where(np.triu(np.ones(corr_spearman_numerical.shape), k=1).astype(bool))
high_corr_spearman_pairs = []
spearman_threshold = 0.7 
for column in upper_spearman.columns:
    for index in upper_spearman.index:
        val = upper_spearman.loc[index, column]
        if pd.notnull(val) and abs(val) > spearman_threshold:
            high_corr_spearman_pairs.append((index, column, val))
if high_corr_spearman_pairs:
    print(f"\n  发现以下连续数值特征对 Spearman 相关性绝对值 > {spearman_threshold}:")
    for pair in sorted(high_corr_spearman_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"    '{pair[0]}' 和 '{pair[1]}': {pair[2]:.3f}")

print("\n--- 编码后分类特征之间的 Cramér's V 相关性矩阵 ---")
cramers_v_data = []
cat_cols = encoded_categorical_features
for i in range(len(cat_cols)):
    row_data = {}
    for j in range(len(cat_cols)):
        if i == j:
            row_data[cat_cols[j]] = 1.0
        elif j < i :
            row_data[cat_cols[j]] = cramers_v_data[j][cat_cols[i]] 
        else:   
            v = cramers_v(df_full_1[cat_cols[i]], df_full_1[cat_cols[j]])
            row_data[cat_cols[j]] = v
    cramers_v_data.append(row_data)
        
cramers_v_matrix = pd.DataFrame(cramers_v_data, index=cat_cols)

plt.figure(figsize=(max(8, len(cat_cols)*0.8), max(6, len(cat_cols)*0.7)))
sns.heatmap(cramers_v_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=0, vmax=1)
plt.title("Cramér's V Matrix (Encoded Categorical Features)")
plt.tight_layout()
plt.show()

upper_cramers = cramers_v_matrix.where(np.triu(np.ones(cramers_v_matrix.shape), k=1).astype(bool))
high_corr_cramers_pairs = []
cramers_threshold = 0.6 
for column in upper_cramers.columns:
    for index in upper_cramers.index:
        val = upper_cramers.loc[index, column]
        if pd.notnull(val) and val > cramers_threshold: 
            high_corr_cramers_pairs.append((index, column, val))
if high_corr_cramers_pairs:
    print(f"\n  发现以下编码后分类特征对 Cramér's V > {cramers_threshold}:")
    for pair in sorted(high_corr_cramers_pairs, key=lambda x: x[2], reverse=True):
        print(f"    '{pair[0]}' 和 '{pair[1]}': {pair[2]:.3f}")
else:
    print(f"  未发现编码后分类特征对之间 Cramér's V > {cramers_threshold}。")

print("\n--- 连续数值特征与二元(0/1)分类特征之间的 Pearson 相关性 ---")
binary_encoded_features = [
    col for col in encoded_categorical_features 
    if df_full_1[col].nunique() == 2 and df_full_1[col].min() == 0 and df_full_1[col].max() == 1
]
df_mixed_subset = df_full_1[continuous_numerical_features + binary_encoded_features]
corr_mixed = df_mixed_subset.corr(method='pearson')
corr_point_biserial_subset = corr_mixed.loc[continuous_numerical_features, binary_encoded_features]
if not corr_point_biserial_subset.empty:
    plt.figure(figsize=(max(6, len(binary_encoded_features)*0.8), 
                        max(6, len(continuous_numerical_features)*0.4)))
    sns.heatmap(corr_point_biserial_subset, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title('Point-Biserial Correlation (Continuous vs. Binary Encoded Categorical)')
    plt.tight_layout()
    plt.show()

    
    pb_threshold = 0.3
    high_pb_pairs = []
    for cat_col in corr_point_biserial_subset.columns: 
        for num_col in corr_point_biserial_subset.index: 
            val = corr_point_biserial_subset.loc[num_col, cat_col]
            if pd.notnull(val) and abs(val) > pb_threshold:
                high_pb_pairs.append((num_col, cat_col, val))
    if high_pb_pairs:
        print(f"\n  发现以下连续数值与二元分类特征对 Pearson 相关性绝对值 > {pb_threshold}:")
        for pair in sorted(high_pb_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"    '{pair[0]}' 和 '{pair[1]}': {pair[2]:.3f}")
    else:
        print(f"  未发现连续数值与二元分类特征间 Pearson 相关性绝对值 > {pb_threshold}。")

else:
    print("  未能计算点双列相关性 ")

ordinal_encoded_features_train = [
    col for col in encoded_categorical_features
    if col not in binary_encoded_features
]
print("\n--- 连续数值特征与有序多分类编码特征之间的 Spearman 相关性 ---")
df_ordinal_continuous_subset = df_full_1[continuous_numerical_features + ordinal_encoded_features_train]
corr_spearman_ordinal_continuous = df_ordinal_continuous_subset.corr(method='spearman')
corr_spearman_subset_display = corr_spearman_ordinal_continuous.loc[continuous_numerical_features, ordinal_encoded_features_train]
if not corr_spearman_subset_display.empty:
    plt.figure(figsize=(max(6, len(ordinal_encoded_features_train)*0.8), 
                        max(6, len(continuous_numerical_features)*0.4)))
    sns.heatmap(corr_spearman_subset_display, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title('Spearman Correlation (Continuous vs Ordinal Encoded Categorical)')
    plt.tight_layout()
    plt.show()

    
    spearman_ord_threshold = 0.3 
    high_spearman_ord_pairs = []
    for ord_col in corr_spearman_subset_display.columns: 
        for num_col in corr_spearman_subset_display.index: 
            val = corr_spearman_subset_display.loc[num_col, ord_col]
            if pd.notnull(val) and abs(val) > spearman_ord_threshold:
                high_spearman_ord_pairs.append((num_col, ord_col, val))
    if high_spearman_ord_pairs:
        print(f"\n  发现以下连续数值与有序分类特征对 Spearman 相关性绝对值 > {spearman_ord_threshold}:")
        for pair in sorted(high_spearman_ord_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"    '{pair[0]}' 和 '{pair[1]}': {pair[2]:.3f}")
    else:
        print(f"  未发现连续数值与有序分类特征间 Spearman 相关性绝对值 > {spearman_ord_threshold}。")
else:
    print("  未能计算连续与有序分类特征的Spearman相关性。")

num_features = continuous_numerical_features
binary_features = [col for col in encoded_categorical_features 
                  if df_full_1[col].nunique() == 2 and set(df_full_1[col].unique()) == {0, 1}]
ordinal_features = [col for col in encoded_categorical_features 
                   if col not in binary_features]
all_features = num_features + binary_features + ordinal_features
corr_matrix = pd.DataFrame(np.zeros((len(all_features), len(all_features))),
                          index=all_features, columns=all_features)
num_corr = df_full_1[num_features].corr(method='spearman')
corr_matrix.loc[num_features, num_features] = num_corr
for i, f1 in enumerate(binary_features + ordinal_features):
    for j, f2 in enumerate(binary_features + ordinal_features):
        if i == j:
            corr_matrix.loc[f1, f2] = 1.0
        else:
            corr_matrix.loc[f1, f2] = cramers_v(df_full_1[f1], df_full_1[f2])
for num in num_features:
    for binary in binary_features:
        corr = df_full_1[[num, binary]].corr(method='pearson').iloc[0, 1]
        corr_matrix.loc[num, binary] = corr
        corr_matrix.loc[binary, num] = corr  
for num in num_features:
    for ordinal in ordinal_features:
        corr = df_full_1[[num, ordinal]].corr(method='spearman').iloc[0, 1]
        corr_matrix.loc[num, ordinal] = corr
        corr_matrix.loc[ordinal, num] = corr  
   
plt.figure(figsize=(20, 18))

ax = sns.heatmap(corr_matrix, annot=corr_matrix.round(2), fmt="", 
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
cbar = ax.collections[0].colorbar
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_ticklabels(['-1 ', '-0.5', '0', '0.5', '1 '])

plt.title("Complete Correlation Matrix", pad=20)
plt.tight_layout()
plt.show()



feature_columns = ['Age', 'Hb', 'PLT', 'TB', 'A', 'PT', 'LSM', 'ALT/AST', 'PV', 'SV', 'SMV', 'PV blood flow velocity', 'Portal Vein Thrombosis', 'Ascites', 'Esophageal gastric varices', 'Splenomegaly', 'Hepatic encephalopathy','RPVF', 'Sex', 'Aetiology']



print(f"\n最终选定的特征列: {feature_columns}")
X = df_full_1[feature_columns]
y = df_full_1[target_col]
print(f"\n特征集 X 的维度: {X.shape}")
print(f"目标变量 y 的维度: {y.shape}")
print("\n--- 特征与目标变量的相关性 ---")

all_correlations_with_target = {}

new_continuous_numerical_features = [
    'Age', 'Hb', 'PLT',  'TB', 'A', 'PT',  'LSM','ALT/AST',
    'PV', 'SV', 'SMV', 'PV blood flow velocity', 
    'RPVF' 
]
if new_continuous_numerical_features:
    corrs_continuous = X[new_continuous_numerical_features].corrwith(y, method='pearson')
    for feature, corr_val in corrs_continuous.items():
        all_correlations_with_target[feature] = {'type': 'Continuous', 'correlation': corr_val, 'method': 'Point-Biserial (Pearson)'}

b_encoded_features = [
    col for col in feature_columns
    if df_full_1[col].nunique() == 2 and df_full_1[col].min() == 0 and df_full_1[col].max() == 1
]
print(b_encoded_features)

o_encoded_features_train = [
    col for col in feature_columns
    if col not in b_encoded_features+new_continuous_numerical_features
]
print(o_encoded_features_train)

if b_encoded_features:
    corrs_binary = X[b_encoded_features].corrwith(y, method='pearson')
    for feature, corr_val in corrs_binary.items():
        all_correlations_with_target[feature] = {'type': 'Binary Encoded', 'correlation': corr_val, 'method': 'Phi (Pearson)'}
        

if o_encoded_features_train:
    corrs_ordinal = X[o_encoded_features_train].corrwith(y, method='spearman')
    for feature, corr_val in corrs_ordinal.items():
        all_correlations_with_target[feature] = {'type': 'Ordinal Encoded', 'correlation': corr_val, 'method': 'Spearman'}

correlations_df = pd.DataFrame.from_dict(all_correlations_with_target, orient='index')
correlations_df = correlations_df.sort_values(by='correlation', ascending=False)

print("\n--- 特征与目标变量的相关性 (按类型分别计算) ---")
print(correlations_df)
plt.figure(figsize=(10, max(8, len(correlations_df) * 0.25)))
sns.barplot(x=correlations_df['correlation'].values, y=correlations_df.index)
plt.title(f'筛选后特征与目标变量 ({target_col})')
plt.xlabel('相关系数值')
plt.ylabel('特征')
plt.grid(axis='x')
plt.tight_layout()
plt.show()
