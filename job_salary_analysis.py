# =============================================================================
#   JOB MARKET SALARY INTELLIGENCE
#   A Comprehensive Data Science Analysis - All 10 Experiments
#   Dataset : job_salary_prediction_dataset.csv (250,000 records)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import random
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, mean_squared_error, r2_score)

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False,
                     'axes.spines.right': False})
sns.set_palette("muted")

# -----------------------------------------------------------------------------
# LOAD DATASET
# -----------------------------------------------------------------------------
df = pd.read_csv('job_salary_prediction_dataset.csv')
print("\n" + "="*55)
print("      JOB MARKET SALARY INTELLIGENCE PROJECT")
print("="*55)
print(f"  Dataset Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Columns        : {list(df.columns)}")
print("="*55)


# =============================================================================
# EXPERIMENT 1 - Exploratory Data Analysis & Dataset Setup
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 1 - Exploratory Data Analysis")
print("="*55)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Basic Statistics ---")
print(df.describe().round(2))

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Categorical Columns ---")
for col in ['education_level', 'industry', 'company_size', 'remote_work']:
    print(f"\n{col}:\n{df[col].value_counts().to_string()}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Experiment 1 - Salary Distribution Overview", fontweight='bold')

axes[0].hist(df['salary'], bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title('Salary Distribution')
axes[0].set_xlabel('Salary (USD)')
axes[0].set_ylabel('Frequency')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

axes[1].boxplot(df['salary'], vert=False, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[1].set_title('Salary Box Plot')
axes[1].set_xlabel('Salary (USD)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

plt.tight_layout()
plt.savefig('exp1_eda.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 1 Complete - Plot saved as exp1_eda.png")


# =============================================================================
# EXPERIMENT 2 - Salary Classification using Logistic Regression
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 2 - Logistic Regression (Classification)")
print("="*55)

df2 = df.copy()
median_sal = df2['salary'].median()
df2['salary_class'] = (df2['salary'] >= median_sal).astype(int)
print(f"\nMedian Salary   : ${median_sal:,.0f}")
print(f"Class 0 (Low)   : {(df2['salary_class']==0).sum():,}")
print(f"Class 1 (High)  : {(df2['salary_class']==1).sum():,}")

le = LabelEncoder()
df2_enc = df2.copy()
for col in ['education_level', 'industry', 'company_size',
            'remote_work', 'job_title', 'location']:
    df2_enc[col] = le.fit_transform(df2_enc[col])

features = ['experience_years', 'education_level', 'skills_count',
            'industry', 'company_size', 'remote_work', 'certifications']
X = df2_enc[features]
y = df2_enc['salary_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_s, y_train)
y_pred = lr_model.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy        : {acc * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Low Salary', 'High Salary']))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'], ax=ax)
ax.set_title('Experiment 2 - Confusion Matrix (Logistic Regression)', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('exp2_logistic.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 2 Complete - Plot saved as exp2_logistic.png")


# =============================================================================
# EXPERIMENT 3 - Salary Trend Forecasting using Linear Regression
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 3 - Linear Regression (Trend Forecasting)")
print("="*55)

exp_sal = df.groupby('experience_years')['salary'].mean().reset_index()
exp_sal.columns = ['experience_years', 'avg_salary']

X_lr = exp_sal[['experience_years']].values
y_lr = exp_sal['avg_salary'].values

lin_model = LinearRegression()
lin_model.fit(X_lr, y_lr)
y_lr_pred = lin_model.predict(X_lr)

r2  = r2_score(y_lr, y_lr_pred)
mse = mean_squared_error(y_lr, y_lr_pred)

print(f"\nSlope (per year) : ${lin_model.coef_[0]:,.2f}")
print(f"Intercept        : ${lin_model.intercept_:,.2f}")
print(f"R2 Score         : {r2:.4f}")
print(f"RMSE             : ${np.sqrt(mse):,.2f}")

print("\n--- Salary Prediction by Experience ---")
for yr in [0, 5, 10, 15, 20, 25]:
    pred = lin_model.predict([[yr]])[0]
    print(f"  {yr:>2} years => Predicted Salary: ${pred:,.2f}")

x_line = np.linspace(0, 25, 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(exp_sal['experience_years'], exp_sal['avg_salary'],
           color='steelblue', label='Actual Avg Salary', zorder=3, s=60)
ax.plot(x_line, lin_model.predict(x_line), color='tomato', linewidth=2.5,
        linestyle='--', label='Linear Trend (Forecast)')
ax.set_title('Experiment 3 - Salary Trend by Experience (Linear Regression)', fontweight='bold')
ax.set_xlabel('Experience (Years)')
ax.set_ylabel('Average Salary (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.legend()
plt.tight_layout()
plt.savefig('exp3_linear.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 3 Complete - Plot saved as exp3_linear.png")


# =============================================================================
# EXPERIMENT 4 - Dataset Creation & Random Sampling
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 4 - Dataset Creation & Random Sampling")
print("="*55)

np.random.seed(42)
random.seed(42)

# 1. Simple Random Sampling
simple_sample = df.sample(n=500, random_state=42)
print(f"\n1. Simple Random Sample   : {len(simple_sample):,} records")
print(f"   Mean Salary            : ${simple_sample['salary'].mean():,.2f}")
print(f"   Std Salary             : ${simple_sample['salary'].std():,.2f}")

# 2. Stratified Sampling
strat_parts = []
for edu in df['education_level'].unique():
    part = df[df['education_level'] == edu].sample(frac=0.002, random_state=42)
    strat_parts.append(part)
strat_sample = pd.concat(strat_parts).reset_index(drop=True)
print(f"\n2. Stratified Sample (0.2% per education level): {len(strat_sample):,} records")
print(strat_sample['education_level'].value_counts().to_string())

# 3. Systematic Sampling
k = 500
sys_sample = df.iloc[::k].reset_index(drop=True)
print(f"\n3. Systematic Sample (every {k}th row): {len(sys_sample):,} records")
print(f"   Mean Salary            : ${sys_sample['salary'].mean():,.2f}")

# Compare Distributions
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Experiment 4 - Sampling Distribution Comparison", fontweight='bold')
for ax, (title, data) in zip(axes, [('Full Dataset (250K)', df),
                                      ('Simple Random (n=500)', simple_sample),
                                      ('Systematic (k=500)', sys_sample)]):
    ax.hist(data['salary'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_xlabel('Salary')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
plt.tight_layout()
plt.savefig('exp4_sampling.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 4 Complete - Plot saved as exp4_sampling.png")


# =============================================================================
# EXPERIMENT 5 - Hypothesis Testing using Z-Test (Salary Mean Analysis)
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 5 - Hypothesis Testing (Z-Test)")
print("="*55)

pop_mean    = df['salary'].mean()
pop_std     = df['salary'].std()
phd_sal     = df[df['education_level'] == 'PhD']['salary']
n           = len(phd_sal)
sample_mean = phd_sal.mean()

z_stat  = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
alpha   = 0.05

print(f"\nH0: Mean salary of PhD holders = Population mean")
print(f"H1: Mean salary of PhD holders != Population mean")
print(f"\nPopulation Mean (mu)   : ${pop_mean:,.2f}")
print(f"Population Std  (sigma)   : ${pop_std:,.2f}")
print(f"PhD Sample Mean (x_mean)   : ${sample_mean:,.2f}")
print(f"Sample Size     (n)   : {n:,}")
print(f"Z-Statistic           : {z_stat:.4f}")
print(f"P-Value               : {p_value:.6f}")
print(f"Significance    (alpha)   : {alpha}")
print("-"*45)
if p_value < alpha:
    print("RESULT: REJECT H0 => PhD salaries are significantly")
    print("        HIGHER than the population mean.")
else:
    print("RESULT: FAIL TO REJECT H0 => No significant difference.")

x     = np.linspace(-5, 5, 500)
y_pdf = stats.norm.pdf(x)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x, y_pdf, 'steelblue', linewidth=2, label='Standard Normal')
ax.fill_between(x, y_pdf, where=(x <= -1.96), color='tomato', alpha=0.4, label='Rejection Region (alpha/2)')
ax.fill_between(x, y_pdf, where=(x >=  1.96), color='tomato', alpha=0.4)
ax.axvline(z_stat, color='green',  linewidth=2,   linestyle='--', label=f'Z-stat = {z_stat:.2f}')
ax.axvline( 1.96,  color='red',    linewidth=1.5, linestyle=':')
ax.axvline(-1.96,  color='red',    linewidth=1.5, linestyle=':', label='+/-Z_crit = +/-1.96')
ax.set_title('Experiment 5 - Z-Test: PhD Salary vs Population Mean', fontweight='bold')
ax.set_xlabel('Z-Score')
ax.set_ylabel('Probability Density')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('exp5_ztest.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 5 Complete - Plot saved as exp5_ztest.png")


# =============================================================================
# EXPERIMENT 6 - Industry-wise Salary Analysis using NumPy
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 6 - NumPy Mean & Shape Analysis")
print("="*55)

industries = sorted(df['industry'].unique())
ind_stats  = {}

print(f"\n{'Industry':<18} {'Count':>7} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 78)

for ind in industries:
    sal_arr = df[df['industry'] == ind]['salary'].to_numpy()
    ind_stats[ind] = {
        'mean': np.mean(sal_arr), 'median': np.median(sal_arr),
        'std':  np.std(sal_arr),  'min': np.min(sal_arr), 'max': np.max(sal_arr),
        'n':    len(sal_arr)
    }
    print(f"{ind:<18} {len(sal_arr):>7,} ${np.mean(sal_arr):>9,.0f} "
          f"${np.median(sal_arr):>9,.0f} ${np.std(sal_arr):>9,.0f} "
          f"${np.min(sal_arr):>9,.0f} ${np.max(sal_arr):>9,.0f}")

all_sal = df['salary'].to_numpy()
print(f"\nFull Salary Array Shape      : {all_sal.shape}")
print(f"Array Dimensions             : {all_sal.ndim}D")
print(f"Data Type                    : {all_sal.dtype}")
print(f"Overall NumPy Mean           : ${np.mean(all_sal):,.2f}")
print(f"Overall NumPy Std            : ${np.std(all_sal):,.2f}")

sorted_ind = dict(sorted(ind_stats.items(), key=lambda x: x[1]['mean'], reverse=True))
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_ind)))
bars = ax.barh(list(sorted_ind.keys()),
               [v['mean'] for v in sorted_ind.values()],
               color=colors, edgecolor='white')
for bar, val in zip(bars, [v['mean'] for v in sorted_ind.values()]):
    ax.text(val + 400, bar.get_y() + bar.get_height()/2,
            f'${val/1e3:.1f}K', va='center', fontsize=9)
ax.set_title('Experiment 6 - Mean Salary by Industry (NumPy)', fontweight='bold')
ax.set_xlabel('Average Salary (USD)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('exp6_numpy_industry.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 6 Complete - Plot saved as exp6_numpy_industry.png")


# =============================================================================
# EXPERIMENT 7 - Data Cleaning & Validation using NumPy (NaN Handling)
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 7 - Data Cleaning & NaN Handling")
print("="*55)

np.random.seed(0)
df7 = df[['experience_years', 'skills_count', 'certifications', 'salary']].copy()

# Inject NaN (~3%)
for col in ['experience_years', 'skills_count', 'salary']:
    idx = np.random.choice(df7.index, size=int(0.03 * len(df7)), replace=False)
    df7.loc[idx, col] = np.nan

print("\n--- Missing Values BEFORE Cleaning ---")
print(df7.isnull().sum().to_string())
print(f"Total NaN cells      : {df7.isnull().sum().sum():,}")
print(f"Rows with any NaN    : {df7.isnull().any(axis=1).sum():,}")

# Clean - fill with median
df7_cleaned = df7.copy()
for col in df7_cleaned.columns:
    if df7_cleaned[col].isnull().any():
        median_val = df7_cleaned[col].median()
        df7_cleaned[col] = df7_cleaned[col].fillna(median_val)
        print(f"  [CHECK] '{col}' NaNs -> filled with median = {median_val:.2f}")

print(f"\n--- Missing Values AFTER Cleaning ---")
print(df7_cleaned.isnull().sum().to_string())
print(f"Total NaN cells      : {df7_cleaned.isnull().sum().sum()}")

# Non-numeric filtering using NumPy
arr = df7_cleaned['salary'].to_numpy()
numeric_mask = np.isfinite(arr) & (arr > 0)
clean_arr    = arr[numeric_mask]
print(f"\nNumPy Non-Numeric Filtering:")
print(f"  Before : {len(arr):,} values")
print(f"  After  : {len(clean_arr):,} values")
print(f"  Removed: {len(arr) - len(clean_arr):,} invalid entries")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Experiment 7 - NaN Heatmap (200-row sample)", fontweight='bold')
sns.heatmap(df7.sample(200, random_state=1).isnull(),
            cmap='Reds', cbar=False, ax=axes[0], yticklabels=False)
axes[0].set_title('BEFORE Cleaning (NaN = Red)', fontweight='bold')
sns.heatmap(df7_cleaned.sample(200, random_state=1).isnull(),
            cmap='Greens', cbar=False, ax=axes[1], yticklabels=False)
axes[1].set_title('AFTER Cleaning (All Clear)', fontweight='bold')
plt.tight_layout()
plt.savefig('exp7_cleaning.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 7 Complete - Plot saved as exp7_cleaning.png")


# =============================================================================
# EXPERIMENT 8 - Financial/Salary Portfolio Modeling using NumPy
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 8 - Salary Portfolio Modeling (NumPy)")
print("="*55)

top_roles = df['job_title'].value_counts().nlargest(6).index.tolist()
portfolio = {}

print(f"\n{'Job Title':<25} {'Count':>6} {'Mean Salary':>13} {'Std Dev':>12} {'Sharpe*':>9}")
print("-" * 68)

for role in top_roles:
    sal   = df[df['job_title'] == role]['salary'].to_numpy()
    mean  = np.mean(sal)
    std   = np.std(sal)
    sharpe = mean / std
    portfolio[role] = {'mean': mean, 'std': std, 'sharpe': sharpe, 'n': len(sal)}
    print(f"{role:<25} {len(sal):>6,} ${mean:>12,.0f} ${std:>11,.0f} {sharpe:>9.2f}")

print("\n* Sharpe ratio = Mean/Std (higher = stable high salary)")

# Equal-weight portfolio
means   = np.array([v['mean'] for v in portfolio.values()])
stds    = np.array([v['std']  for v in portfolio.values()])
weights = np.ones(len(means)) / len(means)

port_mean = np.dot(weights, means)
port_std  = np.sqrt(np.dot(weights**2, stds**2))

print(f"\nEqual-Weight Portfolio (top 6 roles):")
print(f"  Expected Salary : ${port_mean:,.2f}")
print(f"  Portfolio Risk  : ${port_std:,.2f}")
print(f"  Portfolio Sharpe: {port_mean/port_std:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
colors_p = plt.cm.tab10(range(len(portfolio)))
for i, (role, vals) in enumerate(portfolio.items()):
    ax.scatter(vals['std'], vals['mean'], s=200, color=colors_p[i], zorder=3, label=role)
    ax.annotate(role, (vals['std'], vals['mean']),
                textcoords='offset points', xytext=(6, 4), fontsize=8)
ax.scatter(port_std, port_mean, marker='*', s=400, color='gold',
           edgecolor='black', zorder=5, label='Portfolio (Equal-Weight)')
ax.set_title('Experiment 8 - Salary Portfolio: Risk vs Return', fontweight='bold')
ax.set_xlabel('Risk (Std Dev of Salary)')
ax.set_ylabel('Return (Mean Salary)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig('exp8_portfolio.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 8 Complete - Plot saved as exp8_portfolio.png")


# =============================================================================
# EXPERIMENT 9 - Employee Performance & Salary Correlation using NumPy
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 9 - Correlation & Performance Analysis (NumPy)")
print("="*55)

num_df     = df[['experience_years', 'skills_count', 'certifications', 'salary']]
arr_matrix = num_df.to_numpy()
labels_c   = ['exp_years', 'skills', 'certifications', 'salary']

print("\n--- NumPy Descriptive Statistics ---")
print(f"{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-" * 68)
for i, name in enumerate(labels_c):
    col = arr_matrix[:, i]
    print(f"{name:<20} {np.mean(col):>12.2f} {np.std(col):>12.2f} "
          f"{np.min(col):>12.2f} {np.max(col):>12.2f}")

corr_matrix = np.corrcoef(arr_matrix.T)
print("\n--- Pearson Correlation Matrix (NumPy) ---")
header = f"{'':>18}" + "".join(f"{l:>16}" for l in labels_c)
print(header)
for i, row_lbl in enumerate(labels_c):
    row = f"  {row_lbl:<16}" + "".join(f"{corr_matrix[i,j]:>16.4f}" for j in range(4))
    print(row)

# NumPy polyfit trend
np.random.seed(1)
sample_idx = np.random.choice(len(arr_matrix), 2000, replace=False)
xs = arr_matrix[sample_idx, 0]
ys = arr_matrix[sample_idx, 3]
m, b = np.polyfit(xs, ys, 1)
print(f"\nPolyfit Trend (Experience => Salary):")
print(f"  Slope     : ${m:,.2f} per year")
print(f"  Intercept : ${b:,.2f}")

fig = plt.figure(figsize=(13, 5))
ax1 = fig.add_subplot(1, 2, 1)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=labels_c, yticklabels=labels_c, center=0,
            linewidths=0.5, ax=ax1)
ax1.set_title('Experiment 9 - Pearson Correlation Matrix', fontweight='bold')

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(xs, ys, alpha=0.25, s=10, color='steelblue', label='Sample (n=2000)')
x_line_p = np.linspace(xs.min(), xs.max(), 100)
ax2.plot(x_line_p, m * x_line_p + b, color='tomato', linewidth=2,
         label=f'Trend: +${m:.0f}/yr')
ax2.set_title('Experience vs Salary (NumPy polyfit)', fontweight='bold')
ax2.set_xlabel('Experience (Years)')
ax2.set_ylabel('Salary')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax2.legend()
plt.tight_layout()
plt.savefig('exp9_correlation.png', bbox_inches='tight')
plt.show()
print("[OK] Experiment 9 Complete - Plot saved as exp9_correlation.png")


# =============================================================================
# EXPERIMENT 10 - Job Market Visualisation Dashboard
# =============================================================================
print("\n" + "="*55)
print("  EXPERIMENT 10 - Full Visualisation Dashboard")
print("="*55)

edu_order = ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Experiment 10 - Job Market Salary Intelligence Dashboard',
             fontsize=15, fontweight='bold', y=0.98)

# Plot 1: Salary by Education
ax1 = fig.add_subplot(3, 3, 1)
edu_means_g = df.groupby('education_level')['salary'].mean().reindex(edu_order)
ax1.bar(edu_means_g.index, edu_means_g.values,
        color=sns.color_palette('Blues_d', 5), edgecolor='white')
ax1.set_title('Avg Salary by Education', fontweight='bold', fontsize=10)
ax1.tick_params(axis='x', rotation=30, labelsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

# Plot 2: Salary by Remote Work
ax2 = fig.add_subplot(3, 3, 2)
remote_means = df.groupby('remote_work')['salary'].mean().sort_values(ascending=False)
ax2.bar(remote_means.index, remote_means.values,
        color=sns.color_palette('Set2', 3), edgecolor='white')
ax2.set_title('Salary by Remote Work', fontweight='bold', fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

# Plot 3: Salary by Company Size (Box Plot)
ax3 = fig.add_subplot(3, 3, 3)
size_order = ['Startup', 'Small', 'Medium', 'Large', 'Enterprise']
size_data  = [df[df['company_size'] == s]['salary'].sample(1000, random_state=42)
              for s in size_order]
bp = ax3.boxplot(size_data, labels=size_order, patch_artist=True,
                 medianprops={'color': 'black', 'linewidth': 1.5})
for patch, col in zip(bp['boxes'], sns.color_palette('pastel', 5)):
    patch.set_facecolor(col)
ax3.set_title('Salary by Company Size', fontweight='bold', fontsize=10)
ax3.tick_params(axis='x', rotation=20, labelsize=8)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

# Plot 4: Top 8 Job Titles
ax4 = fig.add_subplot(3, 3, 4)
top8 = df.groupby('job_title')['salary'].mean().nlargest(8).sort_values()
ax4.barh(top8.index, top8.values,
         color=sns.color_palette('YlOrRd_r', 8), edgecolor='white')
ax4.set_title('Top 8 Job Titles by Salary', fontweight='bold', fontsize=10)
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax4.tick_params(axis='y', labelsize=7)

# Plot 5: Heatmap - Industry x Education
ax5 = fig.add_subplot(3, 3, 5)
pivot = df.pivot_table(values='salary', index='industry',
                       columns='education_level', aggfunc='mean')[edu_order]
sns.heatmap(pivot / 1000, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.4, ax=ax5, cbar_kws={'label': 'Salary ($K)'})
ax5.set_title('Heatmap: Industry x Education (Avg $K)', fontweight='bold', fontsize=10)
ax5.tick_params(axis='x', rotation=30, labelsize=8)
ax5.tick_params(axis='y', labelsize=8)

# Plot 6: Certifications vs Salary
ax6 = fig.add_subplot(3, 3, 6)
cert_means_g = df.groupby('certifications')['salary'].mean()
ax6.plot(cert_means_g.index, cert_means_g.values, marker='o',
         color='teal', linewidth=2, markersize=8)
ax6.fill_between(cert_means_g.index, cert_means_g.values,
                 alpha=0.15, color='teal')
ax6.set_title('Certifications vs Avg Salary', fontweight='bold', fontsize=10)
ax6.set_xlabel('Number of Certifications')
ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

# Plot 7: Experience vs Salary (Scatter coloured by Skills)
ax7 = fig.add_subplot(3, 3, 7)
s3k = df.sample(3000, random_state=42)
sc  = ax7.scatter(s3k['experience_years'], s3k['salary'],
                  c=s3k['skills_count'], cmap='viridis', alpha=0.5, s=12)
plt.colorbar(sc, ax=ax7, label='Skills Count')
ax7.set_title('Experience vs Salary\n(color = skills)', fontweight='bold', fontsize=10)
ax7.set_xlabel('Experience (Yrs)')
ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

# Plot 8: Top Locations
ax8 = fig.add_subplot(3, 3, 8)
top_loc = df.groupby('location')['salary'].mean().nlargest(8).sort_values()
ax8.barh(top_loc.index, top_loc.values,
         color=sns.color_palette('crest', 8), edgecolor='white')
ax8.set_title('Top 8 Locations by Salary', fontweight='bold', fontsize=10)
ax8.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax8.tick_params(axis='y', labelsize=8)

# Plot 9: Salary KDE by Education
ax9 = fig.add_subplot(3, 3, 9)
pal_e = sns.color_palette('tab10', 5)
for edu_e, col_e in zip(edu_order, pal_e):
    sub_e = df[df['education_level'] == edu_e]['salary'].sample(
        min(3000, (df['education_level'] == edu_e).sum()), random_state=42)
    sub_e.plot.kde(ax=ax9, label=edu_e, color=col_e, linewidth=1.8)
ax9.set_title('Salary KDE by Education Level', fontweight='bold', fontsize=10)
ax9.set_xlabel('Salary')
ax9.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
ax9.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('exp10_dashboard.png', bbox_inches='tight', dpi=130)
plt.show()
print("[OK] Experiment 10 Complete - Plot saved as exp10_dashboard.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*55)
print("  PROJECT SUMMARY - ALL 10 EXPERIMENTS COMPLETE")
print("="*55)
summary = [
    ("Exp 1", "EDA & Setup",              "Salary dist, box plot"),
    ("Exp 2", "Logistic Regression",      "Binary classification, confusion matrix"),
    ("Exp 3", "Linear Regression",        f"R2={r2:.4f}, +${lin_model.coef_[0]:,.0f}/yr"),
    ("Exp 4", "Random Sampling",          "Simple, Stratified, Systematic"),
    ("Exp 5", "Z-Test",                   f"Z={z_stat:.2f}, p={p_value:.4f}"),
    ("Exp 6", "NumPy Mean/Shape",         "Industry-wise NumPy stats"),
    ("Exp 7", "Data Cleaning",            "NaN inject => median impute => verify"),
    ("Exp 8", "Portfolio Modeling",       "Risk-Return Sharpe analysis"),
    ("Exp 9", "Correlation (NumPy)",      "corrcoef matrix + polyfit trend"),
    ("Exp 10","Visualisation Dashboard",  "9-panel heatmap + KDE + scatter"),
]
print(f"\n{'Exp':<8} {'Title':<25} {'Key Outcome'}")
print("-"*65)
for exp, title, outcome in summary:
    print(f"{exp:<8} {title:<25} {outcome}")
print("\n  Dataset : 250,000 records x 10 features")
print("  Output  : 10 chart PNGs saved in current folder")
print("="*55)
