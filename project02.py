import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# 1. Sample data: Group A vs Group B
np.random.seed(42)
data = pd.DataFrame({
    'Value': np.concatenate([
        np.random.normal(50, 10, 200),  # Group A
        np.random.normal(70, 15, 200)   # Group B
    ]),
    'Group': ['A']*200 + ['B']*200
})

# Add some outliers
data.loc[0, 'Value'] = 150
data.loc[200, 'Value'] = 5

# 2. Histogram + KDE
plt.figure()
sns.histplot(data, x='Value', hue='Group', kde=True, bins=30, element='step')
plt.title('Histogram with KDE - Value Distribution by Group')
plt.savefig('hist_kde.png')
plt.show()

# 3. Boxplot for outlier detection
plt.figure()
sns.boxplot(x='Group', y='Value', data=data)
plt.title('Boxplot - Outlier Detection by Group')
plt.savefig('boxplot.png')
plt.show()

# 4. Distribution stats
print("Descriptive Statistics:")
print(data.groupby('Group')['Value'].describe())

print("\nSkewness:")
print(data.groupby('Group')['Value'].skew())