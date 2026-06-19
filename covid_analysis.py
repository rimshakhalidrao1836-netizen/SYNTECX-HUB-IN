import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 1. Load Data
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df = pd.read_csv(url)

print("Data Loaded. Shape:", df.shape)
print(df.head())

# 2. Clean and reshape data
df = df.drop(columns=['Lat', 'Long'])
df = df.melt(id_vars=['Province/State', 'Country/Region'], 
             var_name='Date', value_name='Cases')
df['Date'] = pd.to_datetime(df['Date'])

# Group by country
country_df = df.groupby(['Country/Region', 'Date']).sum().reset_index()

# 3. Select countries to compare
countries = ['US', 'India', 'Brazil', 'Pakistan', 'Italy']
filtered_df = country_df[country_df['Country/Region'].isin(countries)]

# 4. Compute daily and weekly cases
filtered_df['Daily_Cases'] = filtered_df.groupby('Country/Region')['Cases'].diff().fillna(0)
filtered_df['Weekly_Avg'] = filtered_df.groupby('Country/Region')['Daily_Cases'] \
                                      .transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# 5. Plot country comparisons - Total Cases
plt.figure(figsize=(14, 7))
for country in countries:
    data = filtered_df[filtered_df['Country/Region'] == country]
    plt.plot(data['Date'], data['Cases'], label=country)
plt.title('Total COVID-19 Cases by Country', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.tight_layout()
plt.savefig('total_cases_comparison.png')
plt.show()

# 6. Plot rolling average - Daily Cases
plt.figure(figsize=(14, 7))
for country in countries:
    data = filtered_df[filtered_df['Country/Region'] == country]
    plt.plot(data['Date'], data['Weekly_Avg'], label=country)
plt.title('7-Day Rolling Average of Daily COVID-19 Cases', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Daily Cases (7-day avg)')
plt.legend()
plt.tight_layout()
plt.savefig('rolling_avg_comparison.png')
plt.show()

# 7. Detect peaks
peaks = filtered_df.loc[filtered_df.groupby('Country/Region')['Daily_Cases'].idxmax()]
peaks = peaks[['Country/Region', 'Date', 'Daily_Cases']].sort_values('Daily_Cases', ascending=False)
print("\nPeak Daily Cases by Country:")
print(peaks)

# 8. Basic reproduction insight - simple comparison
# R0 approximation: avg new cases in last 7 days / avg new cases in previous 7 days
latest_date = filtered_df['Date'].max()
recent = filtered_df[filtered_df['Date'] >= latest_date - pd.Timedelta(days=14)]

r0_estimate = {}
for country in countries:
    data = recent[recent['Country/Region'] == country]['Daily_Cases']
    if len(data) >= 14:
        recent_avg = data.tail(7).mean()
        prev_avg = data.head(7).mean()
        r0_estimate[country] = round(recent_avg / prev_avg, 2) if prev_avg > 0 else 0

print("\nBasic Reproduction Trend (R0 approx):")
for k, v in r0_estimate.items():
    status = "Growing" if v > 1 else "Declining"
    print(f"{k}: {v} - Cases are {status}")

# 9. Export summary to CSV
peaks.to_csv('peak_cases_summary.csv', index=False)
pd.DataFrame(r0_estimate.items(), columns=['Country', 'R0_Approx']).to_csv('r0_estimate.csv', index=False)

print("\nDone! Charts and CSV files saved in the project folder.")
print("Conclusions: Check the generated PNG files and CSVs for insights.")