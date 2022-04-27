import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

df = pandas.read_csv('./data/data.csv', parse_dates=['dateRep'])
df = df.drop(columns=['day', 'month', 'year', 'geoId', 'countryterritoryCode', 'continentExp'])

df = df.sort_values(by="dateRep")
df = df.loc[df['countriesAndTerritories'] == 'Germany']
df = df.reset_index(drop=True)
df = df.rename({"countriesAndTerritories": "country",
                "dateRep": "date",
                "popData2019": "pop2019",
                "Cumulative_number_for_14_days_of_COVID-19_cases_per_100000": "covid_14_days"},
                axis='columns')
# df = df.head(50)
print(df)
rows = []
deaths = 0
cases = 0

for index, data in df.iterrows():
    deaths += data['deaths']
    cases += data['cases']
    rows.append([data['date'], deaths, cases])

print("Number of days found {:}".format(len(rows)))

# df_1 = pandas.DataFrame(rows, columns=["date", "deaths", "cases"])

# plt.plot(df_1['date'], df_1['deaths'], label='deaths')
# # plt.plot(df_1['date'], df_1['cases'], label='cases')

# plt.xlabel('Date')
# plt.ylabel('Number of deaths')
# plt.title('Deaths of AI terms by date')
# plt.grid(True)
# plt.legend()
# plt.show()


deaths = df_1['deaths'].to_numpy()
deaths_per_day = deaths[1:] - deaths[:-1]
days = np.arange(len(deaths_per_day))

# plt.plot(days, deaths_per_day, label='deaths')
# plt.xlabel('Date')
# plt.ylabel('Number of deaths per day')
# plt.grid(True)
# plt.legend()
# plt.show()



window_size = 21
# window size, polynomial order 3
deaths_per_day_smooth = savgol_filter(deaths_per_day, window_size, 3) 


plt.plot(days, deaths_per_day_smooth, label='deaths (smooth)')
plt.plot(days, deaths_per_day, label='deaths')
plt.xlabel('Date')
plt.ylabel('Number of deaths per day')
plt.grid(True)
plt.legend()
plt.show()

