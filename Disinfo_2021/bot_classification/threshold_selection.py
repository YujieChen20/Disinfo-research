import pandas as pd
from datetime import timedelta, date
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Generate Date (11/15/2020 - 05/15/2021)
def dateRange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)


dateList = []
start_dt = date(2020, 11, 15)
end_dt = date(2021, 5, 15)
for dt in dateRange(start_dt, end_dt):
    dateList.append(dt.strftime("%Y-%m-%d"))

# Path
dirPath = os.path.join(os.path.dirname(__file__), "probabilities")

# Read csv file and
df = pd.DataFrame()
for date in dateList:
    fileName = "probabilities_" + str(date) + ".csv"
    filePath = os.path.join(dirPath, fileName)
    temp = pd.read_csv(filePath).drop(['row_id'], axis=1)
    df = pd.concat([df, temp])

probabilities = df["bot_probability"]
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt.hist(probabilities, edgecolor="black", bins=[i/100 for i in range(0, 101, 5)])
plt.yscale('log')
plt.xticks([i / 100 for i in range(0, 101, 5)])
plt.xlabel("Bot probability")
plt.ylabel("User Count")
plt.title('Distribution of Bot Probability Scores (log scale)')
plt.savefig(os.path.join(os.path.dirname(__file__), "probabilities_histogram_log.png"))

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt.hist(probabilities, edgecolor="black", bins=[i/100 for i in range(0, 101, 5)])
plt.ylim([0, 8000])
plt.xticks([i / 100 for i in range(0, 101, 5)])
plt.xlabel("Bot probability")
plt.ylabel("User Count")
plt.title('Distribution of Bot Probability Scores (y-axis capped)')
plt.savefig(os.path.join(os.path.dirname(__file__), "probabilities_histogram_capped.png"))
