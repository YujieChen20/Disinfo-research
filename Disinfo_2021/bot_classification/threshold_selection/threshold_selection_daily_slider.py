import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, date
import plotly.express as px
import plotly.graph_objects as go


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
    #temp = pd.read_csv(filePath).drop(['row_id'], axis=1)
    #df = pd.concat([df, temp])
    temp = pd.read_csv(filePath).drop(['row_id'], axis=1)
    temp['date'] = str(date)
    df = pd.concat([df, temp])


# create the bins
fig = px.histogram(df, x="bot_probability", animation_frame="date",
             labels={"date": "DATE", "bot_probability": "Bot probability"}, log_y=True, nbins=20)
fig.update_layout(bargap=0.1)
fig.update_traces(xbins=dict( # bins used for histogram
        start=0.0,
        end=1.0,
        size=0.05
    ))
#fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.show()
fig.write_html(os.path.join(os.path.dirname(__file__), "probabilities_histogram_log", "slider.html"))
