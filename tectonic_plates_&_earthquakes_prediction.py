# -*- coding: utf-8 -*-
"""Tectonic_Plates_&_Earthquakes_Prediction.ipynb


Import Libraries
"""

!pip install basemap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from datetime import datetime
import time
from mpl_toolkits.basemap import Basemap
import plotly.express as px

import geopandas as gpd
import folium
from folium import Choropleth
from folium.plugins import HeatMap

import warnings
warnings.filterwarnings('ignore')

"""Read Data

Earthquake Dataset
"""

data1 =pd.read_csv("database.csv")
data1.head()

data1.info()

"""Tectonic plate Dataset"""

tectonic_plates = pd.read_csv("all.csv" )
tectonic_plates.head()

tectonic_plates.info()

"""Data Preprocessing

EDA
"""

#Summary Statistics for Numerical Variables
data1.describe().transpose()

#Summary Statistics for Categorical Variables
data1.describe(include='object').transpose()

#Drop ID 1 unique
data1.drop("ID", axis=1,inplace=True)
data1.head()

data1.isna().sum()

null_columns = data1.loc[:, data1.isna().sum() > 0.66 * data1.shape[0]].columns
print(null_columns)

data1 = data1.drop(null_columns, axis=1)
data1.isna().sum()

data1['Root Mean Square'] = data1['Root Mean Square'].fillna(data1['Root Mean Square'].mean())
data1 = data1.dropna(axis=0).reset_index(drop=True)
data1.isna().sum().sum()

data1.head()

#Summary Statistics for Numerical Variables
tectonic_plates.describe().transpose()

#Summary Statistics for Categorical Variables
tectonic_plates.describe(include='object').transpose()

tectonic_plates.isna().sum()

data1['Type'].unique()

"""Parsing Datetime

Date columns
"""

#Exploring the length of date objects
lengths = data1["Date"].str.len()
lengths.value_counts()

#Find wrong date length 24 in 3 columns
wrongdates = np.where([lengths == 24])[1]
print("Find dates:", wrongdates)
data1.loc[wrongdates]

#fixing the wrong dates and changing the datatype from numpy object to datetime64[ns]
data1.loc[3378, "Date"] = "02/23/1975"
data1.loc[7510, "Date"] = "04/28/1985"
data1.loc[20647, "Date"] = "03/13/2011"
data1['Date']= pd.to_datetime(data1["Date"])
data1.info()

"""Time Columns"""

#Exploring the length of time objects
lengths1 = data1["Time"].str.len()
lengths1.value_counts()

#Find wrong time length 24 in 3 columns
wrongtime = np.where([lengths1 == 24])[1]
print("Find time:", wrongtime)
data1.loc[wrongtime]

#fixing the wrong time and changing the datatype from numpy object to timedelta64[ns]
data1.loc[3378, "Time"] = "02:58:41"
data1.loc[7510, "Time"] = "02:53:41"
data1.loc[20647, "Time"] = "02:23:34"
data1['Time']= pd.to_timedelta(data1['Time'])
data1.info()

"""Concate To Datetime Columns"""

data1["Datetime"]=data1["Date"] +data1["Time"]

data1.head()

data1.info()

"""Visualization"""

#Heatmap only numerical
numeric_columns = [column for column in data1.columns if data1.dtypes[column] != 'object']
corr = data1[numeric_columns].corr()
plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0, fmt='.2f')
plt.show()

"""Heatmap shows numerical and object"""

datacorr=data1.copy()

from sklearn.preprocessing import LabelEncoder
categorical_columns = datacorr.select_dtypes(include=['object']).columns.tolist()
label_encoder = LabelEncoder()
for column in categorical_columns:
    datacorr[column] = label_encoder.fit_transform(datacorr[column])

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(datacorr.corr()),k=1)
sns.heatmap(datacorr.corr(), annot=True, mask=mask, fmt='.2f')
plt.show()

sns.set(palette='BrBG')
data1.hist(figsize=(13,15));

display(px.pie(data1,names = "Type",title = "Types",color ="Type" ,hole = .4))
display(data1["Type"].value_counts())

fig = px.scatter(data1,x = "Date",y = "Magnitude",color = "Type")
fig.show()
data1["Magnitude"].value_counts().head(20)

"""Type of Earthquakes classify  4 type by nature 1.Earthquake and by human (between 1965 - 1995 ) 2.Nuclear Explosions 3.Explosions 4.Rock Bursts


"""

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x = data1.Depth,y = data1.Magnitude,mode = "markers",marker_color = data1.Magnitude,text = data1.Type,marker = dict(showscale = True)))
fig.update_layout(title = "Depth VS Magnitude",xaxis_title="Depth",yaxis_title="Magnitude")

plt.figure(figsize = (8,5))
sns.kdeplot(data = data1,x = "Depth",y = "Magnitude",cmap = "coolwarm",shade = True)
plt.title("Density of Depth VS Magnitude")
plt.show()

"""Density plot between Depth vs Magnitude : Depth Values range 0-100 and Magnitude values range 5.5-6.0"""

sns.pairplot(data=data1,hue='Magnitude',kind='scatter',palette='BrBG')

"""Tectonic plate





"""

#Ploting the tectonic plate's boundaries
tectonic = folium.Map(tiles="cartodbpositron", zoom_start=5)

plates = list(tectonic_plates["plate"].unique())
for plate in plates:
    plate_vals = tectonic_plates[tectonic_plates["plate"] == plate]
    lats = plate_vals["lat"].values
    lons = plate_vals["lon"].values
    points = list(zip(lats, lons))
    indexes = [None] + [i + 1 for i, x in enumerate(points) if i < len(points) - 1 and abs(x[1] - points[i + 1][1]) > 300] + [None]
    for i in range(len(indexes) - 1):
        folium.vector_layers.PolyLine(points[indexes[i]:indexes[i+1]], popup=plate, color="#58508d", fill=False, ).add_to(tectonic)

tectonic

#Data to the eartquake type to only earthquake and dropping others as "Nuclear Explosion","Explosion" and "Rock Burst"
data_onlyquakes= data1.set_index("Type")
data_onlyquakes=data_onlyquakes.drop(["Nuclear Explosion","Explosion","Rock Burst"],axis=0)

#plotting the the eartquake type to only earthquake
tectonic_quake = folium.Map(tiles="cartodbpositron", zoom_start=5)
gradient = {.33: "#7a5195", .66: "#ef5675", 1: "#ffa600"}
plates = list(tectonic_plates["plate"].unique())

for plate in plates:
    plate_vals = tectonic_plates[tectonic_plates["plate"] == plate]
    lats = plate_vals["lat"].values
    lons = plate_vals["lon"].values
    points = list(zip(lats, lons))
    indexes = [None] + [i + 1 for i, x in enumerate(points) if i < len(points) - 1 and abs(x[1] - points[i + 1][1]) > 300] + [None]
    for i in range(len(indexes) - 1):
        folium.vector_layers.PolyLine(points[indexes[i]:indexes[i+1]], popup=plate, fill=False,  color="#58508d").add_to(tectonic_quake)
        HeatMap(data=data_onlyquakes[["Latitude", "Longitude"]], min_opacity=0.5,max_zoom=40,max_val=0.5,radius=1,gradient=gradient).add_to(tectonic_quake)

tectonic_quake

# Create a base map with plate boundaries and Magnitude
Mag_tectonics  = folium.Map(tiles="cartodbpositron", zoom_start=5)
gradient = {.33: "#628d82", .66: "#a3c5bf", 1: "#eafffd"}
plates = list(tectonic_plates["plate"].unique())
for plate in plates:
    plate_vals = tectonic_plates[tectonic_plates["plate"] == plate]
    lats = plate_vals["lat"].values
    lons = plate_vals["lon"].values
    points = list(zip(lats, lons))
    indexes = [None] + [i + 1 for i, x in enumerate(points) if i < len(points) - 1 and abs(x[1] - points[i + 1][1]) > 300] + [None]
    for i in range(len(indexes) - 1):
        folium.vector_layers.PolyLine(points[indexes[i]:indexes[i+1]], fill=False,  color="#58508d").add_to(Mag_tectonics)

def colormag(val):
            if val < 5.9:
                return "#ffcf6a"
            elif val < 6.5:
                return "#fb8270"
            else:
                return "#bc5090"

for i in range(0,len(data1)):
    folium.Circle(location=[data1.iloc[i]["Latitude"], data1.iloc[i]["Longitude"]],radius=2000, color=colormag(data1.iloc[i]["Magnitude"])).add_to(Mag_tectonics)

#A bit of extra code to get legent
from branca.element import Template, MacroElement

template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>jQuery UI Draggable - Default functionality</title>
  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

</head>
<body>


<div id='maplegend' class='maplegend'
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

<div class='legend-title'>Magnitude Scale </div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:#ffcf6a;opacity:1;'></span>values less than 5.9</li>
    <li><span style='background:#fb8270;opacity:1;'></span>values less than 6.5</li>
    <li><span style='background:#bc5090;opacity:1;'></span>values more than 6.5</li>

  </ul>
</div>
</div>

</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

macro = MacroElement()
macro._template = Template(template)

Mag_tectonics.get_root().add_child(macro)

Mag_tectonics

# Create a base map with plate boundaries and deapth
Depth_tectonics  = folium.Map(tiles="cartodbpositron", zoom_start=5)
gradient = {.33: "#628d82", .66: "#a3c5bf", 1: "#eafffd"}
plates = list(tectonic_plates["plate"].unique())
for plate in plates:
    plate_vals = tectonic_plates[tectonic_plates["plate"] == plate]
    lats = plate_vals["lat"].values
    lons = plate_vals["lon"].values
    points = list(zip(lats, lons))
    indexes = [None] + [i + 1 for i, x in enumerate(points) if i < len(points) - 1 and abs(x[1] - points[i + 1][1]) > 300] + [None]
    for i in range(len(indexes) - 1):
        folium.vector_layers.PolyLine(points[indexes[i]:indexes[i+1]],fill=False,  color="#58508d").add_to(Depth_tectonics)

def colordepth(val):
            if val < 50:
                return "#ffcf6a"
            elif val < 100:
                return "#fb8270"
            else:
                return "#bc5090"

for i in range(0,len(data1)):
    folium.Circle(location=[data1.iloc[i]["Latitude"], data1.iloc[i]["Longitude"]],radius=2000, color=colordepth(data1.iloc[i]["Depth"])).add_to(Depth_tectonics)

#A bit of extra code to get legent
from branca.element import Template, MacroElement

template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>jQuery UI Draggable - Default functionality</title>
  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

</head>
<body>


<div id='maplegend' class='maplegend'
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

<div class='legend-title'>Depth Scale </div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:#ffcf6a;opacity:1;'></span>Depth less than 50</li>
    <li><span style='background:#fb8270;opacity:1;'></span>Depth less than 100</li>
    <li><span style='background:#bc5090;opacity:1;'></span>Depth more than 100</li>

  </ul>
</div>
</div>

</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

macro = MacroElement()
macro._template = Template(template)

Depth_tectonics.get_root().add_child(macro)

Depth_tectonics

"""Type And Magnitude without Tectonic plate boundaries"""

import matplotlib.patches as mpatches

eq = data1[data1['Type'] == 'Earthquake']
others = data1[data1['Type'] != 'Earthquake']

fig = plt.figure(figsize = (22, 20))
wmap = Basemap()

longitudes = eq['Longitude'].tolist()
latitudes = eq['Latitude'].tolist()
x_eq, y_eq = wmap(longitudes, latitudes)

longitudes = others['Longitude'].tolist()
latitudes = others['Latitude'].tolist()
x_oth, y_oth = wmap(longitudes, latitudes)


plt.title('Earthquake effective Areas')
wmap.drawcoastlines()
wmap.shadedrelief()
wmap.scatter(x_eq, y_eq, s = 5, c = 'r', alpha = 0.2)
wmap.scatter(x_oth, y_oth, s = 10, c = 'g')

# draw parallels
wmap.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])
# draw meridians
wmap.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
ax = plt.gca()
red_patch = mpatches.Patch(color='r', label='Earthquake')
green_patch = mpatches.Patch(color='g', label='Nuclear Explosion/Rockburst/Others')

handles=[red_patch, green_patch]
labels = ['Earthquake', 'Nuclear Explosion/Rockburst/Others']
plt.legend(handles, labels, loc='upper left')
plt.show()

fig = plt.figure(figsize = (22, 20))
wmap = Basemap()
longitudes = eq['Longitude'].tolist()
latitudes = eq['Latitude'].tolist()
x_eq, y_eq = wmap(longitudes, latitudes)
wmap.drawcoastlines()
wmap.shadedrelief()
# draw parallels
wmap.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])
# draw meridians
wmap.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
plt.title('Earthquake effective Areas with Magnitude Colormap')
sc =wmap.scatter(x_eq, y_eq, s = 30, c = eq['Magnitude'], vmin=5, vmax =9, cmap='OrRd', edgecolors='none')
cbar = plt.colorbar(sc, shrink = .5)
cbar.set_label('Magnitude')
plt.show()

"""Earthquake sensitive areas are western coast of North and South America, center of Atlantic Ocean, Himalian region and Eastern Asian Countries like Indonesia, Japan, Korea.

Magnitude Classes
*   Disastrous: M > =8
*   Major: 7 < =M < 7.9
*   Strong: 6 < = M < 6.9
*   Moderate: 5.5 < =M < 5.9
"""

data1.loc[data1['Magnitude'] >=8, 'Class'] = 'Disastrous'
data1.loc[ (data1['Magnitude'] >= 7) & (data1['Magnitude'] < 7.9), 'Class'] = 'Major'
data1.loc[ (data1['Magnitude'] >= 6) & (data1['Magnitude'] < 6.9), 'Class'] = 'Strong'
data1.loc[ (data1['Magnitude'] >= 5.5) & (data1['Magnitude'] < 5.9), 'Class'] = 'Moderate'

# Magnitude Class distribution
sns.countplot(x="Class", data=data1)
plt.ylabel('Frequency')
plt.title('Magnitude Class VS Frequency')

plt.hist(data1['Magnitude'])
plt.xlabel('Magnitude Size')
plt.ylabel('Number of Occurrences')

"""Model and Predict"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

datacorr = data1[['Latitude', 'Longitude', 'Depth', 'Magnitude']]
datacorr.head()

X, y = datacorr.drop(labels='Depth', axis=1), datacorr['Depth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.dtypes)
print(y_train.dtypes)

results = []

models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Gradient Boost', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=42)),
    ('XGBoost', XGBRegressor(random_state=42)),
    ('KNN',KNeighborsRegressor(n_neighbors=5)),
    ('Decision Tree',DecisionTreeRegressor(random_state=42)),
    ('Bagging Regressor',BaggingRegressor(n_estimators=150, random_state=42))
          ]

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    MSE = mean_squared_error(y_test, y_pred)
    R2_score = r2_score(y_test, y_pred)
    results.append((name, accuracy, MSE, R2_score))
    acc = (model.score(X_train , y_train)*100)
    print(f'The accuracy of the {name} Model Train is {acc:.2f}')
    acc =(model.score(X_test , y_test)*100)
    print(f'The accuracy of the  {name} Model Test is {acc:.2f}')
    plt.scatter(y_test, y_pred,s=10,color='#9B673C')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linewidth = 4)
    plt.show()

dff = pd.DataFrame(results, columns=['Model', 'Accuracy', 'MSE', 'R2_score'])
df_styled_best = dff.style.highlight_max(subset=['Accuracy','R2_score'], color='green').highlight_min(subset=['MSE'], color='green').highlight_max(subset=['MSE'], color='red').highlight_min(subset=['Accuracy','R2_score'], color='red')
display(df_styled_best)
