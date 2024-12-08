import plotly
from plotly import subplots

#Создание простого графика
from plotly.graph_objs import Scatter, Layout  #1


def easy():
    plotly.offline.plot({
        'data': [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
        'layout': Layout(title='hello world')
    })


easy()
#Круговая диаграмма
def wall():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    labels = ["US", "China", "European Union", "Russian Federation",
              "Brazil", "India",
              "Rest of World"]
    # Создание подзаголовков: используйте тип "domain" для кругового подзаголовка
    fig = make_subplots(rows=1, cols=2, specs=
    [[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=labels, values=[16, 15, 12, 6, 5, 4, 42], name="GHG Emissions"),
                  1, 1)
    fig.add_trace(go.Pie(labels=labels, values=[27, 11, 25, 8, 1, 3, 25], name="CO2 Emissions"),
                  1, 2)
    # Используйте `hole` для создания круговой диаграммы, похожей на пончик
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(
        title_text="Global Emissions 1990-2011",
        # Добавьте надписи в центр пирожков с пончиками.
        annotations=[dict(text='GHG', x=0.18, y=0.5, font_size=20, showarrow=False),
                     dict(text='CO2', x=0.82, y=0.5, font_size=20, showarrow=False)])
    fig.show()


#wall()


#Гистограмма(данные о населении Канады)
def gistogramm():
    import plotly.express as px
    df = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(df, x='year', y='pop',
                 hover_data=['lifeExp', 'gdpPercap'],
                 color='lifeExp',
                 labels={'pop': 'population of Canada'}, height=400)
    fig.show()


#gistogramm()


#Точечная диаграмма
import numpy as np  #2
import plotly
import plotly.graph_objs as go


def scatter_1():
    N = 1000
    random_x = np.random.randn(N)
    random_y = np.random.randn(N)
    trace = go.Scatter(x=random_x, y=random_y, mode='markers')
    data = [trace]
    plotly.offline.plot(data, filename='basic-scatter.html')


#scatter_1()


#Линия и точечная диаграмма
def scatlay():
    N = 100
    random_x = np.linspace(0, 1, N)
    random_y0 = np.random.randn(N) + 5
    random_y1 = np.random.randn(N)
    random_y2 = np.random.randn(N) - 5
    # Создание следов
    trace0 = go.Scatter(x=random_x, y=random_y0, mode='markers', name='markers')
    trace1 = go.Scatter(x=random_x, y=random_y1, mode='lines+markers', name='lines+markers')
    trace2 = go.Scatter(x=random_x, y=random_y2, mode='lines', name='lines')

    data = [trace0, trace1, trace2]
    plotly.offline.plot(data, filename='scatter-mode.html')


#scatlay()

#Коробчатые графики
import random
from numpy import *


def box():
    N = 30
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in linspace(0, 360, N)]
    data = [{
        'y': 3.5 * sin(pi * i / N) + i / N +
             (1.5 + 0.5 * cos(pi * i / N)) * random.rand(10),
        'type': 'box',
        'marker': {'color': c[i]}
    } for i in range(int(N))]

    layout = {'xaxis': {'showgrid': False, 'zeroline': False,
                        'tickangle': 60, 'showticklabels': False},
              'yaxis': {'zeroline': False, 'gridcolor': 'white'},
              'paper_bgcolor': 'rgb(233,233,233)',
              'plot_bgcolor': 'rgb(233,233,233)', }
    plotly.offline.plot(data)


#box()


#Контурный график
def conturing():
    from plotly import tools
    import plotly
    import plotly.graph_objs as go
    from plotly import subplots
    trace0 = go.Contour(z=[[2, 4, 7, 12, 13, 14, 15, 16],
                           [3, 1, 6, 11, 12, 13, 16, 17],
                           [4, 2, 7, 7, 11, 14, 17, 18],
                           [5, 3, 8, 8, 13, 15, 18, 19],
                           [7, 4, 10, 9, 16, 18, 20, 19],
                           [9, 10, 5, 27, 23, 21, 21, 21],
                           [11, 14, 17, 26, 25, 24, 23, 22]],
                        line=dict(smoothing=0),
                        )
    trace1 = go.Contour(z=[[2, 4, 7, 12, 13, 14, 15, 16],
                           [3, 1, 6, 11, 12, 13, 16, 17],
                           [4, 2, 7, 7, 11, 14, 17, 18],
                           [5, 3, 8, 8, 13, 15, 18, 19],
                           [7, 4, 10, 9, 16, 18, 20, 19],
                           [9, 10, 5, 27, 23, 21, 21, 21],
                           [11, 14, 17, 26, 25, 24, 23, 22]],
                        line=dict(smoothing=0.85),
                        )
    data = subplots.make_subplots(rows=1, cols=2, subplot_titles=('Without Smoothing', 'With Smoothing'))
    data.append_trace(trace0, 1, 1)
    data.append_trace(trace1, 1, 2)
    plotly.offline.plot(data)


#conturing()

#График временных рядов


#import plotly.graph_objs as go
import pandas as pd


def time_():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
    data = [go.Scatter(
        x=df.Date,
        y=df['AAPL.Close'])]
    plotly.offline.plot(data)


#time_()




#График OHLC(Open-High-Low-Close)

#import plotly.graph_objs as go
from datetime import datetime


def ohlc():
    open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
    high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
    low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
    close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
    dates = [datetime(year=2013, month=10, day=10),
             datetime(year=2013, month=11, day=10),
             datetime(year=2013, month=12, day=10),
             datetime(year=2014, month=1, day=10),
             datetime(year=2014, month=2, day=10)]
    trace = go.Ohlc(x=dates,
                    open=open_data,
                    high=high_data,
                    low=low_data,
                    close=close_data)
    data = [trace]
    plotly.offline.plot(data, filename='ohlc_datetime.html')


#ohlc()
#Усложнённый линейный график
#pip install dash
from dash import Dash, dcc, html, Input, Output
import plotly.express as px


def hard():
    app = Dash(__name__)
    app.layout = html.Div([
        html.H4('Продолжительность жизни в странах по континентам'),
        dcc.Graph(id='graph'),
        dcc.Checklist(
            id='checklist',
            options=['Asia’', 'Europe', 'Africa', 'Americas', 'Oceania'],
            value=['Americas', 'Oceania'],
            inline=True),
    ])

    @app.callback(
        Output("graph", "figure"),
        Input("checklist", "value"))
    def update_line_chart(continents):
        df = px.data.gapminder()
        mask = df.continent.isin(continents)
        fig = px.line(df[mask],
                      x="year", y="lifeExp", color='country')
        return fig

    app.run_server(debug=True)


#hard()
#Поверхностная диаграмма(гора Бруно)
def hard2():
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    # Read data from a csv
    z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
    z = z_data.values
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()


#hard2()

#Карты
def hard3():
    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                     dtype={"fips": str})
    import plotly.express as px
    fig = px.choropleth_mapbox(df, geojson=counties, locations='fips', color='unemp',
                               color_continuous_scale="Viridis",
                               range_color=(0, 12),
                               mapbox_style="carto-positron",
                               zoom=3, center={"lat": 37.0902, "lon": -95.7129},
                               opacity=0.5,
                               labels={'unemp': 'unemployment rate'}
                               )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

#hard3()
