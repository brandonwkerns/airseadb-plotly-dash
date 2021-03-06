from httplib2 import Response
import numpy as np
import pandas as pd
import cftime
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc
import sqlite3
import os
from dash import Dash, html, dcc, Input, Output

from plotting_functions import *  ## Includes color palette.

app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

data_dir = './data'
html_out_dir = '/home/orca/bkerns/public_html/projects/noaa_air_sea_flux/report_feb_2022/interactive_plots'

cruise_list = ['dynamo_2011',]

mapbox_access_token = 'pk.eyJ1IjoiYnJhbmRvbndrZXJucyIsImEiOiJja3ZyOGNlZmcydTdrMm5xZ3d4ZWNlZXZpIn0.OkA0r6XFSY-Dx0bk7UPPZQ'

############### 1. The map and data visualization ###################

##
## 1.1. Get the data (or subset of data) from AirSeaDB
##

## Connect to SQLite database
fn = (data_dir + '/AirSeaDB.sqlite')
con = sqlite3.connect(fn)

## Query the database.

query = '''
    SELECT lon,lat,t_sea_snake,wspd_sonic,decimal_day_of_year,original_file_name
        FROM DATA WHERE t_sea_snake > -999.0
            AND t_sea_snake < 999.0
            AND wspd_sonic > -999.0
            AND wspd_sonic < 999.0
    '''
df = pd.read_sql_query(query, con)

print('Query returned {0:d} observations.'.format(len(df)))

df = df.dropna()
## Close the database.
con.close()

## Prepare the data.
X = df['lon']
Y = df['lat']
T = df['decimal_day_of_year']
sst = df['t_sea_snake']
wspd = df['wspd_sonic']
fn0 = df['original_file_name']

##
## 1.2. Plot Map
##

lat_foc = np.nanmean(Y)
lon_foc = np.nanmean(X)

## Create the Plotly figure with map background.
fig = go.Figure(layout=dict(autosize=True, height=800))

fig.update_layout(
    mapbox=dict(accesstoken=mapbox_access_token, zoom=1, center={'lon':lon_foc,'lat':lat_foc}),
    mapbox_style='satellite',
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor=color_dark_3,
    legend = dict(bgcolor=color_light_2,
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99)
)

##
## 1.3. Add the data to the map.
##

cruise_track = create_cruise_track_trace(X.values, Y.values, fn0.values)
data_markers = create_data_markers_trace(X, Y, T, sst, 'SST [C]', fn0.values)

for ct in cruise_track:
    fig.add_trace(ct)
fig.add_trace(data_markers)

fig.add_trace(add_ndbc())

##
## 1.4. Final formatting steps
##

## Add map grid lines.
add_grid_lines(fig, dx=10)


############### 2. The layout of the app ###################

##
## 2.1. Define individual sections as Div.
##

banner = html.Div([
        html.H1('AirSeaDB', className='banner-header'),
        html.P('A global, searchable, extendable air-sea flux database. (NOTE: Field campaign and date range selection not yet working.)', className='banner-header'),
    ], id='banner')

query_section = html.Div([
    html.H2('Query Data', className='section-header'),
    html.Label('Campaigns:'),
    dcc.Dropdown(['AEROSE (2006)', 'ATOMIC (2020)', 'CALNEX (2010)', 'DYNAMO (2011)', 'JASMINE (1999)', 'NAURU (1999)', 'PACS (1999, Fall)', 'PACS (2020, Spring)', 'PISTON (2019)', 'PISTON_MISOBOB_2018'], multi=True, value=['DYNAMO (2011)',],style={'backgroundColor':'#ffffff'}),
    html.Label('Color by:'),
    dcc.RadioItems(['SST','Wind Speed'], value='SST', id='color-by-variable'),
    html.Label('Subset by:'),
    html.Br(),
    html.Table([
        html.Tr([
            html.Td('Date: '),
            html.Td(
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=cftime.datetime(1995, 8, 5),
                    max_date_allowed=cftime.datetime(2017, 9, 19),
                    initial_visible_month=cftime.datetime(2017, 8, 5),
                    start_date=cftime.datetime(2017, 8, 15),
                    end_date=cftime.datetime(2017, 8, 25),
                    updatemode='bothdates',
                    className='date-picker',style={'backgroundColor':'#ffffff'}
                )
            )
        ])
    ]),
    html.Table([
        html.Tr([
            html.Td('SST: '),
            html.Td([dcc.Input(id='min-sst-input', value=str(np.nanmin(sst)),type='text', size='8', debounce=True, className='input-field', style={'backgroundColor':'#ffffff'}),
            ' to ',dcc.Input(id='max-sst-input', value=str(np.nanmax(sst)),type='text', size='8', debounce=True, className='input-field', style={'backgroundColor':'#ffffff'}),' Celsius'])
        ]),
        html.Tr([
            html.Td('Wind Speed: '),
            html.Td([dcc.Input(id='min-wspd-input', value=str(np.nanmin(wspd)),type='text', size='8', debounce=True, className='input-field', style={'backgroundColor':'#ffffff'}),
            ' to ',dcc.Input(id='max-wspd-input', value=str(np.nanmax(wspd)),type='text', size='8', debounce=True, className='input-field', style={'backgroundColor':'#ffffff'}), ' m/s'])
        ])
    ]),
], id='query-section')


report_and_download_section = html.Div([
    html.H2('Download Data', className='section-header'),
    html.Label('N = {} observations. '.format(len(df)), id='obs-count'),
    html.Button("Download csv", id="btn-download-txt",style={'backgroundColor':'#ff6633'}),
    dcc.Download(id="download-text"),
], id='report-and-download-section')


map_section = html.Div([
    dcc.Loading(
        id="ls-loading-map",
        children = dcc.Graph(
            id='map-with-data',
            figure=fig,
            config={'displayModeBar':True,}
        ),
        type="circle",
    )
], id='map-section')

##
## 2.2. Put the Divs together into the full web page.
##

app.layout = html.Div(children=[
    banner,
    query_section,
    report_and_download_section,
    map_section,
])


############### 3. Interactive functionality (callbacks) ###################

@app.callback(
    Output(component_id='map-with-data', component_property='figure'),
    Output(component_id='obs-count', component_property='children'),
    Input(component_id='min-sst-input', component_property='value'),
    Input(component_id='max-sst-input', component_property='value'),
    Input(component_id='min-wspd-input', component_property='value'),
    Input(component_id='max-wspd-input', component_property='value'),
    Input(component_id='color-by-variable', component_property='value'),
    prevent_initial_call=True,
)
def update_plot_with_selected_values(min_sst_input_value, max_sst_input_value,
                            min_wspd_input_value, max_wspd_input_value, color_by_variable):

    ## Open database connection.
    con1 = sqlite3.connect(fn)
    ## Query the database.
    query = '''
        SELECT lon,lat,t_sea_snake,wspd_sonic,decimal_day_of_year,original_file_name
            FROM DATA WHERE t_sea_snake > {0}
                AND t_sea_snake < {1}
                AND wspd_sonic > {2}
                AND wspd_sonic < {3}
        '''.format(min_sst_input_value, max_sst_input_value, min_wspd_input_value, max_wspd_input_value)
    df1 = pd.read_sql_query(query, con1)
    print('Query returned {0:d} observations.'.format(len(df1)))

    ## Close the database.
    con1.close()

    X1 = df1['lon']
    Y1 = df1['lat']
    T1 = df1['decimal_day_of_year']
    sst1 = df1['t_sea_snake']
    wspd1 = df1['wspd_sonic']
    fn1 = df1['original_file_name']

    fig.data=[]

    cruise_track = create_cruise_track_trace(X.values, Y.values, fn1.values) # Cruise track will always be full track.
    if color_by_variable == 'SST':
        data_markers = create_data_markers_trace(X1, Y1, T1, sst1, 'SST [C]', fn1.values)
    else:
        data_markers = create_data_markers_trace(X1, Y1, T1, wspd1, 'WSPD [m/s]', fn1.values)

    for ct in cruise_track:
        fig.add_trace(ct)
    fig.add_trace(data_markers)

    fig.add_trace(add_ndbc())
    add_grid_lines(fig, dx=10)

    return [fig, 'N = {} observations. '.format(len(df1))]


@app.callback(
    Output("download-text", "data"),
    Input("btn-download-txt", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, filename="AirSeaDB_selected_data.csv")




############### 4. Initialize the app. ###################
if __name__ == '__main__':
    app.run_server(debug=True)

