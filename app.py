import os
os.environ['MPLCONFIGDIR'] = "/var/www/FLASKAPPS/airseadb/graph"
import matplotlib
matplotlib.use('Agg')

from httplib2 import Response
import numpy as np
import pandas as pd
import datetime as dt
import cftime
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc
import psycopg2
from dash import Dash, html, dcc, Input, Output, ctx
import dash_bootstrap_components as dbc

import sys
#print(sys.version)
#print(sys.prefix)

#from plotting_functions import *  ## Includes color palette.


## Color Palette ############
color_dark_1 = '#1d191e'
color_dark_2 = '#1f2138'
color_dark_3 = '#101125'
color_light_1 = '#a4967a'
color_light_2 = '#bfb39b'
#############################


## Initial App Settings
MIN_DATE = cftime.datetime(1995, 8, 5)
MAX_DATE = cftime.datetime.strptime(dt.datetime.utcnow().strftime('%Y%m%d'),'%Y%m%d')



def get_cbar_range(data, force_range=None):
    min_data_value = np.nanmin(data)
    max_data_value = np.nanmax(data)
    if not (force_range is None):
        mid_data_value = np.nanmedian(data)
        min_data_value = mid_data_value - 0.5*force_range
        max_data_value = mid_data_value + 0.5*force_range
    return (min_data_value, max_data_value)


def backward_diff(x):
    dx = 0.0*x
    dx[1:] = x[1:] - x[:-1]
    dx[0] = dx[1]
    return dx


def create_cruise_track_trace(lon, lat, fn):

    ## Figure out field campaign name.
    campaign_name = fn[0].split('/')[0]

    ## Find Break Points: Where a new cruise (or cruise leg) has clearly started.
    dlon = backward_diff(lon)
    dlat = backward_diff(lat)
    dlatlon = np.sqrt(np.power(dlon,2) + np.power(dlat,2))
    break_point_indices = [0] + [x for x in range(len(dlatlon)) if dlatlon[x] > 1.0] + [len(dlatlon)]

    tracks = []
    for ii in range(len(break_point_indices)-1):
        tracks += [go.Scattermapbox(lon=lon[break_point_indices[ii]:break_point_indices[ii+1]],
                        lat=lat[break_point_indices[ii]:break_point_indices[ii+1]],
                        mode='lines', line={'width':2.0, 'color':'white'}, name=campaign_name,
                        showlegend=False, hoverinfo='skip')]

    return tracks


def create_data_markers_trace(lon, lat, T, Z, label, fn, skip=10):

    ## Figure out field campaign name.
    campaign_name = fn[0].split('/')[0]
    campaign_names = [x.split('/')[0] for x in fn]

    cmin,cmax = get_cbar_range(Z)

    if 'SST' in label:
        cmap = 'jet'
    elif 'WSPD' in label:
        cmap = cc.CET_L20
    else:
        cmap = cc.CET_R1

    return go.Scattermapbox(lon=lon[::skip], lat=lat[::skip],
                    marker=dict(color=Z[::skip], cmin=cmin, cmax=cmax, colorscale=cmap,
                        colorbar=dict(len=0.6, title=label, y=0.99, yanchor='top', x=0.99, xanchor='right',bgcolor=color_light_2)),
                    hovertemplate = '%{text}<br>lon: %{lon:.2f}<br>lat: %{lat:.2f}<br>'+label+': %{marker.color:.2f}<extra></extra>',
                    text= campaign_names[::skip],
                    showlegend=False)


def add_ndbc():

    fn = '/var/www/FLASKAPPS/airseadb/data/activestations.xml'
    df = pd.read_xml(fn)
    # df = df[df['met'] != 'n'] # Tried to subset by met, but no success.

    marker_sizes =  7+0.0*df['lon'].values
    marker_symbols =  ["star" for x in range(len(df['lon']))]
    return go.Scattermapbox(lon=df['lon'].values, lat=df['lat'].values, text=df['name'].values,
                            marker = {'size': marker_sizes, 'symbol': marker_symbols, 'allowoverlap':True},
                            hovertemplate = '%{text}<extra></extra>')


def add_grid_lines(fig, dx=5, width=0.3, color='grey'):
    for y in np.arange(-80,80+dx,dx):
        fig.add_trace(go.Scattermapbox(lon=np.array([0,360]),lat=np.array([y,y]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=False, legendgroup='llgrid',hoverinfo='skip'))
    for y in [0.0,]:
        fig.add_trace(go.Scattermapbox(lon=np.array([0,360]),lat=np.array([y,y]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=False, legendgroup='llgrid',hoverinfo='skip'))
    for x in np.arange(0,360+dx,dx):
        fig.add_trace(go.Scattermapbox(lon=np.array([x,x]),lat=np.array([-90,90]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=False, legendgroup='llgrid',hoverinfo='skip'))


## In order to run as WSGI under Apache,
## the requests_pathname_prefix needs to be set.
## Otherwise, the application will only display "loading" and hang!
## For the interactive version for development purposes
##    (__name__ is '__main__') run on command line, use '/'.
## For deployment, use the directory under http://orca, e.g., '/airseadb/'.

if __name__ == '__main__':
    ## Use this for development.
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        requests_pathname_prefix='/') 
else:
    ## Use this for deployment on orca.
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        requests_pathname_prefix='/airseadb/') 


## Generate the application for WSGI.
## "server" gets imported as "application" for WSGI, in app.wsgi.
server = app.server

# data_dir = '/var/www/FLASKAPPS/airseadb/data'
# html_out_dir = '/home/orca/bkerns/public_html/projects/noaa_air_sea_flux/report_feb_2022/interactive_plots'

cruise_list = ['dynamo_2011',]

mapbox_access_token = 'pk.eyJ1IjoiYnJhbmRvbndrZXJucyIsImEiOiJja3ZyOGNlZmcydTdrMm5xZ3d4ZWNlZXZpIn0.OkA0r6XFSY-Dx0bk7UPPZQ'

############### 1. The map and data visualization ###################

##
## 1.1. Get the data (or subset of data) from AirSeaDB
##

## Connect to SQLite database
# fn = (data_dir + '/AirSeaDB.sqlite')
# con = sqlite3.connect(fn)

## Query the database.
def query_data(db_host, db_name, user, password, query_text, verbose=False):
    """
    Query Postgres table and return results in a DataFrame.
    """

    conn_str = "host={0:s} dbname={1:s} user={2:s} password={3:s}".format(db_host, db_name, user, password)
    with psycopg2.connect(conn_str) as conn:
        cur = conn.cursor()
        cur.execute(query_text)

        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        if verbose:
            print('Query returned {0:d} observations.'.format(len(df)))

        df = df.dropna()

    return df

## Initially, load all the points into memory.
query = '''
    SELECT datetime,lon,lat,t_sea_snake,wspd_psd,decimal_day_of_year,original_file_names
        FROM ship_data WHERE t_sea_snake > -999.0
            AND t_sea_snake < 999.0
            AND wspd_psd > -999.0
            AND wspd_psd < 999.0
    '''
df = query_data('localhost', 'airseadb', 'bkerns', 'huracan5', query, verbose=True)

## Prepare the data.
X = df['lon']
Y = df['lat']
T = df['decimal_day_of_year']
sst = df['t_sea_snake']
wspd = df['wspd_psd']
fn0 = df['original_file_names']

min_date0 = df['datetime'].min()
max_date0 = df['datetime'].max()
print(min_date0, max_date0)

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
## 1.3.5. Add data to scatter plot.
##

fig_scatter = go.Figure(layout=dict(autosize=True, height=400))
scatter_trace = go.Scattergl(x=sst, y=wspd, mode='markers')  ## Scatter with WebGL (Scattergl) for better performance with large data.
fig_scatter.add_trace(scatter_trace)
fig_scatter.update_xaxes(showline = True, linecolor = 'black', linewidth = 1, mirror = True)
fig_scatter.update_yaxes(showline = True, linecolor = 'black', linewidth = 1, mirror = True)
fig_scatter.update_layout(height=600, width=600,
    xaxis_title = 'SST [C]', yaxis_title='Wind Speed [m/s]',
    title_text="Selected AirSeaDB Data")
fig_scatter.update_xaxes(showgrid=True, gridwidth=1, griddash='dot', gridcolor='darkgrey')
fig_scatter.update_yaxes(showgrid=True, gridwidth=1, griddash='dot', gridcolor='darkgrey')

##
## 1.4. Final formatting steps
##

## Add map grid lines.
add_grid_lines(fig, dx=10)


############### 2. The layout of the app ###################

##
## 2.1. Define individual sections as Div.
##

## List of field campaigns and year(years).
query = 'SELECT DISTINCT program, years FROM ship_data'
df_programs = query_data('localhost', 'airseadb', 'bkerns', 'huracan5', query, verbose=False)


def programs_df_to_list(df):
    return ['{0:s} ({1:s})'.format(df['program'][x], df['years'][x]) for x in range(len(df))]

dropdown_programs_list = programs_df_to_list(df_programs)

banner = html.Div([
        html.H1('AirSeaDB', className='banner-header'),
        html.P('A global, searchable, extendable air-sea flux database.', className='banner-header'),
    ], id='banner')

query_section = html.Div([
    html.H2('Query Data', className='section-header'),
    html.Label('Placeholder', id = 'campaigns-select-label'),
    ## Dropdown initially has all the field campaigns/programs selected.
    dcc.Dropdown(dropdown_programs_list, multi=True, value=dropdown_programs_list,
        placeholder = 'Select One Or More.',
        style={'backgroundColor':'#ffffff'}, id = 'selected-programs',
        optionHeight=30, maxHeight=150),
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
                    min_date_allowed=min_date0, #MIN_DATE, #cftime.datetime(1995, 8, 5),
                    max_date_allowed=max_date0, #MAX_DATE, #cftime.datetime(2017, 9, 19),
                    initial_visible_month=min_date0, #cftime.datetime(2006, 6, 1),
                    start_date=min_date0, #MIN_DATE, #cftime.datetime(2017, 8, 15),
                    end_date=max_date0, #MAX_DATE, #cftime.datetime(2017, 8, 25),
                    updatemode='bothdates',
                    with_portal=True,
                    number_of_months_shown=3,
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





report_and_download_section = html.Div([
    html.H2('Download Data', className='section-header'),
    # html.Label('N = {} observations. '.format(len(df)), id='obs-count'),
    html.Label('Placeholder', id='obs-count'),
    html.Button("Download csv", id="btn-download-txt",style={'backgroundColor':'#ff6633'}),
    dcc.Download(id="download-text"),
], id='report-and-download-section')


plotting_section = html.Div([
    dcc.Loading(
        id="ls-loading-scatter",
        children = dcc.Graph(
            id='scatter-plot-with-data',
            figure=fig_scatter,
            config={'displayModeBar':True}
        ),
        type="circle",
    )
], id='plotting-section')


##
## 2.2. Put the Divs together into the full web page.
##

app.layout = dbc.Container(children=[
    dbc.Row(dbc.Col(banner, md=12)),
    dbc.Row([dbc.Col(query_section, md=2), dbc.Col(map_section, md=10)]),
    dbc.Row([dbc.Col(report_and_download_section, md=2), dbc.Col(plotting_section, md=6)]),
], fluid=True)


############### 3. Interactive functionality (callbacks) ###################

@app.callback(
    Output(component_id='map-with-data', component_property='figure'),
    Output(component_id='scatter-plot-with-data', component_property='figure'),
    Output(component_id='obs-count', component_property='children'),
    Output(component_id='campaigns-select-label', component_property='children'),
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='selected-programs', component_property='value'),
    Input(component_id='selected-programs', component_property='options'),
    Input(component_id='min-sst-input', component_property='value'),
    Input(component_id='max-sst-input', component_property='value'),
    Input(component_id='min-wspd-input', component_property='value'),
    Input(component_id='max-wspd-input', component_property='value'),
    Input(component_id='color-by-variable', component_property='value'),
    prevent_initial_call=False, #True,
)
def update_plot_with_selected_values(start_date, end_date, selected_programs, selected_program_options, min_sst_input_value, max_sst_input_value,
                            min_wspd_input_value, max_wspd_input_value, color_by_variable):

    if len(selected_programs) == 0:
        selected_programs_in = '(\'NULL\')'
    else:
        selected_programs_in = '('
        for n, x in enumerate(selected_programs): 
            if n < len(selected_programs)-1:
                selected_programs_in += ('\''+x+'\',')
            else:
                selected_programs_in += ('\''+x+'\'')

        selected_programs_in += ')'
    print(selected_programs_in)

    ## Query the database.
    query = '''
        SELECT datetime,lon,lat,t_sea_snake,wspd_psd,decimal_day_of_year,original_file_names,concat("program", ' (', "years", ')') AS "program_year"
            FROM ship_data WHERE t_sea_snake > {0}
                AND t_sea_snake < {1}
                AND wspd_psd > {2}
                AND wspd_psd < {3}
                AND datetime BETWEEN '{4}' AND '{5}'
                AND concat("program", ' (', "years", ')') IN {6}
        '''.format(min_sst_input_value, max_sst_input_value, min_wspd_input_value, max_wspd_input_value, start_date, end_date,selected_programs_in)
    print(query)
    df1 = query_data('localhost','airseadb','bkerns','huracan5',query,verbose=True)

    X1 = df1['lon']
    Y1 = df1['lat']
    dT1 = df1['datetime']
    T1 = df1['decimal_day_of_year']
    sst1 = df1['t_sea_snake']
    wspd1 = df1['wspd_psd']
    fn1 = df1['original_file_names']

    fig.data=[]

    if len(df1) > 0:

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

    fig_scatter.update_traces(x=sst1, y=wspd1)

    ## Campaigns label string.
    n_total_campaigns = len(selected_program_options)
    campaigns_label_str = 'Campaigns ({0} of {1} selected)'.format(len(selected_programs), n_total_campaigns)

    ## The 3 outputs correspond with the 3 Output entries in the callback above.
    return [fig, fig_scatter, 'N = {} observations. '.format(len(df1)), campaigns_label_str]


@app.callback(
    Output("download-text", "data"),
    Input(component_id='date-picker-range', component_property='start_date'),
    Input(component_id='date-picker-range', component_property='end_date'),
    Input(component_id='min-sst-input', component_property='value'),
    Input(component_id='max-sst-input', component_property='value'),
    Input(component_id='min-wspd-input', component_property='value'),
    Input(component_id='max-wspd-input', component_property='value'),
    Input(component_id='color-by-variable', component_property='value'),
    Input("btn-download-txt", "n_clicks"),
    prevent_initial_call=True,
)
def func(start_date, end_date, min_sst_input_value, max_sst_input_value,
        min_wspd_input_value, max_wspd_input_value, color_by_variable,n_clicks):

    if ctx.triggered_id == 'btn-download-txt':

        ## Query the database.
        query = '''
            SELECT * FROM ship_data WHERE t_sea_snake > {0}
                    AND t_sea_snake < {1}
                    AND wspd_psd > {2}
                    AND wspd_psd < {3}
                    AND datetime BETWEEN '{4}' AND '{5}'
            '''.format(min_sst_input_value, max_sst_input_value, min_wspd_input_value, max_wspd_input_value, start_date, end_date)
        df1 = query_data('localhost','airseadb','bkerns','huracan5',query,verbose=True) #   pd.read_sql_query(query, con1)

        ## The 1 output corresponds with the 1 Output entry in the callback above.
        return dcc.send_data_frame(df1.to_csv, filename="AirSeaDB_selected_data.csv")
    else:
        return None




############### 4. Initialize the app. ###################
if __name__ == '__main__':
    app.run_server(debug=True)

