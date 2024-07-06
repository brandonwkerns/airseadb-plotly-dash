import os
# os.environ['MPLCONFIGDIR'] = "/var/www/FLASKAPPS/airseadb/graph"
import matplotlib
# matplotlib.use('Agg')

from httplib2 import Response
import numpy as np
import pandas as pd
import datetime as dt
import cftime
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc
import psycopg2
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc

import sys

import datashader as ds
from datashader.utils import lnglat_to_meters
import datashader.transfer_functions as tf



## Color Palette ############
color_dark_1 = '#1d191e'
color_dark_2 = '#1f2138'
color_dark_3 = '#101125'
color_light_1 = '#a4967a'
color_light_2 = '#bfb39b'
#############################


## Database Host
# HOST='localhost'  # For invoking on the command line with "python app.py"
HOST='database'  # For using Docker with the option --add-host=database:<host-ip>

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


def create_cruise_track_trace(lon, lat, fn, skip=1, color='darkgrey'):

    coordinates = [[-180, -80],[180,-80],[180,80],[-180,80]]
    x0, y0 = lnglat_to_meters(coordinates[0][0], coordinates[0][1]) # To web mercartor.
    x1, y1 = lnglat_to_meters(coordinates[2][0], coordinates[2][1]) # To web mercartor.
    cvs = ds.Canvas(plot_width=1800, plot_height=800, x_range=[x0,x1], y_range=[y0,y1])

    dff = pd.DataFrame(dict(Lon=lon, Lat=lat))
    dff.loc[:, 'Easting'], dff.loc[:, 'Northing'] = lnglat_to_meters(dff.Lon,dff.Lat) # To web mercartor.
    agg = cvs.points(dff, x='Easting', y='Northing', agg=ds.any())

    img = tf.shade(tf.spread(agg,1), cmap=[color,])[::-1].to_pil()

    # return tracks
    return img, coordinates


def create_cruise_track_labels(lon, lat, labels):

    # skip = 1000
    labels_list = np.unique(labels)
    X = []
    Y = []
    L = []
    for this_label in labels_list:
        idx_this_label = sorted(np.argwhere(labels == this_label))
        # print((this_label, idx_this_label[0][::skip]))
        X += [lon[idx_this_label[0][0]]]
        Y += [lat[idx_this_label[0][0]]]
        L += [labels[idx_this_label[0][0]]]
        X += [lon[idx_this_label[-1][0]]]
        Y += [lat[idx_this_label[-1][0]]]
        L += [labels[idx_this_label[-1][0]]]


    ## Find Break Points: Where a new cruise (or cruise leg) has clearly started.
    dlon = backward_diff(lon)
    dlat = backward_diff(lat)
    dlatlon = np.sqrt(np.power(dlon,2) + np.power(dlat,2))
    break_point_indices = [0] + [x for x in range(len(dlatlon)) if dlatlon[x] > 1.0] + [len(dlatlon)-1]

    # tracks = []
    for ii in break_point_indices:
        X += [lon[ii]]
        Y += [lat[ii]]
        L += [labels[ii]]


    trace = go.Scattermapbox(lon=X, lat=Y, text=L, mode='markers+text',
        textposition='top right',
        textfont=dict(color='white'),
        showlegend=False)

    # return tracks
    return trace


def create_data_markers_trace(lon, lat, T, Z, label, fn, skip=1):

    cmin,cmax = get_cbar_range(Z)

    if 'SST' in label:
        cmap = cc.rainbow #'jet'
    elif 'WSPD' in label:
        cmap = cc.CET_L20
    else:
        cmap = cc.CET_R1

    coordinates = [[-180, -80],[180,-80],[180,80],[-180,80]]
    x0, y0 = lnglat_to_meters(coordinates[0][0], coordinates[0][1]) # To web mercartor.
    x1, y1 = lnglat_to_meters(coordinates[2][0], coordinates[2][1]) # To web mercartor.
    cvs = ds.Canvas(plot_width=1800, plot_height=800, x_range=[x0,x1], y_range=[y0,y1])

    dff = pd.DataFrame(dict(Lon=lon, Lat=lat, Data=Z))
    dff.loc[:, 'Easting'], dff.loc[:, 'Northing'] = lnglat_to_meters(dff.Lon,dff.Lat) # To web mercartor.
    agg = cvs.points(dff, x='Easting', y='Northing', agg=ds.mean('Data'))
    img = tf.shade(tf.spread(agg,3), cmap=cmap)[::-1].to_pil()

    # return tracks
    return img, coordinates


def add_ndbc():

    fn = 'data/activestations.xml'
    df = pd.read_xml(fn)
    # df = df[df['met'] != 'n'] # Tried to subset by met, but no success.

    marker_sizes =  7+0.0*df['lon'].values
    marker_symbols =  ["star" for x in range(len(df['lon']))]
    return go.Scattermapbox(lon=df['lon'].values, lat=df['lat'].values, text=df['name'].values,
                            marker = {'size': marker_sizes, 'symbol': marker_symbols, 'allowoverlap':True},
                            uid='ndbc_markers', hovertemplate = '%{text}<extra></extra>', showlegend=False)


def add_grid_lines(fig, dx=5, width=0.3, color='grey'):
    for y in np.arange(-80,80+dx,dx):
        fig.add_trace(go.Scattermapbox(lon=np.array([0,360]),lat=np.array([y,y]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=False, legendgroup='llgrid',hoverinfo='skip', uid='meridians')
                        )
    for y in [0.0,]:
        fig.add_trace(go.Scattermapbox(lon=np.array([0,360]),lat=np.array([y,y]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=False, legendgroup='llgrid',hoverinfo='skip',uid='equator')
                        )
    for x in np.arange(0,360+dx,dx):
        fig.add_trace(go.Scattermapbox(lon=np.array([x,x]),lat=np.array([-90,90]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=False, legendgroup='llgrid',hoverinfo='skip',uid='parallels')
                        )


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
mapbox_access_token = 'pk.eyJ1IjoiYnJhbmRvbndrZXJucyIsImEiOiJja3ZyOGNlZmcydTdrMm5xZ3d4ZWNlZXZpIn0.OkA0r6XFSY-Dx0bk7UPPZQ'

############### 1. The map and data visualization ###################

##
## 1.1. Get the data (or subset of data) from AirSeaDB
##

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

        # df = df.dropna()

    return df

## Initially, load all the points into memory.
query = '''
    SELECT datetime,lon,lat,t_sea_snake,wspd_psd,decimal_day_of_year,concat("program", ' (', "years", ')') AS "program_year"
        FROM ship_data WHERE t_sea_snake > -999.0
            AND t_sea_snake < 999.0
            AND wspd_psd > -999.0
            AND wspd_psd < 999.0
    '''
df = query_data(HOST, 'airseadb', 'bkerns', 'huracan5', query, verbose=True)

## Prepare the data.
X = df['lon']
Y = df['lat']
T = df['decimal_day_of_year']
sst = df['t_sea_snake']
wspd = df['wspd_psd']
program_year = df['program_year']

min_date0 = df['datetime'].min()
max_date0 = df['datetime'].max()
# print(min_date0, max_date0)

## Saildrone
## Initially, load all the points into memory.
query_sd = '''
    SELECT datetime,lon,lat,sst_sbe,sst_ctd,wspd,id
        FROM saildrone_data
    '''
df_sd = query_data(HOST, 'airseadb', 'bkerns', 'huracan5', query_sd, verbose=True)

print(len(df_sd))
## Prepare the data.
X_sd = df_sd['lon']
Y_sd = df_sd['lat']
# T_sd = df_sd['Datetime']
sst_sd = df_sd['sst_sbe']
wspd_sd = df_sd['wspd']
id_sd = df_sd['id']

min_date0_sd = df_sd['datetime'].min()
max_date0_sd = df_sd['datetime'].max()

print(np.nanmin(X_sd), np.nanmax(X_sd))
print(min_date0_sd, max_date0_sd)



## ARM
## Initially, load all the points into memory.
query_sd = '''
    SELECT datetime,lon,lat,t_sea,wspd_AMF2,concat("program", ' (', "years", ')') AS "program_year"
        FROM arm_data
    '''
df_arm = query_data('localhost', 'airseadb', 'bkerns', 'huracan5', query_sd, verbose=True)

print(len(df_sd))
## Prepare the data.
X_arm = df_arm['lon'] - 360.0
Y_arm = df_arm['lat']
# T_sd = df_sd['Datetime']
sst_arm = df_arm['t_sea']
wspd_arm = 0.0*df_arm['t_sea'] #df_arm['wspd_AMF2']
# id_arm = df_sd['id']
program_year_arm = df_arm['program_year']

min_date0_arm = df_arm['datetime'].min()
max_date0_arm = df_arm['datetime'].max()

print(np.nanmin(X_arm), np.nanmax(X_arm))
print(min_date0_arm, max_date0_arm)


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
    font_color = 'black',
    legend = dict(bgcolor=color_light_2,
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99)
)

##
## 1.3. Add the data to the map.
##
## Only include the "background" cruise track traces here?
## This is not needed. Data is added to the map when the update function is called upon initial app load.
##

## Skip ship tracks to improve loading time.
## Plotting every 100 seems to be OK for zoom level 1. Every 1000 eliminates most of the detail.
## TODO: Check if skip value is OK for finer zooms.
# cruise_track = create_cruise_track_trace(X.values, Y.values, program_year.values, skip=100) # Cruise track will always be full track.
# for ct in cruise_track:
#     fig.add_trace(ct)
cruise_track_img, cruise_track_coords = create_cruise_track_trace(X.values, Y.values, program_year.values) # Cruise track will always be full track.
cruise_track_layer_dict = {
                    "sourcetype": "image",
                    "source": cruise_track_img,
                    "coordinates": cruise_track_coords}
## The cruise layer dictionary is used in the callback function below.
fig.add_trace(create_cruise_track_labels(X.values, Y.values, program_year.values))

print(X_sd.values)
print(Y_sd.values)
print(np.unique(id_sd.values))
cruise_track_img_sd, cruise_track_coords_sd = create_cruise_track_trace(X_sd.values, Y_sd.values, id_sd.values, color='cyan') # Cruise track will always be full track.
cruise_track_layer_dict_sd = {
                    "sourcetype": "image",
                    "source": cruise_track_img_sd,
                    "coordinates": cruise_track_coords_sd}
## The cruise layer dictionary is used in the callback function below.
# fig.add_trace(create_cruise_track_labels(X.values, Y.values, program_year.values))


cruise_track_img_arm, cruise_track_coords_arm = create_cruise_track_trace(X_arm.values, Y_arm.values, program_year_arm.values) # Cruise track will always be full track.
cruise_track_layer_dict = {
                    "sourcetype": "image",
                    "source": cruise_track_img_arm,
                    "coordinates": cruise_track_coords}
## The cruise layer dictionary is used in the callback function below.
fig.add_trace(create_cruise_track_labels(X_arm.values, Y_arm.values, program_year_arm.values))

print(X_arm.values)
print(Y_arm.values)
print(np.unique(program_year_arm.values))
cruise_track_img_arm, cruise_track_coords_arm = create_cruise_track_trace(X_arm.values, Y_arm.values, program_year_arm.values, color='cyan') # Cruise track will always be full track.
cruise_track_layer_dict_arm = {
                    "sourcetype": "image",
                    "source": cruise_track_img_arm,
                    "coordinates": cruise_track_coords_arm}
## The cruise layer dictionary is used in the callback function below.
# fig.add_trace(create_cruise_track_labels(X.values, Y.values, program_year.values))


## Add NDBC buoy markers.
fig.add_trace(add_ndbc())

##
## 1.3.5. Add data to scatter plot.
##

fig_scatter = go.Figure(layout=dict(autosize=True, height=400))
print('--- Start Scatter Trace: '+dt.datetime.now().strftime('%H:%M:%S'), flush=True)
scatter_trace = go.Scattergl(x=sst[::1], y=wspd[::1], mode='markers')  ## Scatter with WebGL (Scattergl) for better performance with large data.
fig_scatter.add_trace(scatter_trace)
print('--- End Scatter Trace: '+dt.datetime.now().strftime('%H:%M:%S'), flush=True)
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
query_arm = 'SELECT DISTINCT program, years FROM arm_data'
df_programs = query_data('localhost', 'airseadb', 'bkerns', 'huracan5', query, verbose=False)
df_programs_arm = query_data('localhost', 'airseadb', 'bkerns', 'huracan5', query_arm, verbose=False)

def programs_df_to_list(df):
    return ['{0:s} ({1:s})'.format(df['program'][x], df['years'][x]) for x in range(len(df))]

dropdown_programs_list = ['Buoys']
dropdown_programs_list += programs_df_to_list(df_programs)
dropdown_programs_list += programs_df_to_list(df_programs_arm)


## List of field campaigns and year(years).
query = 'SELECT DISTINCT id FROM saildrone_data'
df_saildrone_deployments = query_data(HOST, 'airseadb', 'bkerns', 'huracan5', query, verbose=False)

print(df_saildrone_deployments['id'].values.tolist())
dropdown_programs_list += df_saildrone_deployments['id'].values.tolist()



banner = html.Div([
        html.H1('AirSeaDB', className='banner-header'),
        html.P('A global, searchable, extendable air-sea flux database.', className='banner-header'),
    ], id='banner')

query_section = html.Div([
    html.H2('Select Data', className='section-header'),
    html.Label('Campaigns', id = 'campaigns-select-label'),
    ## Dropdown initially has all the field campaigns/programs selected.
    html.Div(
        dbc.Button(
                "Expand List",
                id="collapse-button-programs",
                className="mb-3",
                color="primary",
                n_clicks=0,
            ),
        className="d-grid gap-2"),
    dbc.Collapse(dcc.Dropdown(dropdown_programs_list, multi=True, value=dropdown_programs_list,
        placeholder = 'Select One Or More.',
        style={'backgroundColor':'#ffffff'}, id = 'selected-programs',
        optionHeight=30, maxHeight=150), id="collapse-programs", is_open=False,),
    html.Br(),
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

## Button callbacks for the Programs and Field Campaigns list.
@app.callback(
    Output("collapse-programs", "is_open"),
    Output("collapse-button-programs", "children"),
    [Input("collapse-button-programs", "n_clicks")],
    [State("collapse-programs", "is_open")],
)
def toggle_collapse(n, is_open):
    ## List starts collapsed, n = 0.
    ## For odd n, it will be expanded.
    ## For even n, it will be collapsed. 
    if n % 2 == 1:
        return (True, 'Collapse List')
    else:
        return (False, 'Expand List')


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
    # print(selected_programs_in)

    ##
    ## Query the database.
    ##

    ## Ship Data
    query = '''
        SELECT datetime,lon,lat,t_sea_snake,wspd_psd,decimal_day_of_year,concat("program", ' (', "years", ')') AS "program_year"
            FROM ship_data WHERE t_sea_snake > {0}
                AND t_sea_snake < {1}
                AND wspd_psd > {2}
                AND wspd_psd < {3}
                AND datetime BETWEEN '{4}' AND '{5}'
                AND concat("program", ' (', "years", ')') IN {6}
        '''.format(min_sst_input_value, max_sst_input_value, min_wspd_input_value, max_wspd_input_value, start_date, end_date,selected_programs_in)
    # print(query)
    df1 = query_data(HOST,'airseadb','bkerns','huracan5',query,verbose=True)

    X1 = df1['lon']
    Y1 = df1['lat']
    dT1 = df1['datetime']
    T1 = df1['decimal_day_of_year']
    sst1 = df1['t_sea_snake']
    wspd1 = df1['wspd_psd']
    program_year = df1['program_year']

    ## Reset the traces with uid = 'ship_data'
    idx_ship_data = [x for x in range(len(fig.data)) if fig.data[x]['uid'] == 'ship_data']
    # print(idx_ship_data)

    if len(idx_ship_data) > 0:
        #fig.data.pop(idx_ship_data[0])
        fig['data'] = tuple(fig['data'][x] for x in range(len(fig['data'])) if not x in idx_ship_data)

    if len(df1) > 0:

        if color_by_variable == 'SST':
            Z = sst1
            label = 'SST [C]'
        else:
            Z = wspd1
            label = 'WSPD [m/s]'

        cruise_track_data_img, cruise_track_coords = create_data_markers_trace(X1, Y1, T1, Z, label, program_year.values)
        cruise_track_data_layer_dict = {
                            "sourcetype": "image",
                            "source": cruise_track_data_img,
                            "coordinates": cruise_track_coords}
        fig.update_layout(mapbox_layers = [cruise_track_layer_dict, cruise_track_layer_dict_sd, cruise_track_data_layer_dict])

        # Dummy trace for color bar.
        cmin,cmax = get_cbar_range(Z)

        if 'SST' in label:
            cmap = 'jet'
        elif 'WSPD' in label:
            cmap = cc.CET_L20
        else:
            cmap = cc.CET_R1

        dummy_trace=go.Scatter(x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    colorscale=cmap, cmin=cmin, cmax=cmax, 
                    colorbar=dict(len=0.6, title=label, y=0.99, yanchor='top', x=0.99,
                    xanchor='right',bgcolor=color_light_2),
                    showscale=True),
                hoverinfo='none',
                showlegend=False)

        fig.add_trace(dummy_trace)

    else:
        fig.update_layout(mapbox_layers = [cruise_track_layer_dict, cruise_track_layer_dict_sd])





    ## ARM Ship Data (Magic)
    query = '''
        SELECT datetime,lon,lat,t_sea,wspd_amf2,decimal_day_of_year,concat("program", ' (', "years", ')') AS "program_year"
            FROM arm_data WHERE t_sea > {0}
                AND t_sea < {1}
                AND wspd_amf2 > {2}
                AND wspd_amf2 < {3}
                AND datetime BETWEEN '{4}' AND '{5}'
                AND concat("program", ' (', "years", ')') IN {6}
        '''.format(min_sst_input_value, max_sst_input_value, min_wspd_input_value, max_wspd_input_value, start_date, end_date,selected_programs_in)
    # print(query)
    df2 = query_data('localhost','airseadb','bkerns','huracan5',query,verbose=True)

    X2 = df2['lon'] - 360.0
    Y2 = df2['lat']
    dT2 = df2['datetime']
    T2 = df2['decimal_day_of_year']
    sst2 = df2['t_sea']
    wspd2 = df2['wspd_amf2']
    program_year = df2['program_year']

    ## Reset the traces with uid = 'ship_data'
    # idx_ship_data = [x for x in range(len(fig.data)) if fig.data[x]['uid'] == 'ship_data']

    # if len(idx_ship_data) > 0:
    #     fig['data'] = tuple(fig['data'][x] for x in range(len(fig['data'])) if not x in idx_ship_data)

    if len(df2) > 0 and len(df2) > 0:

        if color_by_variable == 'SST':
            Z = sst2
            label = 'SST [C]'
        else:
            Z = wspd2
            label = 'WSPD [m/s]'

        print(X2)
        print(Z)
        cruise_track_data_img_arm, cruise_track_coords_arm = create_data_markers_trace(X2, Y2, T2, Z, label, program_year.values)
        cruise_track_data_layer_dict_arm = {
                            "sourcetype": "image",
                            "source": cruise_track_data_img_arm,
                            "coordinates": cruise_track_coords_arm}
        fig.update_layout(mapbox_layers = [cruise_track_layer_dict, cruise_track_layer_dict_sd, cruise_track_data_layer_dict, cruise_track_data_layer_dict_arm])

        # Dummy trace for color bar.
        cmin,cmax = get_cbar_range(Z)

        if 'SST' in label:
            cmap = 'jet'
        elif 'WSPD' in label:
            cmap = cc.CET_L20
        else:
            cmap = cc.CET_R1

        dummy_trace2=go.Scatter(x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    colorscale=cmap, cmin=cmin, cmax=cmax, 
                    colorbar=dict(len=0.6, title=label, y=0.99, yanchor='top', x=0.99,
                    xanchor='right',bgcolor=color_light_2),
                    showscale=True),
                hoverinfo='none',
                showlegend=False)

        fig.add_trace(dummy_trace2)

    else:
        fig.update_layout(mapbox_layers = [cruise_track_layer_dict, cruise_track_layer_dict_sd, cruise_track_layer_dict_arm])












    ## Update Scatter Plot
    fig_scatter.update_traces(x=sst1, y=wspd1)

    ## Campaigns label string.
    n_total_campaigns = len(selected_program_options)
    campaigns_label_str = 'Campaigns ({0} of {1})'.format(len(selected_programs), n_total_campaigns)

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
        df1 = query_data(HOST,'airseadb','bkerns','huracan5',query,verbose=True) #   pd.read_sql_query(query, con1)

        ## The 1 output corresponds with the 1 Output entry in the callback above.
        return dcc.send_data_frame(df1.to_csv, filename="AirSeaDB_selected_data.csv")
    else:
        return None




############### 4. Initialize the app. ###################
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=9000, use_reloader=False)

