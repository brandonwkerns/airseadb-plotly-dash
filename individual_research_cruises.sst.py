import numpy as np
import pandas as pd
import cftime
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc
import sqlite3
import os

data_dir = '/home/orca/bkerns/projects/noaa_air_sea_flux/noaa_psd_ship_obs/data/processed'
html_out_dir = '/home/orca/bkerns/public_html/projects/noaa_air_sea_flux/report_feb_2022/interactive_plots'

cruise_list = ['dynamo_2011',]

mapbox_access_token = 'pk.eyJ1IjoiYnJhbmRvbndrZXJucyIsImEiOiJja3ZyOGNlZmcydTdrMm5xZ3d4ZWNlZXZpIn0.OkA0r6XFSY-Dx0bk7UPPZQ'

for this_cruise in cruise_list:
    print('Doing '+this_cruise)

    ##
    ## 1. Get the data (or subset of data) from AirSeaDB
    ##

    ## Connect to SQLite database
    fn = (data_dir + '/'+this_cruise+'.sqlite')
    con = sqlite3.connect(fn)
    # cur = con.cursor()

    ## Query the database.
    df = pd.read_sql_query('SELECT lon,lat,t_sea_snake,decimal_day_of_year FROM DATA', con)
    # df = pd.read_sql_query('SELECT lon,lat,t_sea_snake,decimal_day_of_year FROM DATA WHERE t_sea_snake > 29.0', con)
    print('Query returned {0:d} observations.'.format(len(df)))

    ## Close the database.
    con.close()

    ## Prepare the data.
    X = df['lon']
    Y = df['lat']
    T = df['decimal_day_of_year']
    sst = df['t_sea_snake']

    ##
    ## 2. Plot Map
    ##

    lat_foc = np.nanmean(Y)
    lon_foc = np.nanmean(X)

    ## Create the Plotly figure with map background.
    fig = go.Figure(layout=dict(width=800, height=500, title = this_cruise.upper()+' (Colored by SST)')) #, zoom=5, center={'lon':lon_foc,'lat':lat_foc}))

    fig.update_layout(
        mapbox=dict(accesstoken=mapbox_access_token, zoom=3, center={'lon':lon_foc,'lat':lat_foc}),
        mapbox_style='streets', 
    )

    ##
    ## 3. Add the data to the map.
    ##
    fig.add_trace(go.Scattermapbox(lon=X,lat=Y,
                    mode='lines', line={'width':2.0, 'color':'black'}, name='Cruise Track',
                    showlegend=True, hoverinfo='skip')
                )


    def get_cbar_range(data, force_range=None):
        min_data_value = np.nanmin(data)
        max_data_value = np.nanmax(data)
        if not (force_range is None):
            mid_data_value = np.nanmedian(data)
            min_data_value = mid_data_value - 0.5*force_range
            max_data_value = mid_data_value + 0.5*force_range
        return (min_data_value, max_data_value)


    min_cbar_range = 2.0
    cmin,cmax = get_cbar_range(sst)
    # cmin,cmax = get_cbar_range(sst, force_range=min_cbar_range)

    fig.add_trace(go.Scattermapbox(lon=X, lat=Y,
                    marker=dict(color=sst, cmin=cmin, cmax=cmax, colorscale=cc.CET_R1,
                        colorbar=dict(len=0.6, title='SST [C]', y=0.1, yanchor='bottom')),
                    hovertemplate = '%{text}<br>lon: %{lon:.2f}<br>lat: %{lat:.2f}<br>SST: %{marker.color:.2f}',
                    text=T,
                    name = 'Cruise Data', showlegend=True)
                )

    # fig.add_trace(go.Scattermapbox(lon=X2[0:1],lat=Y2[0:1],
    #                 mode='markers+text',
    #                 marker=dict(size=10, symbol='star'),
    #                 text=['Begin: '+T2.values[0].strftime('%Y-%m-%dT%H:%M:%S'),], showlegend=False,
    #                 textposition='middle right',
    #                 textfont=dict(size=12, color='black'))
    #             )

    # print(['Begin: '+T2.values[0].strftime('%Y-%m-%dT%H:%M:%S'),'End: '+T2.values[-1].strftime('%Y-%m-%dT%H:%M:%S'),])

    # fig.add_trace(go.Scattermapbox(lon=X2[[0,-1]],lat=Y2[[0,-1]],
    #                 mode='markers+text',
    #                 marker=dict(size=10, symbol='star'),
    #                 text=['Begin: '+T2.values[0].strftime('%Y-%m-%dT%H:%M:%S'),'End: '+T2.values[-1].strftime('%Y-%m-%dT%H:%M:%S'),],
    #                 showlegend=False,
    #                 textposition='middle right',
    #                 textfont=dict(size=12, color='black'))
    #             )

    ##
    ## 4. Final formatting steps
    ##

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    ## Add map grid lines.
    for y in np.arange(-80,85,5):
        fig.add_trace(go.Scattermapbox(lon=np.array([0,360]),lat=np.array([y,y]),
                        mode='lines', line={'width':0.3, 'color':'grey'}, showlegend=False, legendgroup='llgrid'))
    for y in [0.0,]:
        fig.add_trace(go.Scattermapbox(lon=np.array([0,360]),lat=np.array([y,y]),
                        mode='lines', line={'width':0.3, 'color':'grey'}, showlegend=True, legendgroup='llgrid', name='Lat/Lon Grid'))
    for x in np.arange(0,365,5):
        fig.add_trace(go.Scattermapbox(lon=np.array([x,x]),lat=np.array([-90,90]),
                        mode='lines', line={'width':0.3, 'color':'grey'}, showlegend=False, legendgroup='llgrid'))

    ##
    ## 5. Output
    ##
    fn_out_html = (html_out_dir + '/individual_research_cruise_map__'+this_cruise+'__sst.html')
    print('--> '+fn_out_html)
    os.makedirs(html_out_dir, exist_ok=True)
    fig.write_html(fn_out_html, include_plotlyjs='cdn')


print('Done.')
