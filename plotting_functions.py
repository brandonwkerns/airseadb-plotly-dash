import numpy as np
import pandas as pd
import cftime
import plotly.express as px
import plotly.graph_objects as go
import colorcet as cc
import sqlite3
import os

## Color Palette ############
color_dark_1 = '#1d191e'
color_dark_2 = '#1f2138'
color_dark_3 = '#101125'
color_light_1 = '#a4967a'
color_light_2 = '#bfb39b'
#############################


def get_cbar_range(data, force_range=None):
    min_data_value = np.nanmin(data)
    max_data_value = np.nanmax(data)
    if not (force_range is None):
        mid_data_value = np.nanmedian(data)
        min_data_value = mid_data_value - 0.5*force_range
        max_data_value = mid_data_value + 0.5*force_range
    return (min_data_value, max_data_value)


def create_cruise_track_trace(lon, lat):
    return go.Scattermapbox(lon=lon,lat=lat,
                    mode='lines', line={'width':2.0, 'color':'black'}, name='Cruise Track',
                    showlegend=True, hoverinfo='skip') 


def create_data_markers_trace(lon, lat, T, Z, label, skip=10):

    cmin,cmax = get_cbar_range(Z)

    return go.Scattermapbox(lon=lon[::skip], lat=lat[::skip],
                    marker=dict(color=Z[::skip], cmin=cmin, cmax=cmax, colorscale=cc.CET_R1,
                        colorbar=dict(len=0.6, title=label, y=0.1, yanchor='bottom', x=0.99, xanchor='right',bgcolor=color_light_2)),
                    hovertemplate = '%{text}<br>lon: %{lon:.2f}<br>lat: %{lat:.2f}<br>'+label+': %{marker.color:.2f}',
                    text=T,
                    name = 'Cruise Data', showlegend=True)


def add_grid_lines(fig, dx=5, width=0.3, color='grey'):
    for y in np.arange(-80,80+dx,dx):
        fig.add_trace(go.Scattermapbox(lon=np.array([0,360]),lat=np.array([y,y]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=False, legendgroup='llgrid'))
    for y in [0.0,]:
        fig.add_trace(go.Scattermapbox(lon=np.array([0,360]),lat=np.array([y,y]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=True, legendgroup='llgrid', name='Lat/Lon Grid'))
    for x in np.arange(0,360+dx,dx):
        fig.add_trace(go.Scattermapbox(lon=np.array([x,x]),lat=np.array([-90,90]),
                        mode='lines', line={'width':width, 'color':color}, showlegend=False, legendgroup='llgrid'))


