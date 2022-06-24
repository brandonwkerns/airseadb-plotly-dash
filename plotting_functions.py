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


