#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:30:12 2021

@author: arvind
"""
###############################
import os
os.chdir("/home/arvind/arvind/jjas")


import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

##################################
ds =xr.open_dataset('2003_2017_Prate_ens_jjas_fcst.nc',decode_times = False)
ds1 =xr.open_dataset('gpcp_jjas_2003_2017.nc')













lat = ds.variables['Y'][:]
lon = ds.variables['X'][:]
#obs = ds1.variables['precip'][:,:,:,:]
#fcst = ds.variables['precip'][:,:,:,:]

time = ds1.variables['T'][:]
pcp_obs = ds1.precip.values


pcp_fcst =ds.prcp.values*86400








########################### functions #######################
def BC(pcp_fcst,pcp_obs):
    mean_pcp_fcst = np.nanmean(pcp_fcst , axis = 0)
    mean_pcp_obs  = np.nanmean(pcp_obs,   axis = 0)
    std_pcp_fcst  = np.std(pcp_fcst , axis = 0)
    std_pcp_obs   = np.std(pcp_obs , axis = 0)
    fcst_anom     = pcp_fcst - mean_pcp_fcst
    temp = np.divide(std_pcp_obs,std_pcp_fcst)
    temp1 =np.multiply(temp, fcst_anom)
    BC   = temp1 +mean_pcp_obs
    return BC
bias_pcp_fcst = BC(pcp_fcst, pcp_obs)



def ACCR(pcp_obs, bias_pcp_fcst):    
    mean_pcp_obs = np.nanmean(pcp_obs, axis =0)
    mean_pcp_fcst = np.nanmean(bias_pcp_fcst, axis = 0)
    anom_o       = pcp_obs - mean_pcp_obs
    anom_f       = pcp_fcst - mean_pcp_fcst
    mean_anom_obs = np.nanmean(anom_o , axis = 0)
    mean_anom_fcst = np.nanmean(anom_f ,axis = 0)
    anom_diff_o =anom_o - mean_anom_obs
    anom_diff_f =anom_f - mean_anom_fcst
    std_anom_o  =np.std(anom_o, axis =0)
    std_anom_f =np.std(anom_f, axis =0)
    bottom = np.multiply(std_anom_o , std_anom_f)
    temp = np.multiply(anom_diff_o, anom_diff_f)
    top = np.nanmean(temp, axis = 0)
    ACCR = top/bottom
    return ACCR
ACOR = ACCR(pcp_obs,bias_pcp_fcst)


from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

#levels = [-0.45,-0.40,-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.00,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.45,0.50]



#colours = ['#730000', '#b30909', '#e62222', '#de652c', '#e6a235', '#f7c16a', '#fff587', '#fcf8c7', '#fffef7', 
#'#e3e1f7','#cbc7eb', '#aea8e3', '#847bd4', '#776be3', '#6254e3', '#2a15eb', '#1607a3', '#0f0385','#140b6b']



#colours =  ['#140b6b',  '#0f0385' , '#1607a3', '#2a15eb','#6254e3','#776be3','#847bd4',
#          '#aea8e3','#cbc7eb','#e3e1f7','#fffef7','#fcf8c7', '#fff587',  '#e6a235',
#            '#de652c', '#e62222','#b30909','#730000']

#colours = grayscale_cmap('jet')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def discrete_cmap(N, base_cmap=None):
	"""Create an N-bin discrete colormap from the specified input map"""
	# Note that if base_cmap is a string or None, you can simply do
	#    return plt.cm.get_cmap(base_cmap, N)
	# The following works for string, None, or a colormap instance:
	base = plt.cm.get_cmap(base_cmap)
	color_list = base(np.linspace(0, 1, N))
	cmap_name = base.name + str(N)
	#base.set_bad(color='white')
	#return base.from_list(cmap_name, color_list, N)
	return LinearSegmentedColormap.from_list(cmap_name, color_list, N)

#discrete_cmap(10, base_cmap = 'jet')

#cmap1 = LinearSegmentedColormap.from_list("",["red","violet","blue"], 10)
cmap = LinearSegmentedColormap.from_list("", ["green","white","brown"],10)
#blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

cp= plt.contourf(lon, lat,ACOR ,transform=ccrs.PlateCarree(),cmap ="jet")
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')
plt.title("Above normal_test",
          fontsize = '14', loc ='Center')

pos = ax.get_position()
print(pos)
cbar_ax = fig.add_axes([pos.x0, 0.28, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',drawedges=True)
cb.dividers.set_color('black')
fig1 = plt.gcf()
fig1.savefig('Above_normal.png', dpi=100)
plt.legend()






def rmse(pcp_fcst,pcp_obs):
    temp = np.mean(((pcp_fcst- pcp_obs)**2),axis =0)
    rmse = np.sqrt(temp)
    return rmse
def bias(pcp_fcst, pcp_obs):
    diff = pcp_fcst - pcp_obs
    bias = np.mean(diff , axis = 0)
    return bias
def MSSS_bias(pcp_obs, bias_pcp_fcst):
   
    var_obs_pcp  = np.var(pcp_obs , axis = 0)
    
    N = len(time)
    MSE_C = ((N-1)/(N))*var_obs_pcp

    diff_sq =(bias_pcp_fcst - pcp_obs)**2
    MSE_f  = np.average(diff_sq, axis =0)
    MSSS_bias = 1 -(MSE_f/MSE_C)
    return MSSS_bias
def std_pcp_obs(pcp_obs):
    std_pcp_obs = np.std(pcp_obs, axis = 0)
    return std_pcp_obs
def std_pcp_fcst(pcp_fcst):
    std_pcp_fcst = np.std(pcp_fcst ,axis = 0)
    return std_pcp_fcst


bias_pcp_fcst = BC(pcp_fcst, pcp_obs)
ACOR = ACCR(pcp_obs, bias_pcp_fcst)
RMSE = rmse(bias_pcp_fcst, pcp_obs)
BIAS = bias(pcp_fcst,pcp_obs)
MSSS_bias=MSSS_bias(pcp_obs , bias_pcp_fcst)
CLIM_OBS =np.nanmean(pcp_obs, axis = 0)
CLIM_FCST =np.nanmean(pcp_fcst, axis = 0)
SD_OBS = std_pcp_obs(pcp_obs)
SD_FCST= std_pcp_fcst(pcp_fcst)








scores = [ACOR,MSSS_bias,CLIM_OBS,CLIM_FCST]
titles  = ['ACOR','MSSS','Climatology observed','Climatology forecasted']
cmap  = ['seismic','RdBu','Spectral','bwr']


def plot(lon,lat,score):
    fig = plt.figure(figsize=(12,8))
    plt.xlabel('Longitude', fontsize ='4')
    plt.ylabel('Latitude', fontsize = '4')
 #   fig, ax = plt.subplots(2, 4)
    
    
    for i in range(4):
      
        ax = plt.subplot(2,2,i+1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_global()
        ax.coastlines('110m')
      
       
        for score in scores:
            for cm in cmap:
                if i==0:
                    levels = [-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ]
                    colours =  ['#140b6b',  '#0f0385' , '#1607a3', '#2a15eb','#6254e3','#776be3','#847bd4',
            '#aea8e3','#cbc7eb','#e3e1f7','#fffef7','#fcf8c7', '#fff587',  '#e6a235',
            '#de652c', '#e62222','#b30909','#730000']
                cp =plt.contourf(lon,lat,scores[i],transform =ccrs.PlateCarree(),colors =colours, levels =levels, extend ='both')
                if i==1:
                    levels = [-0.9,-0.8,-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    colours =['#02385c', '#306e96','#5586a6','#779db5', '#9cc3db', '#c0e4fa', '#e3f3fc',
          '#f5fbff', '#fcfcfc', '#ffffff', '#fff8e0', '#ffd796', '#ffa95e',
          '#ff8345', '#f7794f', '#b82e00','#b81c00','#850707']
                cp =plt.contourf(lon,lat,scores[i],transform =ccrs.PlateCarree(),colors =colours, levels =levels, extend = 'both')

                if i==2:
                    levels =[0.5,1,2,3,4,5,6,8,10,12,14,16,20]
                    colours =['#ffffff' ,'#8dc0e3','#63aadb','#2361b8', '#02429c', '#2f9459',
          '#2cbf69', '#74ad23', '#f0eb56','#f0d156', '#e8bf1c','#d18828',
          '#b50d1e', '#73000c']
                cp =plt.contourf(lon,lat,scores[i],transform =ccrs.PlateCarree(),colors =colours, levels =levels, extend = 'both')

                
                if i==3:
                    levels =[0.5,1,2,3,4,5,6,8,10,12,14,16,20]
                    colours =['#ffffff' ,'#8dc0e3','#63aadb','#2361b8', '#02429c', '#2f9459',
          '#2cbf69', '#74ad23', '#f0eb56','#f0d156', '#e8bf1c','#d18828',
          '#b50d1e', '#73000c']
                    cp =plt.contourf(lon,lat,scores[i],transform =ccrs.PlateCarree(),colors =colours, levels =levels, extend = 'both')

                    
                    

                    

                    
                
                
                
      
        plt.colorbar(cp,orientation ='vertical',shrink = 0.6)
            
                             
        for title in titles:
            plt.title("%s"%titles[i]  , fontsize='8',loc = 'center')
   
        ax.set_xticks([0,60, 120, 180,240, 300, 360], crs=ccrs.PlateCarree())
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
            
        
    
        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        
        plt.ylabels_left = False
        if i==0:
            plt.ylabel("Latitude",fontsize = '12')
        if i==3:
            plt.xlabel('Longitude', fontsize ='12')
            
    fig1 = plt.gcf()
    fig1.savefig('temp.png', dpi=150)
 
    plt.show()
plot(lon,lat,scores)



























