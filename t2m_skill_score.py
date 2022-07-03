############# Surface Temperature Skill Score ##########
import os
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4



IC = 'JanIC'
seas = 'fma'
target = 'FEB-APR'
initial = 'Jan'
lead_time ='1'
mon = 'JAN'




##############Loading Data Sets #################################



ds =xr.open_dataset('/data2/home/arvind/HC_Data/'+mon+'/T2M/ncep_t2m_2003_2017_'+seas+'.nc')
ds1 =xr.open_dataset('/data2/home/arvind/HC_Data/'+mon+'/T2M/T2M_regrid_2003_2017_'+seas+'_'+IC+'.nc')




lat = ds.variables['Y'][:]
lon = ds.variables['X'][:]
time = ds.variables['T'][:]


obs  = ds.temp.values+273.15


fcst = ds1.t2m.values

np.seterr(divide='ignore', invalid='ignore')


#################### Anomaly Correlation Coefficient #############



def BC(fcst, obs):
    fcst_mean = np.nanmean(fcst, axis = 0)
    obs_mean  = np.nanmean(obs, axis = 0)
    sd_fcst   = np.std(fcst , axis = 0)
    sd_obs    = np.std(obs, axis = 0)
    fcst_anom = fcst - fcst_mean
    BC_1          = fcst_anom + obs_mean
    temp  = np.divide(sd_obs, sd_fcst)
    temp1 = np.multiply(temp,fcst_anom)
    BC    = temp1 + obs_mean
    return BC
bias_fcst = BC(fcst,obs)


def ACCR(obs, bias_fcst):
    obs_mean  = np.nanmean(obs, axis = 0)
    fcst_mean = np.nanmean(fcst, axis = 0)
    anom_o       = obs - obs_mean
    anom_f    = fcst - fcst_mean
    mean_anom_obs = np.nanmean(anom_o,axis = 0)
    mean_anom_fcst =np.nanmean(anom_f, axis = 0)
    anom_diff_o =anom_o - mean_anom_obs
    anom_diff_f =anom_f - mean_anom_fcst
    std_anom_o  =np.std(anom_o, axis =0)
    std_anom_f =np.std(anom_f, axis =0)
    bottom = np.multiply(std_anom_o , std_anom_f)
    temp = np.multiply(anom_diff_o, anom_diff_f)
    top = np.nanmean(temp, axis = 0)
    ACCR = top/bottom
    return ACCR


ACOR = ACCR(obs, fcst)



SH = round(np.nanmean(ACOR[44:72,:]),4)
TRP = round(np.nanmean(ACOR[28:44,:]),4)
NH = round(np.nanmean(ACOR[0:28, 0:72]), 4)
EU = round(np.nanmean(ACOR[0:28,0:72]),4)
IND = round(np.nanmean(ACOR[20:36,22:44]),4)
PAC = round(np.nanmean(ACOR[0:28,48:120]),4)
GLB = round(np.nanmean(ACOR),4)



















#############  Plotting ACOR ################# 
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt





scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()
levels = [-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ]

colours =  ['#140b6b',  '#0f0385' , '#1607a3', '#2a15eb','#6254e3','#776be3','#847bd4',
            '#aea8e3','#cbc7eb','#e3e1f7','#fffef7','#fcf8c7', '#fff587',  '#e6a235',
            '#de652c', '#e62222','#b30909','#730000']
cp = plt.contourf(lon, lat,ACOR ,transform=ccrs.PlateCarree(),levels = levels ,colors =colours, extend ='both')
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')

plt.title('['+target+'/MMCFSV2:CPC]\nTemperature Anomaly(with bias-correction)\nAnomaly Correlation for  15 yrs(2003-2017)\nInitial : '+initial+', Lead time = '+lead_time+' (Target month : '+target+')\n',
          fontsize = '14', loc ='left')


pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',ticks = levels, drawedges=True)
cb.dividers.set_color('black')

plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, GLB, bbox ={'facecolor':'None'})
fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/'+mon+'/plot_t2m/'+seas+'/acor1.png', dpi=100)
plt.legend()





np.seterr(divide='ignore', invalid='ignore')

############################## RMSE ############
def rmse(fcst,obs):
    temp = np.mean(((fcst- obs)**2),axis =0)
    rmse = np.sqrt(temp)
    return rmse

result = rmse(bias_fcst, obs)

SH = round(np.nanmean(result[44:72,:]),4)
TRP = round(np.nanmean(result[28:44,:]),4)
NH = round(np.nanmean(result[0:28, 0:72]), 4)
EU = round(np.nanmean(result[0:28,0:72]),4)
IND = round(np.nanmean(result[20:36,22:44]),4)
PAC = round(np.nanmean(result[0:28,48:120]),4)
GLB = round(np.nanmean(result),4)




scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()
levels = [0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5]
colours =['#ffffff' ,'#8dc0e3','#63aadb','#2361b8', '#02429c', '#2f9459',
          '#2cbf69', '#74ad23', '#f0eb56','#f0d156', '#e8bf1c','#d18828',
          '#b50d1e', '#73000c']

cp = plt.contourf(lon, lat,result ,transform=ccrs.PlateCarree(),levels = levels ,colors =colours, extend ='both')
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')

plt.title('['+target+'/MMCFSV2:CPC]\nTemperature Anomaly(with bias-correction)[K]\nRoot Mean Squre Error for  15 yrs(2003-2017)\nInitial : '+initial+', Lead time = '+lead_time+' (Target month : '+target+')\n',
          fontsize = '14', loc ='left')


pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',ticks = levels,drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, GLB, bbox ={'facecolor':'None'})






fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/'+mon+'/plot_t2m/'+seas+'/rmse.png', dpi=100)
plt.legend()

############################## BIAS ###########################

def bias(fcst, obs):
    diff = fcst - obs
    bias = np.mean(diff , axis = 0)
    return bias
#######################Regional Statastics Calculation ###############
BIAS = bias(fcst,obs)
SH = round(np.nanmean(BIAS[44:72,:]),4)
TRP = round(np.nanmean(BIAS[28:44,:]),4)
NH = round(np.nanmean(BIAS[0:28, 0:72]), 4)
EU = round(np.nanmean(BIAS[0:28,0:72]),4)
IND = round(np.nanmean(BIAS[20:36,22:44]),4)
PAC = round(np.nanmean(BIAS[0:28,48:120]),4)
GLB = round(np.nanmean(BIAS),4)

##############Plotting##########
scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

levels = [-8, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6, 8]

colours =['#6e010c' ,'#a80819','#de211b', '#e69022', '#ffb95e', '#ffd39c',
          '#ffeacf', '#fff8f0','#ffffff', '#fcfbfa', '#cee7f2', '#9ccfe6',
          '#6eb4d4', '#3ba2d1', '#2186b5', '#0e72a1', '#066c9c', '#005178']





cp= plt.contourf(lon, lat,BIAS ,transform=ccrs.PlateCarree(),
                 levels = levels ,colors =colours, extend ='both')
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')
plt.title('['+target+'/MMCFSV2 : CPC]\nTemperature (Kelvin)\nBIAS for  15 yrs(2003-2017)\nInitial : '+initial+', Lead time = '+lead_time+' (Target month :'+target+')\n',
          fontsize = '14', loc ='left')


pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',ticks = levels, drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, GLB, bbox ={'facecolor':'None'})









fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/'+mon+'/plot_t2m/'+seas+'/bias.png', dpi=100)
plt.legend()

################## Mean Squre Skill Score #################



def MSSS_bias(obs, bias_fcst):
    mean_obs = np.nanmean(obs,axis= 0)
    var_obs  = np.var(obs , axis = 0)
    std_obs  = np.std(obs , axis = 0)
    N = len(time)
    MSE_C = ((N-1)/(N))*var_obs

    diff_sq =(bias_fcst - obs)**2
    MSE_f  = np.average(diff_sq, axis =0)
    MSSS_bias = 1 -(MSE_f/MSE_C)
    return MSSS_bias


MSSS_bias= MSSS_bias(obs , bias_fcst)

SH = round(np.nanmean(MSSS_bias[44:72,:]),4)
TRP = round(np.nanmean(MSSS_bias[28:44,:]),4)
NH = round(np.nanmean(MSSS_bias[0:28, 0:72]), 4)
EU = round(np.nanmean(MSSS_bias[0:28,0:72]),4)
IND = round(np.nanmean(MSSS_bias[20:36,22:44]),4)
PAC = round(np.nanmean(MSSS_bias[0:28,48:120]),4)
GLB = round(np.nanmean(MSSS_bias),4)

scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

levels = [-0.9,-0.8,-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#colours =['#6e010c' ,'#a80819','#de211b', '#d96436', '#bf4830', '#de9973',
 #         '#ffeacf', '#fff8f0','#ffffff', '#fcfbfa', '#cee7f2', '#9ccfe6',
  #        '#6eb4d4', '#3ba2d1', '#2186b5', '#0e72a1', '#066c9c', '#005178']

colours =['#02385c', '#306e96','#5586a6','#779db5', '#9cc3db', '#c0e4fa', '#e3f3fc',
          '#f5fbff', '#fcfcfc', '#ffffff', '#fff8e0', '#ffd796', '#ffa95e',
          '#ff8345', '#f7794f', '#b82e00','#b81c00','#850707']


cp= plt.contourf(lon, lat,MSSS_bias ,transform=ccrs.PlateCarree(),levels = levels,
            colors =colours, extend ='both')
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')
#plt.title("Precipitation MSSS with BC for season NDJ 2003-2019(17yrs) ", fontsize = '16')
#cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.025,ax.get_position().height])
#plt.colorbar(cp, cax=cax)
plt.title('['+target+'/MMCFSV2: CPC]\nTemperature (Kelvin)\nMSSS with BC for 15 yrs(2003-2017)\nInitial : '+initial+', Lead time = '+lead_time+' (Target month : '+target+')\n', fontsize = '14' , loc = 'left')


pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',ticks = levels,drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, GLB, bbox ={'facecolor':'None'})


fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/'+mon+'/plot_t2m/'+seas+'/MSSS.png', dpi=100)
plt.legend()

################## Climatology Observed ####################3


clim_obs = np.nanmean(obs, axis = 0)
SH = round(np.nanmean(clim_obs[44:72,:]),4)
TRP = round(np.nanmean(clim_obs[28:44,:]),4)
NH = round(np.nanmean(clim_obs[0:28, 0:72]), 4)
EU = round(np.nanmean(clim_obs[0:28,0:72]),4)
IND = round(np.nanmean(clim_obs[20:36,22:44]),4)
PAC = round(np.nanmean(clim_obs[0:28,48:120]),4)
GLB = round(np.nanmean(clim_obs),4)





scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()



levels = [233, 238, 243, 248, 253, 263, 268, 273, 278,283,288,293,298,303,308,313]



#colours =['#ffffff' ,'#8dc0e3','#63aadb','#2361b8', '#02429c', '#2f9459',
#          '#2cbf69', '#74ad23', '#f0eb56','#f0d156', '#e8bf1c','#d18828',
#          '#b50d1e', '#73000c']





cp= plt.contourf(lon, lat,clim_obs ,transform=ccrs.PlateCarree(),levels =levels,cmap ='RdBu', extend ='both')
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')
plt.title('['+target+'/CPC]\nTemperature [K] \nTemperature Climatology Observed 15 yrs(2003-2017)\nInitial : '+initial+', Lead time = '+lead_time+' (Target month : '+target+')\n',          fontsize = '14', loc ='left')



pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',ticks = levels,drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, GLB, bbox ={'facecolor':'None'})

fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/'+mon+'/plot_t2m/'+seas+'/clim_obs.png', dpi=100)
plt.legend()


####################### Climatology forecasted ########################


clim_fcst = np.nanmean(fcst, axis = 0)
SH = round(np.nanmean(clim_fcst[44:72,:]),4)
TRP = round(np.nanmean(clim_fcst[28:44,:]),4)
NH = round(np.nanmean(clim_fcst[0:28, 0:72]), 4)
EU = round(np.nanmean(clim_fcst[0:28,0:72]),4)
IND = round(np.nanmean(clim_fcst[20:36,22:44]),4)
PAC = round(np.nanmean(clim_fcst[0:28,48:120]),4)
GLB = round(np.nanmean(clim_fcst),4)





scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()



levels = [233, 238, 243, 248, 253, 263, 268, 273, 278,283,288,293,298,303,308,313]

#colours =['#ffffff' ,'#8dc0e3','#63aadb','#2361b8', '#02429c', '#2f9459',
#          '#2cbf69', '#74ad23', '#f0eb56','#f0d156', '#e8bf1c','#d18828',
#          '#b50d1e', '#73000c']





cp= plt.contourf(lon, lat,clim_fcst ,transform=ccrs.PlateCarree(),levels =levels,cmap = 'RdBu', extend ='both')
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.OCEAN)


plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')
plt.title('['+target+'/MMCFSV2]\nTemperature [K] \nClimatology Forecasted for 15 yrs(2003-2017)\nInitial : '+initial+', Lead time = '+lead_time+' (Target month : '+target+')\n',
          fontsize = '14', loc ='left')

pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',ticks = levels, drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, GLB, bbox ={'facecolor':'None'})
fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/'+mon+'/plot_t2m/'+seas+'/clim_fcst.png', dpi=100)
plt.legend()



####################standard deviation forecasted #################



def std_fcst(fcst):
    std_fcst = np.std(fcst ,axis = 0)
    return std_fcst
SD_model    = std_fcst(fcst)

SH = round(np.nanmean(SD_model[44:72,:]),4)
TRP = round(np.nanmean(SD_model[28:44,:]),4)
NH = round(np.nanmean(SD_model[0:28, 0:72]), 4)
EU = round(np.nanmean(SD_model[0:28,0:72]),4)
IND = round(np.nanmean(SD_model[20:36,22:44]),4)
PAC = round(np.nanmean(SD_model[0:28,48:120]),4)
GLB = round(np.nanmean(SD_model),4)




scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

levels = [0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5]
colours =['#ffffff' ,'#8dc0e3','#63aadb','#2361b8', '#02429c', '#2f9459',
          '#2cbf69', '#74ad23', '#f0eb56','#f0d156', '#e8bf1c','#d18828',
          '#b50d1e', '#73000c']
cp= plt.contourf(lon, lat,SD_model,transform=ccrs.PlateCarree(),
                 levels = levels ,colors =colours, extend ='both')
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')
plt.title('['+target+'/MMCFSV2]\nTemperature [K] \nStandard deviation forecast for 15 yrs (2003-2017) \nInitial : '+initial+', Lead time = '+lead_time+' (Target month : '+target+')\n',
          fontsize = '14', loc ='left')

pos = ax.get_position()

cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',ticks= levels,drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, GLB, bbox ={'facecolor':'None'})
fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/'+mon+'/plot_t2m/'+seas+'/Standard_deviation_forecasted.png', dpi=100)
plt.legend()



#######################################################
def std_obs(obs):
    std_obs = np.std(obs ,axis = 0)
    return std_obs
SD_obs    = std_obs(obs)
SH = round(np.nanmean(SD_obs[44:72,:]),4)
TRP = round(np.nanmean(SD_obs[28:44,:]),4)
NH = round(np.nanmean(SD_obs[0:28, 0:72]), 4)
EU = round(np.nanmean(SD_obs[0:28,0:72]),4)
IND = round(np.nanmean(SD_obs[20:36,22:44]),4)
PAC = round(np.nanmean(SD_obs[0:28,48:120]),4)
GLB = round(np.nanmean(SD_obs),4)






scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

levels = [0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5]
colours =['#ffffff' ,'#8dc0e3','#63aadb','#2361b8', '#02429c', '#2f9459',
          '#2cbf69', '#74ad23', '#f0eb56','#f0d156', '#e8bf1c','#d18828',
          '#b50d1e', '#73000c']
cp= plt.contourf(lon, lat,SD_obs,transform=ccrs.PlateCarree(),
                 levels = levels ,colors =colours, extend ='both')
ax.coastlines(scale)
ax.set_xticks([0, 30,60,90, 120,150, 180,210, 240,270, 300,330, 360], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.xlabel('Longitude', fontsize ='14')
plt.ylabel('Latitude', fontsize = '14')
plt.title('['+target+'/MMCFSV2]\nTemperature [K] \nStandard deviation forecast for 15 yrs (2003-2017) \nInitial : '+initial+', Lead time = '+lead_time+' (Target month : '+target+')\n',
          fontsize = '14', loc ='left')

pos = ax.get_position()

cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',ticks= levels,drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "GLB")
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, GLB, bbox ={'facecolor':'None'})
fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/'+mon+'/plot_t2m/'+seas+'/Standard_deviation_observed.png', dpi=100)
plt.legend()














