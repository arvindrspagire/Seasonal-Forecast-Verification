import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4
ds =xr.open_dataset("/data2/home/arvind/HC_Data/MAR/Rainfall/gpcp_2003_2017_ond.nc")
ds1 =xr.open_dataset("/data2/home/arvind/HC_Data/MAR/Rainfall/ens_regrid_2003_2017_ond_MarIC.nc")
lat = ds.variables['Y'][:]
lon = ds.variables['X'][:]
pcp_obs = ds.variables['precip'][:,:,:]
pcp_fcst = ds1.variables['precip'][:,:,:]
time = ds.variables['T'][:]





























################## Anomaly Correlation Coefficient ##################
pcp_obs = ds.precip.values
pcp_fcst =ds1.precip.values*86400
def BC(pcp_fcst,pcp_obs):
    mean_pcp_fcst = np.nanmean(pcp_fcst , axis = 0)
    mean_pcp_obs  = np.nanmean(pcp_obs,   axis = 0)
    std_pcp_fcst  = np.std(pcp_fcst , axis = 0)
    std_pcp_obs   = np.std(pcp_obs , axis = 0)
    fcst_anom     = pcp_fcst - mean_pcp_fcst
    BC_1          = fcst_anom + mean_pcp_obs
    temp =  (std_pcp_obs/ std_pcp_fcst)
    temp1 = temp*fcst_anom
    BC   = temp1 +mean_pcp_obs
    return BC
bias_pcp_fcst = BC(pcp_fcst, pcp_obs)

def ACCR(pcp_obs, bias_pcp_fcst):    
    mean_pcp_obs = np.nanmean(pcp_obs, axis =0)
    mean_pcp_fcst = np.nanmean(pcp_fcst, axis = 0)
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
ACOR = ACCR(pcp_obs, pcp_fcst)
SH = round(np.nanmean(ACOR[44:72,:]), 4)
TRP = round(np.nanmean(ACOR[28:44,:]),4)
NH  =round(np.nanmean(ACOR[0:28,0:72]),4)
EU  = round(np.nanmean(ACOR[0:28,0:72]),4)
IND =round(np.nanmean(ACOR[20:36,22:44]),4)
PAC = round(np.nanmean(ACOR[0:28,48:120]),4)
N34 =round(np.nanmean(ACOR[33:39,68:128]),4)
GLB =round(np.nanmean(ACOR), 4)

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

levels = [-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ]



#colours = ['#730000', '#b30909', '#e62222', '#de652c', '#e6a235', '#f7c16a', '#fff587', '#fcf8c7', '#fffef7', 
#'#e3e1f7','#cbc7eb', '#aea8e3', '#847bd4', '#776be3', '#6254e3', '#2a15eb', '#1607a3', '#0f0385','#140b6b']



colours =  ['#140b6b',  '#0f0385' , '#1607a3', '#2a15eb','#6254e3','#776be3','#847bd4',
            '#aea8e3','#cbc7eb','#e3e1f7','#fffef7','#fcf8c7', '#fff587',  '#e6a235',
            '#de652c', '#e62222','#b30909','#730000']


cp= plt.contourf(lon, lat,ACOR ,transform=ccrs.PlateCarree(),
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
plt.title("[OCT-DEC/MMCFSV2: GPCP_V2.3]\nRainfall Anomaly(with bias-correction)\nAnomaly Correlation for  15 yrs(2003-2017)\nInitial : Mar, Lead time = 3 (Target month : OCT-DEC)\n",
          fontsize = '14', loc ='left')

pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',drawedges=True)

cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "N34")
plt.figtext(pos.x0+0.70, 0.18,"GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.60, 0.16, N34, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.70, 0.16, GLB, bbox ={'facecolor':'None'})

fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/MAR/plot/ond/acor.png', dpi=100)
plt.legend()








#########################RMSE#######################################



pcp_obs = ds.precip.values
pcp_fcst =ds1.precip.values*86400
def BC(pcp_fcst,pcp_obs):
    mean_pcp_fcst = np.nanmean(pcp_fcst , axis = 0)
    mean_pcp_obs  = np.nanmean(pcp_obs,   axis = 0)
    std_pcp_fcst  = np.std(pcp_fcst , axis = 0)
    std_pcp_obs   = np.std(pcp_obs , axis = 0)
    fcst_anom     = pcp_fcst - mean_pcp_fcst
    BC_1          = fcst_anom + mean_pcp_obs
    temp =  (std_pcp_obs/ std_pcp_fcst)
    temp1 = temp*fcst_anom
    BC   = temp1 +mean_pcp_obs
    return BC
pcp_fcst = ds1.precip.values
pcp_obs  = ds.precip.values
bias_pcp_fcst = BC(pcp_fcst, pcp_obs)








def rmse(pcp_fcst,pcp_obs):
    temp = np.mean(((pcp_fcst- pcp_obs)**2),axis =0)
    rmse = np.sqrt(temp)
    return rmse

    
result = rmse(bias_pcp_fcst, pcp_obs)

SH = round(np.nanmean(result[44:72,:]), 4)
TRP = round(np.nanmean(result[28:44,:]),4)
NH  =round(np.nanmean(result[0:28,:]),4)
EU  = round(np.nanmean(result[0:28,0:72]),4)
IND =round(np.nanmean(result[20:36,22:44]),4)
PAC = round(np.nanmean(result[0:28,48:120]),4)
N34 =round(np.nanmean(result[33:39,68:128]),4)
GLB =round(np.nanmean(result), 4)

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

levels = [0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5]
colours =['#ffffff' ,'#8dc0e3','#63aadb','#2361b8', '#02429c', '#2f9459', 
          '#2cbf69', '#74ad23', '#f0eb56','#f0d156', '#e8bf1c','#d18828',
          '#b50d1e', '#73000c']
cp= plt.contourf(lon, lat,result ,transform=ccrs.PlateCarree(),
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
plt.title("[OCT-DEC/MMCFSV2 : GPCP_V2.3]\nRAINFALL [mm/day]\nRMSE with bias correction for 15 yrs(2003-2017)\nInitial : Mar, Lead time = 3 (Target month : OCT-DEC)\n",
          fontsize = '14', loc ='left')


pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "N34")
plt.figtext(pos.x0+0.70, 0.18,"GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.60, 0.16, N34, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.70, 0.16, GLB, bbox ={'facecolor':'None'} )
#rectangle = pac.Rectangle((0,1), 10,5, fc='None',ec="black")
#plt.gca().add_patch(rectangle)

fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/MAR/plot/ond/rmse.png', dpi=100)
plt.legend()

#######################BIAS########################################
pcp_obs = ds.precip.values
pcp_fcst =ds1.precip.values*86400




def bias(pcp_fcst, pcp_obs):
    diff = pcp_fcst - pcp_obs
    bias = np.mean(diff , axis = 0)
    return bias
BIAS = bias(pcp_fcst,pcp_obs)
SH = round(np.nanmean(BIAS[44:72,:]), 4)
TRP = round(np.nanmean(BIAS[28:44,:]),4)
NH  =round(np.nanmean(BIAS[0:28,:]),4)
EU  = round(np.nanmean(BIAS[0:28,0:72]),4)
IND =round(np.nanmean(BIAS[20:36,22:44]),4)
PAC = round(np.nanmean(BIAS[0:28,48:120]),4)
N34 =round(np.nanmean(BIAS[33:39,68:128]),4)
GLB =round(np.nanmean(BIAS), 4)




from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
scale = '110m'
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

levels = [-8, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6, 8]

colours =['#6e010c' ,'#a80819','#de211b', '#d96436', '#bf4830', '#de9973',
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
plt.title("[OCT-DEC/MMCFSV2 : GPCP_V2.3]\nRAINFALL [mm/day]\nBIAS for  15 yrs(2003-2017)\nInitial : Mar, Lead time = 3 (Target month : OCT-DEC)\n",
          fontsize = '14', loc ='left')


pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "N34")
plt.figtext(pos.x0+0.70, 0.18,"GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.60, 0.16, N34, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.70, 0.16, GLB, bbox ={'facecolor':'None'} )


#rectangle = pac.Rectangle(,1), 10,5, fc='None',ec="black")
#plt.gca().add_patch(rectangle)

fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/MAR/plot/ond/bias.png', dpi=100)
plt.legend()


############################# Mean Square Skill Score ##################



pcp_fcst = ds1.precip.values*86400
pcp_obs = ds.precip.values

# Bias Correction
def BC(pcp_fcst,pcp_obs):
    mean_pcp_fcst = np.nanmean(pcp_fcst , axis = 0)
    mean_pcp_obs  = np.nanmean(pcp_obs,   axis = 0)
    std_pcp_fcst  = np.std(pcp_fcst , axis = 0)
    std_pcp_obs   = np.std(pcp_obs , axis = 0)
    fcst_anom     = pcp_fcst - mean_pcp_fcst
    BC_1          = fcst_anom + mean_pcp_obs
    temp =  (std_pcp_obs/ std_pcp_fcst)
    temp1 = temp*fcst_anom
    BC   = temp1 +mean_pcp_obs
    return BC
bias_pcp_fcst = BC(pcp_fcst, pcp_obs)


#bC_pcp_fcst = BC(pcp_fcst, pcp_obs)
pcp_obs = ds.precip.values



def MSSS_bias(pcp_obs, bias_pcp_fcst):
    mean_obs_pcp = np.nanmean(pcp_obs,axis= 0)
    var_obs_pcp  = np.var(pcp_obs , axis = 0)
    std_obs_pcp  = np.std(pcp_obs , axis = 0)
    N = len(time)
    MSE_C = ((N-1)/(N))*var_obs_pcp

    diff_sq =(bias_pcp_fcst - pcp_obs)**2
    MSE_f  = np.average(diff_sq, axis =0)
    MSSS_bias = 1 -(MSE_f/MSE_C)
    return MSSS_bias




MSSS_bias= MSSS_bias(pcp_obs , bias_pcp_fcst)
SH = round(np.nanmean(MSSS_bias[44:72,:]), 4)
TRP = round(np.nanmean(MSSS_bias[28:44,:]),4)
NH  =round(np.nanmean(MSSS_bias[0:28,:]),4)
EU  = round(np.nanmean(MSSS_bias[0:28,0:72]),4)
IND =round(np.nanmean(MSSS_bias[20:36,22:44]),4)
PAC = round(np.nanmean(MSSS_bias[0:28,48:120]),4)
N34 =round(np.nanmean(MSSS_bias[33:39,68:128]),4)
GLB =round(np.nanmean(MSSS_bias), 4)










# Plotting SD_analysis



from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
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
plt.title("[OCT-DEC/MMCFSV2: GPCP_V2.3]\nRAINFALL (mm/day)\nMSSS with BC for 15 yrs(2003-2017)\nInitial : Mar, Lead time = 3 (Target month : OCT-DEC)\n",          fontsize = '14', loc ='left')

pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x0, 0.26, pos.width, 0.02])
cb =plt.colorbar(cp,cax = cbar_ax, orientation='horizontal',drawedges=True)
cb.dividers.set_color('black')
plt.figtext(pos.x0, 0.21, "[REGIONAL STATISTICS]", bbox = {'facecolor':'yellow'})

plt.figtext(pos.x0, 0.18, "NH")
plt.figtext(pos.x0+0.10, 0.18, "TRP")
plt.figtext(pos.x0+0.20, 0.18, "SH")
plt.figtext(pos.x0+0.30, 0.18, "EU")
plt.figtext(pos.x0+0.40, 0.18, "PAC")
plt.figtext(pos.x0+0.50, 0.18,"IND")
plt.figtext(pos.x0+0.60, 0.18, "N34")
plt.figtext(pos.x0+0.70, 0.18,"GLB")
#plt.figtext(pos.x0, 0.22, "1.25   2.4     6.1    0.5    3.15    2.00    10.25", ha="left", fontsize=12)
plt.figtext(pos.x0, 0.16, NH, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.10, 0.16, TRP, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.20, 0.16, SH, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.30, 0.16, EU, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.40, 0.16, PAC, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.50, 0.16, IND, bbox ={'facecolor':'None'} )
plt.figtext(pos.x0+0.60, 0.16, N34, bbox ={'facecolor':'None'})
plt.figtext(pos.x0+0.70, 0.16, GLB, bbox ={'facecolor':'None'} )



#rectangle = pac.Rectangle((0,1), 10,5, fc='None',ec="black")
#plt.gca().add_patch(rectangle)

fig1 = plt.gcf()
fig1.savefig('/data2/home/arvind/HC_Data/MAR/plot/ond/MSSS.png', dpi=100)
plt.legend()









