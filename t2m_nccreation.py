import os
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import netCDF4
import glob

MON = 'DEC'
path = '/data2/home/arvind/HC_Data/'+MON+'/T2M'
seas = 'jun'
for filename in sorted(glob.glob(os.path.join(path,'20*_T2m_ens.nc'))):
    print(filename)
  
    data = netCDF4.Dataset(filename,'r')
    rainfall = data.variables['t2m'][:,:,:,:]
    Data = np.average(rainfall, axis = 1)
    
    
    rf = Data[5,:,:]
    print(rf.shape)
    ncfile =netCDF4.Dataset(filename+seas,mode='w',format='NETCDF4_CLASSIC')
    lat_dim = ncfile.createDimension('lat',576)
    lon_dim = ncfile.createDimension('lon',1152)
    time_dim= ncfile.createDimension('time',1)
    ncfile.title ='ensm_mean_aug_2008'
    ncfile.subtitle ="INCOIS-NCMRWF Rainfall ensemble member"
    lat =ncfile.createVariable('lat', np.float32,('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = ncfile.createVariable('time',np.float32 , ('time',))
    time.standard_name ='time'
    time.units = 'hours since 1900-01-01 00:00:00'
    time.long_name = 'time'
    time.calendar = 'standard'
    t2m = ncfile.createVariable('t2m',np.float64,('time','lat','lon'))
    t2m.units = 'K'
    t2m.standard_name = 'ensm_aug_month_2008_rainfall'
    nlats = len(lat_dim); nlons = len(lon_dim) ; ntimes =1
    time = data.variables['time'][:]
    time._FillValue = '-2147483647'


    lon[:] = data.variables['lon'][:]
    lat[:] = data.variables['lat'][:]
    t2m[:,:,:] = rf

    print("-- Wrote data, precip.shape is now ", t2m.shape)
    print("-- Min/Max values:",t2m[:,:].min(), t2m[:,:].max())

    print(ncfile)
    ncfile.close()
    print('Dataset is closed!')
   

    
