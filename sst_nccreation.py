import os
from netCDF4 import Dataset as netcdf_dataset
import numpy as np

#import xarray as xr
#ds = xr.open_mfdataset('/data2/home/arvind/HC_Data/FEB/Rainfall/20*.nc', parallel = True)
import glob
import netCDF4
MON = 'DEC'
path = '/data2/home/arvind/HC_Data/'+MON+'/SST'
seas = 'jun'
for filename in sorted(glob.glob(os.path.join(path,'20*_SST_ens.nc'))):
    print(filename)
  
    data = netCDF4.Dataset(filename,'r')
    sst = data.variables['sst'][:,:,:,:]
    print(sst.shape)
    
   
    Data = np.average(sst, axis = 1)
    print(Data.shape)
    
    
    sst1 = Data[5,:,:]
    print(sst1.shape)
    ncfile =netCDF4.Dataset(filename+seas,mode='w',format='NETCDF4_CLASSIC')
    lat_dim = ncfile.createDimension('lat',410)
    lon_dim = ncfile.createDimension('lon',720)
    time_dim= ncfile.createDimension('time',3)
    ncfile.title ='ensm_mean_aug_2008'
    ncfile.subtitle ="INCOIS-NCMRWF Rainfall ensemble member"
    # Creating a Variable

    lat =ncfile.createVariable('lat', np.float32,('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'

    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'

    time = ncfile.createVariable('time',np.float32 , ('time',))
    time.standard_name ='time'
    time.units = 'days since 1800-01-01 00:00:00'

    time.long_name = 'time'
    time.calendar = 'standard'



    sst = ncfile.createVariable('sst',np.float64,('time','lat','lon'))
    sst.units = 'K'
    sst.standard_name = 'ensm_monthly and seasonal sst'
    nlats = len(lat_dim); nlons = len(lon_dim) ; ntimes =3
    time = data.variables['time'][:]
#    time._FillValue = '-2147483647'


    lon[:] = data.variables['lon'][:]
    lat[:] = data.variables['lat'][:]


    #time[:] = dataset.variables['time'][4]
    sst[:,:,:] = sst1

    print("-- Wrote data, precip.shape is now ", sst.shape)
    print("-- Min/Max values:",sst[:,:].min(), sst[:,:].max())

    print(ncfile)
   # close the Dataset.
    ncfile.close();
    print('Dataset is closed!')
   

    
