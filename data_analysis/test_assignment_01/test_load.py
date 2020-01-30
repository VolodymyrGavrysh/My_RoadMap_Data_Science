
import numpy as np
import pandas as pd
from dask_ml.preprocessing import StandardScaler
import gc
import time
import dask.dataframe as dask


# set our files
test = 'test.tsv'
train = 'train.tsv'


class LoadBigCsvFile:
    '''load data from tsv, transform and scale, add two columns
    Input .csv, .tsv files
    Output .tsv transformed file
    '''
    def __init__(self, train, test, scaler=StandardScaler(copy=False)):

        self.train = train
        self.test = test
        self.scaler = scaler

    def read_data(self):

        # use dask
        try:
            data_train = dask.read_csv(self.train, \
                                     dtype={n:'int16' for n in range(1, 300)}, engine='c').reset_index()
            data_test = dask.read_csv(self.test, \
                                    dtype={n:'int16' for n in range(1, 300)}, engine='c').reset_index()
        except: (IOError, OSError), 'can not open file'

        #if any data?
        assert len(data_test) != 0 and len(data_train) != 0, 'No data in files'

        # fit and transform
        self.scaler.fit(data_train.iloc[:,1:])
        del data_train # del not needed file
        test_transformed = self.scaler.transform(data_test.iloc[:,1:])

        # compute  values and add columns
        test_transformed['max_feature_2_abs_mean_diff'] = abs(test_transformed.mean(axis=1) - test_transformed.max(axis=1))
        test_transformed['max_feature_2_index'] = test_transformed.idxmin(axis=1)
        test_transformed['job_id'] = data_test.iloc[:,0]

        del data_test # del not needed file

        return test_transformed


if __name__ == "__main__":

    # init class
    start_time = time.time()
    data = LoadBigCsvFile(train, test).read_data()
    gc.collect()
    print('class loaded in %s seconds' % (time.time() - start_time))

    time.sleep(1) # set some time gap

    # save to hdf for later use or modification
    start_time = time.time()
    data.to_hdf('test_proc.hdf',  key='df1')
    print('file saved in hdf in %s seconds' % (time.time() - start_time))

    time.sleep(1) # set some time gap
    print()
    # check the file and its content
    start_time = time.time()
    hdf_read = dask.read_hdf('test_proc.hdf', key='df1', mode='r', chunksize=10000)
    print('file load into system in %s seconds' % (time.time() - start_time))
    print(hdf_read.head(3))
