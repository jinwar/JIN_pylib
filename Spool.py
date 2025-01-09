

from collections import OrderedDict
from datetime import timedelta
from glob import glob
import os
import re
import sys
import numpy as np
from tqdm.notebook import tqdm
from .Data2D_XT import merge_data2D
import pickle
from copy import deepcopy
import pandas as pd
from dateutil.parser import parse
from time import time


class spool:

    """
    spool class to handle dataset on storage. Mimicing spool structure in dascore.
    A pandas DataFrame and a callable reader function is required 
    to construct the class

    The DataFrame should have three columns:
        file: file name including relative path
        start_time: starting time of the file
        end_time: ending time of the file
    
    reader is the function to load data file into Patch
        if reader does not support partial reading:
        patch = reader(filename) 
        if reader supports partial reading:
        patch = reader(filename,start_time=None, end_time=None) 
    
    Example:
        sp = spool(df,reader,support_partial_reading=False)
    """

    def __init__(self,df = None,
                reader = None,
                support_partial_reading = False):
        if df is None:
            self._df = None
        else:
            self.set_database(df)
        self._reader = reader
        self._partial_reading = support_partial_reading
        self._cashe_size_limit = 1.0 # in GB
        self._cashe = OrderedDict({})
        self._debug = False
        pass

    def set_database(self,df):
        df = df.sort_values(by='start_time',ignore_index=True)
        self._df = df

    def set_reader(self,reader,support_partial_reading = False):
        """
        set reader function to load data file into Patch
            if reader does not support partial reading:
            patch = reader(filename) 
            if reader supports partial reading:
            patch = reader(filename,start_time=None, end_time=None) 
        Usage:
            sp.set_reader(reader,support_partial_reading = False):
        """
        self._reader = reader
        self._partial_reading = support_partial_reading
    
    def set_cashe_size(self,s):
        """
        Set cashe size limit (in GB) for spool.
        """
        self._cashe_size_limit = s
    
    def _estimate_cashe_size(self):
        s = 0.0
        for p in self._cashe.values():
            s += sys.getsizeof(p.data)
        s = s/1024**3  # convert to GB
        return s
    
    def _load_to_cashe(self,ind):
        filename = self._df['file'].iloc[ind]
        # check whether file is in cashe already
        # if exist, move the file to the end
        if filename in self._cashe.keys():
            self._cashe.move_to_end(filename)
            return False
        if self._debug:
            print(f'Loading file: {filename}')
        patch = self._reader(filename)
        self._cashe[filename] = patch
        # remove old data until cashe is smaller than limit
        while (self._estimate_cashe_size() > self._cashe_size_limit) \
            & (len(self._cashe)>1):
            self._cashe.popitem(last=False)
        return True
    
    def _check_inputtime(self,t,t0):
        out_t = t
        if t is None:
            out_t = t0
        if isinstance(t,str):
            out_t = pd.to_datetime(t)
        
        return out_t

    def select_time(self, bgtime=None, edtime=None):
        """
        select data in the time range
        """
        bgtime = self._check_inputtime(bgtime,self._df['start_time'].min())
        edtime = self._check_inputtime(edtime,self._df['end_time'].max())
        ind = np.where((self._df.start_time<edtime)
                 &(self._df.end_time>bgtime))[0]
        output = deepcopy(self)
        output.set_database(self._df.iloc[ind])
        if output._df['start_time'].iloc[0] < bgtime:
            output._df.iloc[0, output._df.columns.get_loc('start_time')] = bgtime
        if output._df['end_time'].iloc[-1] > edtime:
            output._df.iloc[-1, output._df.columns.get_loc('end_time')] = edtime
        return output
    
    def _get_data_nopl(self,bgtime,edtime):
        """
        function to load data without partial loading support
        """

        ind = np.where((self._df.start_time<edtime)
                 &(self._df.end_time>bgtime))[0]

        if len(ind) == 0:
            raise ValueError('No data found in the time range')

        patch_list = []
        for i in ind:
            self._load_to_cashe(i)
            p = self._cashe[self._df['file'].iloc[i]]
            p = p.select_time(bgtime,edtime,makecopy=True)
            patch_list.append(p)
        
        merged_data = merge_data2D(patch_list)

        return merged_data
    
    def _get_data_pl(self,bgtime,edtime):
        # not tested
        ind = np.where((self._df.start_time<edtime)
                 &(self._df.end_time>bgtime))[0]
        
        if len(ind) == 0:
            raise ValueError('No data found in the time range')

        patch_list = []
        for i in ind:
            file = self._df['file'].iloc[i]
            try:
                if self._debug:
                    print(f'Loading file: {file}')
                p = self._reader(file,bgtime,edtime)
                patch_list.append(p)
            except Exception as e:
                print(f'Error in reading file: {file}')
                print(f'Error: {e}')

        merged_data = merge_data2D(patch_list)

        return merged_data

    def get_data(self,bgtime=None,edtime=None):
        bgtime = self._check_inputtime(bgtime,self._df['start_time'].min())
        edtime = self._check_inputtime(edtime,self._df['end_time'].max())
        if self._debug:
            print(f'Loading data from {bgtime} to {edtime}')
            print(type(bgtime))
        if self._partial_reading:
            return self._get_data_pl(bgtime,edtime)
        else:
            return self._get_data_nopl(bgtime,edtime)

    def get_time_segments(self,max_dt=None):
        """
        Spool method to obtain continuous time segments in the spool.
        by checking the time differnce between start_timea and end_time
        in the database
        max_dt: maximum time difference tolerance for continuous data 
        """
        df = self._df

        dt = (df['start_time'].iloc[1:].values - df['end_time'].iloc[:-1].values)\
                /np.timedelta64(1,'s')

        if max_dt is None:
            max_dt = np.median(dt)*1.5
        ind = np.where(dt > max_dt)[0]
        ind = np.concatenate(([-1],ind,[len(df)-1]))

        time_segs = []

        for i in range(len(ind)-1):
            bgtime = df['start_time'].iloc[ind[i]+1]
            edtime = df['end_time'].iloc[ind[i+1]]
            time_segs.append((bgtime,edtime))
        
        return time_segs

    def _check_data(self, bgtime, edtime):
        ind = np.where((self._df.start_time<edtime)
                 &(self._df.end_time>bgtime))[0]
        if len(ind) == 0:
            return False
        else:
            return True
        

    def save_pickle(self,filename):
        """
        quick and dirty save method
        """
        with open(filename,'wb') as f:
            pickle.dump(self,f)
        print('saved the class to: ' + filename)

    def load_pickle(self,filename):
        """
        quick and dirty load method
        """
        # test if class data file exist
        with open(filename,'rb') as f:
            temp = pickle.load(f)
        self.__dict__.update(temp.__dict__)
    
    def get_chunks(self, length, overlap=0):
        """
        Get a chunk of data with given length and overlap
        """
        if overlap > length:
            raise ValueError('Overlap should be smaller than length')
        if overlap < 0:
            raise ValueError('Overlap should be positive')
        if length < 0:
            raise ValueError('Length should be positive')
        
        time_segs = self.get_time_segments()
        chunk_list = []
        length = timedelta(seconds= length)
        overlap = timedelta(seconds= overlap)
        for bgtime, edtime in time_segs:
            while bgtime < edtime:
                if edtime - bgtime < length:
                    chunk_list.append((bgtime,edtime))
                    break
                else:
                    chunk_list.append((bgtime,bgtime+length))
                    bgtime += length - overlap
        return chunk_list


    def _get_total_length(self):
        sp_df = self._df()
        sp_length = (sp_df['end_time'].max()-sp_df['start_time'].min()).total_seconds()
        return sp_length
    
    def __add__(self,sp2):
        """
        Concatenate two spool objects
        """
        if self._df is None:
            return sp2
        if sp2._df is None:
            return self
        df = pd.concat([self._df,sp2._df],ignore_index=True)
        sp = deepcopy(self)
        sp.set_database(df)
        return sp


def sp_process(sp : spool, output_path, process_fun, pre_process=None, post_process=None,
               patch_size=1, overlap=0, save_file_size=200, 
               overwrite=False, **kargs):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if overwrite:
        files = glob(output_path+'/*.h5')
        for file in files:
            os.remove(file)
    
    time_tracker = dict(dataio=0, pre_process=0, process=0, post_process=0, dataoutput=0)
    
    time_segs = sp.get_time_segments()
    print('Found {} continuous datasets'.format(len(time_segs)))

    sp_output = []
    sp_size = 0
    for bgt, edt in tqdm(sp.get_chunks(patch_size, overlap)):
        if sp._check_data(bgt, edt):
            try: 
                tic = time()
                data = sp.get_data(bgt, edt)
                time_tracker['dataio'] += time() - tic

                tic = time()
                if pre_process is not None:
                    data = pre_process(data)
                time_tracker['pre_process'] += time() - tic

                tic = time()
                data = process_fun(data, **kargs)
                time_tracker['process'] += time() - tic

                tic = time()
                if post_process is not None:
                    data = post_process(data)
                time_tracker['post_process'] += time() - tic

                sp_output.append(data)
                sp_size += data.data.nbytes/1024**2

                tic = time()
                if sp_size > save_file_size:
                    _output_spool(sp_output,output_path)
                    sp_output = []
                    sp_size=0
                time_tracker['dataoutput'] += time() - tic
            except Exception as e:
                print('Error in processing data: {} - {}'.format(bgt, edt))
                print('Error: {}'.format(e))
        else:
            print('No data found in the time range: {} - {}'.format(bgt, edt))

    if len(sp_output)>0:
        _output_spool(sp_output,output_path)

    print('processing succeeded')
    print('Time spent on data io: {:.2f} s'.format(time_tracker['dataio']))
    print('Time spent on pre-processing: {:.2f} s'.format(time_tracker['pre_process']))
    print('Time spent on processing: {:.2f} s'.format(time_tracker['process']))
    print('Time spent on post-processing: {:.2f} s'.format(time_tracker['post_process']))
    print('Time spent on data output: {:.2f} s'.format(time_tracker['dataoutput']))

    return True


def sp_process_pipeline(sp : spool, output_path, process_fun, pre_process=None, post_process=None,
               patch_size=1, overlap=0, save_file_size=200, 
               overwrite=False):
    
    import concurrent.futures

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if overwrite:
        files = glob(output_path+'/*.h5')
        for file in files:
            os.remove(file)
    
    time_tracker = dict(dataio=0, pre_process=0, process=0, post_process=0, dataoutput=0)
    
    time_segs = sp.get_time_segments()
    print('Found {} continuous datasets'.format(len(time_segs)))

    sp_output = []
    sp_size = 0

    def load_data(time_range):
        bgt, edt = time_range
        tic = time()
        data = sp.get_data(bgt, edt)
        time_tracker['dataio'] += time() - tic
        return data
    
    def _process_data(data):
        tic = time()
        if pre_process is not None:
            data = pre_process(data)
        time_tracker['pre_process'] += time() - tic

        tic = time()
        data = process_fun(data)
        time_tracker['process'] += time() - tic

        tic = time()
        if post_process is not None:
            data = post_process(data)
        time_tracker['post_process'] += time() - tic

        return data
    
    time_chunks = sp.get_chunks(patch_size, overlap)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as io_executor, \
        concurrent.futures.ThreadPoolExecutor() as process_executor:

        future_load = io_executor.submit(load_data, time_chunks[0])
        future_process = None

        for bgt, edt in tqdm(time_chunks[1:]):
            if sp._check_data(bgt, edt):
                # try: 
                data = future_load.result()

                future_process = process_executor.submit(_process_data, data)
                
                future_load = io_executor.submit(load_data, (bgt, edt))

                data = future_process.result()

                sp_output.append(data)
                sp_size += data.data.nbytes/1024**2

                tic = time()
                if sp_size > save_file_size:
                    _output_spool(sp_output,output_path)
                    sp_output = []
                    sp_size=0
                time_tracker['dataoutput'] += time() - tic
                # except Exception as e:
                #     print('Error in processing data: {} - {}'.format(bgt, edt))
                #     print('Error: {}'.format(e))
            else:
                print('No data found in the time range: {} - {}'.format(bgt, edt))

    if len(sp_output)>0:
        _output_spool(sp_output,output_path)

    print('processing succeeded')
    print('Time spent on data io: {:.2f} s'.format(time_tracker['dataio']))
    print('Paralleized processing:')
    print('Time spent on pre-processing: {:.2f} s'.format(time_tracker['pre_process']))
    print('Time spent on processing: {:.2f} s'.format(time_tracker['process']))
    print('Time spent on post-processing: {:.2f} s'.format(time_tracker['post_process']))
    print('Time spent on data output: {:.2f} s'.format(time_tracker['dataoutput']))

    return True


def load_pickle(filename):
    sp = spool()
    sp.load_pickle(filename)
    return sp

def _output_spool(sp_output, output_path):        
    patch_output = merge_data2D(sp_output)
    output_filename = os.path.join(output_path,_get_filename(patch_output))
    patch_output.saveh5(output_filename)
    return output_filename

def _get_filename(patch, time_string_length=19):
    bgstr = str(patch.start_time)[:time_string_length]
    edstr = str(patch.start_time + timedelta(seconds = patch.taxis[-1]))[:time_string_length]
    filename = bgstr+'_to_'+edstr+'.h5'
    filename = _clean_filename(filename)
    return filename

def _clean_filename(filename):
    # Remove illegal characters
    cleaned_filename = re.sub(r'[\/:*?"<>|]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    cleaned_filename = cleaned_filename.strip('. ').replace(' ', '_')
    
    return cleaned_filename


def std(DASdata, down_size=60):
    output_data = DASdata.copy()
    data = []
    N = len(DASdata.taxis)//down_size
    for i in range(down_size):
        data.append(np.std(DASdata.data[:,i*N:(i+1)*N], axis=1))
    output_data.data = np.array(data).T
    output_data.taxis = np.array([DASdata.taxis[i*N+N//2] for i in range(down_size)])
    return output_data