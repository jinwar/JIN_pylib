

from collections import OrderedDict
from datetime import timedelta
from glob import glob
import os
import sys
import numpy as np
from .Data2D_XT import merge_data2D
import pickle


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
        patch = self._reader(filename)
        self._cashe[filename] = patch
        # remove old data until cashe is smaller than limit
        while (self._estimate_cashe_size() > self._cashe_size_limit) \
            & (len(self._cashe)>1):
            self._cashe.popitem(last=False)
        return True
    
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
            p.select_time(bgtime,edtime)
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
            p = self._reader(file,bgtime,edtime)
            patch_list.append(p)

        merged_data = merge_data2D(patch_list)

        return merged_data

    def get_data(self,bgtime,edtime):
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


def sp_process(sp : spool, output_path, process_fun, pre_process=None, post_process=None,
               patch_size=1, overlap=0, save_file_size=200, 
               overwrite=True, **kargs):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if overwrite:
        files = glob(output_path+'/*.h5')
        for file in files:
            os.remove(file)
    
    time_segs = sp.get_time_segments()
    print('Found {} continuous datasets'.format(len(time_segs)))

    sp_output = []
    sp_size = 0
    for bgt, edt in sp.get_chunks(patch_size, overlap):
        if sp._check_data(bgt, edt):
            data = sp.get_data(bgt, edt)
            if pre_process is not None:
                data = pre_process(data)
            data = process_fun(data, **kargs)
            if post_process is not None:
                data = post_process(data)
            sp_output.append(data)
            sp_size += data.data.nbytes/1024**2
            if sp_size > save_file_size:
                _output_spool(sp_output,output_path)
                sp_output = []
                sp_size=0
        else:
            print('No data found in the time range: {} - {}'.format(bgt, edt))

    if len(sp_output)>0:
        _output_spool(sp_output,output_path)

    print('processing succeeded')
    return True


def _output_spool(sp_output, output_path):        
    patch_output = merge_data2D(sp_output)
    output_filename = os.path.join(output_path,_get_filename(patch_output))
    patch_output.saveh5(output_filename)
    return output_filename

def _get_filename(patch, time_string_length=19):
    bgstr = str(patch.start_time)[:time_string_length]
    edstr = str(patch.start_time + timedelta(seconds = patch.taxis[-1])])[:time_string_length]
    filename = bgstr+'_to_'+edstr+'.h5'
    filename = _clean_filename(filename)
    return filename

def _clean_filename(filename):
    # Remove illegal characters
    cleaned_filename = re.sub(r'[\/:*?"<>|]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    cleaned_filename = cleaned_filename.strip('. ').replace(' ', '_')
    
    return cleaned_filename