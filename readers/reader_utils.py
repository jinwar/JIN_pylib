
from glob import glob
from tqdm import tqdm
from .. import Spool
import pandas as pd

def create_spool_common(datapath,get_time_range, reader
                        , search_pattern = '*.h5', support_partial_reading = False):
    files = glob(datapath+'/'+search_pattern)
    bgtimes = []
    edtimes = []
    final_files = []
    print('Indexing Files....')
    for file in tqdm(files):
        try:
            bgt,edt = get_time_range(file)
            bgtimes.append(bgt)
            edtimes.append(edt)
            final_files.append(file)
        except:
            print('Error reading file:',file)
            continue
    
    df = pd.DataFrame({'file':final_files,'start_time':bgtimes,'end_time':edtimes})

    sp = Spool.spool(df,reader, support_partial_reading=support_partial_reading)
    return sp