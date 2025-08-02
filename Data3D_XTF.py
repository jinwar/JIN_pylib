
import .Data2D_XT

class Data3D:

    def __init__(self):
        self.data = None   # data, 2D array
        self.start_time = None  # starting time using datetime
        self.taxis = []  # time axis in second from start_time
        self.chans = [] # fiber channel number
        self.daxis = []  # fiber physical distance or location
        self.faxis = []  # fiber frequency axis
        self.attrs = {'Python Class Version':'1.1'} # data attributes
        self.history = []


Data3D.save_h5 = Data2D_XT.Data2D.save_h5
Data3D.load_h5 = Data2D_XT.Data2D.load_h5