import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

class PickFrac():

    def __init__(self,pickfile,fig):
        self.pickfile = pickfile
        self.fig = fig
        try:
            self.pickdf = pd.read_csv(pickfile)
        except:
            self.pickdf = pd.DataFrame(columns=['time','md'])
        self.lines = []
        self.cx = np.array([-1,1])
    
    def draw(self):
        plt.draw()
        self.cid = self.fig.canvas.mpl_connect('key_press_event',self.event_handle)
        self.update_lines()
    
    def update_lines(self):
        try:
            for l in self.lines:
                l.pop(0).remove()
        except:
            pass
        lines = []
        edtime = plt.gca().axis()[1]
        for i,row in self.pickdf.iterrows():
            bgtime = pd.to_datetime(row['time'])
            bgtime = mdates.date2num(bgtime)
            lines.append(plt.gca().plot([bgtime,edtime],[row['md'],row['md']],'k--'))
        self.lines = lines
        plt.draw()

    
    def event_handle(self,event):
        if event.key == 'a':
            self.pickdf.loc[len(self.pickdf),'time'] = mdates.num2date(event.xdata).strftime('%Y-%m-%d %H:%M:%S')
            self.pickdf.loc[len(self.pickdf)-1,'md'] = event.ydata
            self.update_lines()

        if event.key == 'd':
            mds = self.pickdf['md'].values.flatten()
            mindist = np.min(np.abs(mds-event.ydata))
            if mindist < 20:
                ind = np.argmin(np.abs(mds-event.ydata))
                self.pickdf = self.pickdf.drop(ind)
                self.pickdf.reset_index(drop=True,inplace=True)
            self.update_lines()

        if event.key == '=':
            self.cx = self.cx/1.2
            plt.clim(self.cx)
            plt.draw()

        if event.key == '-':
            self.cx = self.cx*1.2
            plt.clim(self.cx)
            plt.draw()
            
    def save(self):
        self.pickdf.to_csv(self.pickfile,index=False)