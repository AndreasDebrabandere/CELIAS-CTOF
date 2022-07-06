from numpy import array, arange

def dhms_in_days(str):
	
	a = str.split(":")
	                    
	days = float(a[0])
	hours = float(a[1])
	minutes = float(a[2])
	seconds = float(a[3])

	Days = days + (hours*3600.0 + minutes*60.0 + seconds *1.0)/(24.0*3600.0)
	return Days


def days_in_dhms(days):

	fulldays = int(days)	

	hours = (days-float(fulldays))*24.
	#print hours	
	fullhours = int(hours)
	
	minutes = (hours - float(fullhours))*60.	
	#print minutes	
	fullminutes = int(minutes)
		
	seconds = (minutes - float(fullminutes))*60.
	#print seconds	
	fullseconds = int(seconds)

	DHMS = "%i:%i:%i:%i" %(fulldays, fullhours, fullminutes, fullseconds)
	return DHMS  
	#print "The accuracy of the timevalue is +-1 second!"


def doytodaymonthstr(doy,year):
    monthl=[31,28,31,30,31,30,31,31,30,31,31]
    day=doy
    month=0
    if year%4==0:
        monthl[1]+=1
    while day-monthl[month]>0:
        day-=monthl[month]
        month+=1
    return "%.2i%.2i"%(month+1,day)

class pmdata(object):
    """
    Class to deal with SOHO/PM data.
    year -> year of data.  
    timeframe -> list of periods [[t1,t2],...,[tn-1,tn]]
    path -> should give the path to the pha data
    
    time,vel,dens,vth,dir -> contains time,velocity,density,thermal velocity, and direction of protons
    """
      
    #def __init__(self,timeframe,year=1996,path="/data/etph/soho/celias/pm/"):#old path
    #def __init__(self,timeframe,year=1996,path="/data/projects/soho/celias/pm/"):
    def __init__(self,timeframe,year=1996,path="/lhome/njanitzek/Projects/SOHO/data/mission_data/SOHO/CELIAS/PM/"):
        self.timeframe=timeframe
        self.year=year
        self.path=path
        self.time=[]
        self.vel=[]
        self.dens=[]
        self.vth=[]
        self.dir=[]
        self.load()
        self.conv_time()	

    def load(self):
        time=[]
        vel=[]
        dens=[]
        vth=[]
        dir=[]
        for tf in self.timeframe:
            for doy in range(int(tf[0]),int(tf[1])+1):

                infi=open("%s%.4i_%.3i_%s.pm"%(self.path,self.year,doy,doytodaymonthstr(doy,self.year)),"r")
                for i in range(29):
                    infi.readline()
                for s in infi:
                    k=s.split()
                    k2=k[3].split(":")
                    tmpt=float(k2[0])+float(k2[1])/24.+float(k2[2])/(24.*60.)+float(k2[3])/(24.*60.*60.)
                    if (tmpt>tf[0]) and (tmpt<tf[1]):
                        time.append(tmpt)
                        vel.append(float(k[4]))
                        dens.append(float(k[5]))
                        vth.append(float(k[6]))
                        dir.append(float(k[7]))
        self.time=array(time)
        self.vel=array(vel)
        self.dens=array(dens)
        self.vth=array(vth)
        self.dir=array(dir)
        print("pmdata loaded")
                                        
	
    def conv_time(self):
        self.time_dhms = []   	
        for t in self.time:
            T = days_in_dhms(t)
            self.time_dhms.append(T)	
        self.time_dhms = array(self.time_dhms)


    def check_timegaps(self):
        timegaps = []
        i=1 
        while i in arange(len(self.time)):
            if (self.time[i] - self.time[i-1]) > 0.0036:
                timegaps.append(i-1)
            i = i+1
        print("Output: 	pm.time-indices (timegap condition: more than (302+10) seconds per cycle")		
        return timegaps


			
	
			
