from numpy import loadtxt,savetxt,zeros,append,unique,amin,isnan,array,median,diff, round,ones,arange,histogram2d,clip,amax,searchsorted,max,abs,ravel,concatenate,percentile, split,insert,setdiff1d,where,invert,shape,histogram,sort,delete,log,around,meshgrid,ravel, transpose,sort,interp,invert,in1d,vstack,log,intersect1d,mean,sqrt,sum,average,float64,ndarray,histogramdd
from Libsoho.libctof import tof_to_mq,getionvel,convsecs70toDoY
from time import perf_counter as clock
from Libsoho._pmdata import pmdata
import pickle
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from CTOF_cal import tof,ESSD,tofsig,Esig

import numpy as np#take out later


#define phase space correction for CTOF
def ps_correction(N,v,v_ref):#counts as integer, v in  km/s
    N_cor=N*(v_ref/v)# 1D-correction for solar wind ions!
    return N_cor

#combined phase space and real space correction for solar wind ions!
def ps2_correction(N,v,v_ref):#counts as integer, v in  km/s
    N_cor=N*(v_ref**2/v**2)# 1D-correction for solar wind ions!
    return N_cor





def plot_PRhists(HET,HET_x,HET_y,HMR,HMR_x,HMR_y,figx=13.9,figy=7,fontsize=32,ticklabelsize=20,X1_lims=[100,600],Y1_lims=[1,100],X2_lims=[0,130],Y2_lims=[0,70],Wspace=0.0,save_figure=False,figpath="/home/asterix/janitzek/ctof/achapter2_finalplots/brcor/",figname="PR_definition"):	
	#HET,HMR are weighted histograms divided by unweighted histograms in ET and M-MpQ, respectively
	"""
	used example:
	plot_PRhists(HPR,HP[1],HP[2],HR,H[1],H[2],X1_lims=[190,610],Y1_lims=[1,101],X2_lims=[15,103],Y2_lims=[5,65],Wspace=0.3)
	"""
	
	fig,axs=plt.subplots(1,2,figsize=(figx,figy))
	fig.subplots_adjust(wspace=Wspace)
	ax1=axs[0]
	
	#ET plot
	p1=ax1.pcolor(HET_x,HET_y,HET.T,vmin=1,vmax=5)
	cb1 = fig.colorbar(p1,ax=ax1,pad=0.00,ticks=[1,2,3,4,5])	
	#cb1.set_label(r"$\rm{Priority \ Range}$",fontsize=fontsize)
	for ctick in cb1.ax.get_yticklabels():
		ctick.set_fontsize(ticklabelsize)

	
	ax1.set_xlabel(r"$\rm{TOF \ [ch]}$",fontsize=fontsize)		
	ax1.set_ylabel(r"$\rm{ESSD \ [ch]}$",fontsize=fontsize)
	ax1.tick_params(axis="x", labelsize=ticklabelsize)
	ax1.tick_params(axis="y", labelsize=ticklabelsize)

	ax1.set_xlim(X1_lims[0],X1_lims[1])
	ax1.set_ylim(Y1_lims[0],Y1_lims[1])

	#M-MPQ plot
	ax2=axs[1]
	p2=ax2.pcolor(HMR_x,HMR_y,HMR.T,vmin=1,vmax=5)
	cb2 = fig.colorbar(p2,ax=ax2,pad=0.00,ticks=[1,2,3,4,5])	
	cb2.set_label(r"$\rm{priority \ range}$",fontsize=fontsize)
	for ctick in cb2.ax.get_yticklabels():
		ctick.set_fontsize(ticklabelsize)

	ax2.set_xlabel(r"$\rm{mass-per-charge \ [ch]}$",fontsize=fontsize)		
	ax2.set_ylabel(r"$\rm{mass \ [ch]}$",fontsize=fontsize)
	ax2.tick_params(axis="x", labelsize=ticklabelsize)
	ax2.tick_params(axis="y", labelsize=ticklabelsize)

	ax2.set_xlim(X2_lims[0],X2_lims[1])
	ax2.set_ylim(Y2_lims[0],Y2_lims[1])

	if save_figure==True:
		plt.savefig(figpath+figname,bbox_inches='tight')
		




#plot functions (so thet one does not have to reload the whole class for replotting)
def plot_PHAMR_PR_multiPR(steps,counts_MR_allranges,counts_PHA_allranges,ESA_stop,steprange=[0,117],Hspace=0,Wspace=0.0,legendsize=15,ticks_fontsize=20):
	
	#subplot settings
	fig, axarr = plt.subplots(2,2, sharex=True)
	axs=ravel(axarr)
	fig.subplots_adjust(hspace=Hspace, wspace=Wspace)
	
	fig.suptitle("DOY 215 1996, cycle 133", fontsize=25)
		
	i=0
	while i<len(counts_MR_allranges):
		PR_counts_MR=counts_MR_allranges[i]
		PR_counts_PHA=counts_PHA_allranges[i]
		plot_heigth=1.4*max(PR_counts_MR)
		
		
		#plot histograms and lines
		axs[i].bar(steps-0.5,PR_counts_MR,width=1,color="r",label="MR",alpha=0.5)	
		axs[i].bar(steps-0.5,PR_counts_PHA,width=1,color="b",label="PHA",alpha=0.5)
		axs[i].plot([ESA_stop,ESA_stop],[0,plot_heigth],linewidth=3.0,color="r",label="ESA stop")
		
		#axis labels
		#axs[i].set_xlabel(r"$E/q \ step$",fontsize=30)#latex style
		#axs[i].set_ylabel(r"$Counts\ per\ bin$",fontsize=30)##latex style
		if i>1:
			axs[i].set_xlabel("E/q step",fontsize=20)
		if i==0 or i==2: 
			axs[i].set_ylabel("Counts per bin",fontsize=20)
		#axs[i].set_title("Priority range %i"%(i+1),fontsize=20)

		#legend
		legend=axs[i].legend(loc="upper center",prop={'size':legendsize},title="Priority range %i"%(i+1),ncol=3)
		plt.setp(legend.get_title(),fontsize=legendsize)

		#set ticks and grid
		mxts=arange(0,116,5)
		MXTs=arange(0,116,10)
		myts=[arange(0,plot_heigth,5),arange(0,plot_heigth,5),arange(0,plot_heigth,25),arange(0,plot_heigth,200)]
		MYTs=[arange(0,plot_heigth,10),arange(0,plot_heigth,10),arange(0,plot_heigth,50),arange(0,plot_heigth,400)]
	
		major_xticks=MXTs
		minor_xticks=mxts
		major_yticks=MYTs[i]
		minor_yticks=myts[i]

		axs[i].set_xticks(major_xticks)                                                       
		axs[i].set_xticks(minor_xticks, minor=True)                                           
		axs[i].set_yticks(major_yticks)                                                       
		axs[i].set_yticks(minor_yticks, minor=True) 
		axs[i].grid(which='both') 

		axs[i].tick_params(labelsize=ticks_fontsize)	

		if i==1 or i==3:
			axs[i].yaxis.set_ticks_position("right")

		#plot edges
		axs[i].axis([steprange[0],steprange[1],0,plot_heigth])

		i=i+1		




def plot_boxrates_spectrum(steps,steps_MR_alltimes,steps_diff_MR_alltimes,counts_MR_alltimes,counts_PHA_alltimes,csteps_alltimes,steprange=[0,117],Hspace=0,legendsize=20,ticks_fontsize=20):

	#subplot settings
	fig, axarr = plt.subplots(2,1, sharex=True)
	axs=ravel(axarr)
	fig.subplots_adjust(hspace=Hspace, wspace=0)
	fig.suptitle("m-m/q box 201 (Si7+)",fontsize=25)
	
	i=0
	while i<len(counts_MR_alltimes):
		
		BR_counts_PHA=counts_PHA_alltimes[i]
		mr_steps=steps_MR_alltimes[i]
		mr_steps_diff=steps_diff_MR_alltimes[i]
		BR_counts_MR=counts_MR_alltimes[i]/mr_steps_diff
		plot_heigth=1.3*max(BR_counts_MR)
		
		#plot
		#axs[i].bar(steps-0.5,BR_counts_MR,width=1,color="r",label="MR counts",alpha=0.5)#equal mr binsize	
		axs[i].bar(mr_steps-0.5,BR_counts_MR, width=mr_steps_diff, color="r", alpha=0.5, linewidth=2,label="MR")
		axs[i].bar(steps-0.5,BR_counts_PHA,width=1,color="b",label="PHA",alpha=0.5)
		axs[i].plot([csteps_alltimes[i],csteps_alltimes[i]],[0, plot_heigth],linewidth=3.0,color="r",label="central step")
		
		#legend
		if i==0:
			legend=axs[i].legend(loc="upper left",prop={'size':legendsize},title="DOY 182, cycle 5:")
		else:
			legend=axs[i].legend(loc="upper right",prop={'size':legendsize},title="DOY 215, cycle 133:")
		plt.setp(legend.get_title(),fontsize=legendsize)
		
		
		
		
		#axis labels
		#axs[i].set_xlabel(r"$E/q \ step$",fontsize=30)#latex style
		#axs[i].set_ylabel(r"$Counts\ per\ bin$",fontsize=30)##latex style
		
		axs[i].set_xlabel("E/q step",fontsize=20)
		axs[i].set_ylabel("Counts per bin",fontsize=20)
		#if i==0:
		#	axs[i].set_title("m-m/q box 201 (Si7+)",fontsize=25)
		
		#set ticks and grid
		mxts=arange(0,116,5)
		MXTs=arange(0,116,10)
		myts=[arange(0,plot_heigth,1),arange(0,plot_heigth,1)]
		MYTs=[arange(0,plot_heigth,5),arange(0,plot_heigth,5)]
	
		major_xticks=MXTs
		minor_xticks=mxts
		major_yticks=MYTs[i]
		minor_yticks=myts[i]

		axs[i].set_xticks(major_xticks)                                                       
		axs[i].set_xticks(minor_xticks, minor=True)                                           
		axs[i].set_yticks(major_yticks)                                                       
		axs[i].set_yticks(minor_yticks, minor=True)
		axs[i].grid(which='major') 

		axs[i].tick_params(labelsize=ticks_fontsize)

		
		
		#plot edges
		axs[i].axis([steprange[0],steprange[1],0,plot_heigth])
		
		
		i=i+1		
	


def get_stepgrid(mr_steps,mr_counts):
	steps=arange(0,117)
	counts=zeros((len(steps)))
	cmask=mr_counts>0
	
	mr_steps_masked=mr_steps[cmask]
	mr_counts_masked=mr_counts[cmask]
	inds=searchsorted(steps,mr_steps_masked)
	counts[inds]=mr_counts_masked
	return steps,counts 


#conversion form ESA-steps to ion velocities		
def step_to_vel(step,q_e,m_amu):
    U_0=0.331095#in kV	
    r=1.040926
    q=q_e*1.6*10**-19
    m=m_amu*1.67*10**-27	
    v=sqrt(2*q*U_0*r**(116-step)*1000/m)/1000.
    return v#in km/s

def vel_to_step(m,q,v):#in u,e,km/s
	U_0=0.331095e3#in V
	r=1.040926#dimensionless
	s_max=115
	Q=q*1.602e-19
	M=m*1.66e-27
	V=v*1000.
	step=s_max-log(M*V**2/(2*Q*U_0))/log(r)
	return step

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def dt_to_posix(dt):
    """
    Returns UTC timestamp from UTC datetime object.

    Parameters
    -----------
    p_time : float or array like
        long int, list or array of POSIX timestamps

    Returns
    --------
    data : int or numpy array
        posix int or array of posix int
    """
    from calendar import timegm
    from numpy import array, ndarray
    if type(dt) in [list, ndarray]:
        posix = array([timegm(t.timetuple()) for t in array(dt)])
    else:
        posix = timegm(dt.timetuple())
    return posix

def posix_to_dt(posix):
    """
    returns a datetime object from given posix time

    Parameters
    -----------
    posix : float or array like
        A single posix time or a list/array of posix times

    Returns
    --------
    data : float or array like
        A single datetime object or a np array of datetime objects
    """
    from datetime import datetime
    from numpy import array, ndarray
    if type(posix) in [list, ndarray]:
        time = array(posix)
        dt = [datetime.utcfromtimestamp(t) for t in time]
    else:
        dt = datetime.utcfromtimestamp(posix)
    return dt

def LOBT_to_DOY(LOBT):
	"""
	converts DOY from SOHO LOBT, starting on 1970-01-01 at 00.00.00 to DOY 1996.
	Not completed days are given in decimal places. Accuracy is 1 second.  
	"""	
	DOY=posix_to_dt(LOBT).timetuple().tm_yday
	#print DOY 
	hour=posix_to_dt(LOBT).hour
	minute=posix_to_dt(LOBT).minute
	second=posix_to_dt(LOBT).day
	day_dec=(hour*3600+minute*60+second)/(24*3600.)
	#print day_dec 
	DOY_dec=DOY+day_dec
	return DOY_dec


def LOBT_to_DOY_array(LOBT_array):
    DOY_array=zeros((len(LOBT_array)))
    i=0
    while i<len(LOBT_array):
        DOY=LOBT_to_DOY(LOBT_array[i])
        DOY_array[i]=DOY
        i=i+1
    return DOY_array


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


def min_in_days(minutes):
	days=minutes/(24*60.)
	return days	

class ctoflv1(object):
		
	#def __init__(self,timeframe,minute_frame=[0,1440],year=1996,path="/data/auto/ivar/berger/ctof/lv1/",loadlv="lv1"):
	#def __init__(self,timeframe,minute_frame=[0,1440],year=1996,path="/home/auto/hatschi/janitzek/mission_data/SOHO/CELIAS/CTOF/lv1/",loadlv="lv1"):
    #def __init__(self,timeframe,minute_frame=[0,1440],year=1996,path="/home/hatschi/janitzek/mission_data/SOHO/CELIAS/CTOF/lv1/",loadlv="lv1"):
	def __init__(self,timeframe,minute_frame=[0,1440],year=1996,path="/lhome/njanitzek/Projects/SOHO/data/mission_data/SOHO/CELIAS/CTOF/lv1/",loadlv="lv1"):

		"""
		This class is ment to deal with CTOF PHA data. 
		year -> year of data (mind that CTOF only functioned properly from DoY 80 to 230 in 1996) 
		timeframe -> list of periods [[t1,t2],...,[tn-1,tn]]
		path -> should give the path to the pha data

		times,secs,tof,energy,range,step -> contains the information for each individual PHA word.
		time -> contains a list of the starting times of the instrumental cycles.Each cycle (a complete ESA sweep) is about 300s. 
		"""
		start = clock()      
		self.year=year
		self.timeframe=timeframe
		self.minute_frame=minute_frame        
		self.path=path
		self.times=zeros((0))
		self.secs=zeros((0))
		self.range=zeros((0))
		self.energy=zeros((0))
		self.tof=zeros((0))
		self.step=zeros((0))
		self.vel=zeros((0))
		self.vsw=zeros((0))
		self.dsw=zeros((0))
		self.tsw=zeros((0))
		self.mpq=zeros((0))
		self.w=zeros((0))
		self.vsw=zeros((0))
		self.vth=zeros((0))	
		self.vsw_safe = zeros((0))	
		self.vsw_sorted = zeros((0))
		self.vth_sorted = zeros((0))        
		self.dsw=zeros((0))
		self.dsw_sorted = zeros((0))        
		self.w=zeros((0))
		self.br=zeros((0))
		self.ratestep=zeros((0))
		self.ratetcr=zeros((0))
		self.ratedcr=zeros((0))
		self.ratessr=zeros((0))
		self.ratefsr=zeros((0))
		self.rateher=zeros((0))
		self.ratehpr=zeros((0))
		self.ratesecs=zeros((0))
		self.time_indices = zeros((0))
		self.time_indices_shifted = zeros((0))	
		self.time_dhms = zeros((0))	
		self.time_dhmsu = zeros((0))	
		self.pmtime =  zeros((0))
		self.pmtime_dhms = zeros((0))
		self.pmtime_shifted = zeros((0))
		self.pmtime_shifted_dhms = zeros((0))


		if loadlv=="lv1":
			self.loadlv11()
			#self.load_CTOFfull()

			#self.calc_mpq()
			#self.calc_ionvel()
			#self.get_vsw_quick()
			#self.calc_baserates()
			pass	
		"""
		elif loadlv=="lv1br":
			self.loadlv1br()
			self.calc_ionvel()
		end = clock()	
		#self.load_rates()
		"""	
	
		start_quick = clock()		
		self.get_vsw_quick()
		end_quick = clock()
		self._add_mmpq()
      

	def loadlv1(self):
		for tf in self.timeframe:
			for day in range(int(tf[0]),int(tf[1]+1.)):
				tmpdat=loadtxt(self.path+"cph"+str(self.year%100)+"%.3i.dat"%(day),skiprows=3)
				self.times=append(self.times,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,0])
				self.secs=append(self.secs,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,1])
				self.range=append(self.range,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,2])
				self.energy=append(self.energy,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,3])
				self.tof=append(self.tof,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,4])
				self.step=append(self.step,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,5])
		self.time=unique(self.times)
		print("pha data loaded")
		return True

	def loadlv1br(self):
		for tf in self.timeframe:
			for day in range(int(tf[0]),int(tf[1]+1.)):
				tmpdat=loadtxt(self.path+"cphbr"+str(self.year%100)+"%.3i.dat"%(day),skiprows=3)
				self.times=append(self.times,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,0])
				self.secs=append(self.secs,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,1])
				self.range=append(self.range,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,2])
				self.energy=append(self.energy,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,3])
				self.tof=append(self.tof,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,4])
				self.step=append(self.step,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,5])
				self.vsw=append(self.vsw,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,6])
				self.dsw=append(self.dsw,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,7])
				self.tsw=append(self.tsw,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,8])
				self.mpq=append(self.mpq,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,9])
				self.w=append(self.w,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,10])
				self.br=append(self.br,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,11])
		self.time=unique(self.times)
		print("phabr data loaded")
		return True
        
        
        
        
	def loadlv11(self):# allows to cut of minute slices of data
		t0=clock()
		mf=array(self.minute_frame)
		mfd=min_in_days(mf)  	
		print("selected minuteframe in days")
		print(mfd)
		for tf in self.timeframe:
			for day in range(int(tf[0]),int(tf[1]+1.)):
				tmpdat=loadtxt(self.path+"cph"+str(self.year%100)+"%.3i.dat"%(day),skiprows=3)
				self.times=append(self.times,tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,0])
				self.secs=append(self.secs,tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,1])
				self.range=append(self.range,tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,2])
				self.energy=append(self.energy,tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,3])
				self.tof=append(self.tof,tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,4])
				self.step=append(self.step,tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,5])
				self.time=unique(self.times)
		t1=clock()
		print("pha data loaded")
		print("pha loading time: %.5f"%(t1-t0))  
		return self.times,self.secs,self.range,self.energy,self.tof,self.step#has to be changed when _ctovlv1 is used in the backgound!
		#return True

		
		
	def load_CTOFfull(self):
		"""
		Loads CTOF PHA and CRA (and maybe other data files) and synchronizes these data.
		"""
		t0=clock()
		#print t0
		self.times=[]
		self.secs=[]
		self.range=[]
		self.energy=[]
		self.tof=[]
		self.step=[]
		
		mf=array(self.minute_frame)
		mfd=min_in_days(mf)  	
		print("selected minuteframe in days")
		print(mfd)
		for tf in self.timeframe:
			for day in range(int(tf[0]),int(tf[1]+1.)):
				print(day)
				tmpdat=loadtxt(self.path+"cph"+str(self.year%100)+"%.3i.dat"%(day),skiprows=3)
				if len(tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,0])>0:
					self.times.append(tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,0])
					self.secs.append(tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,1])
					self.range.append(tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,2])
					self.energy.append(tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,3])
					self.tof.append(tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,4])
					self.step.append(tmpdat[(tmpdat[:,0]>(tf[0]+mfd[0]))*(tmpdat[:,0]<(tf[1]-1+mfd[1]))][:,5])
		    
		
		self.times=concatenate(self.times)
		self.secs=concatenate(self.secs)
		self.range=concatenate(self.range)
		self.energy=concatenate(self.energy)
		self.tof=concatenate(self.tof)
		self.step=concatenate(self.step)
		
		self.time=unique(self.times)
		t1=clock()
		return t1,t0
		
		print("pha data loaded")
		print("pha loading time: %.5f"%(t1-t0))
		return True
	

	def load_CTOFcra(self,path="/home/asterix/janitzek/ctof/reconstruct_PHA/"):
		path="/data/missions/soho/celias/lv1/ctof/cra/1996/"
		
		"""
		Loads CTOF PHA and CRA (and maybe other data files) and synchronizes these data.
		"""
		
		t0=clock()
		#print t0
		#tmpdat=secs_cra=loadtxt(path+"c"+str(self.year%100)+"%.3i.cra"%(day),skiprows=13)
		
		

		
		
		self.times_cra=[]
		self.secs_cra=[]
		self.step_cra=[]  
		self.vvps=[]
		self.halt=[]
		self.speed=[]
		self.fsr=[]
		self.dcr=[]
		self.tcr=[]
		self.ssr=[]
		self.hpr=[]
		self.her=[]

		mf=array(self.minute_frame)
		mfd=min_in_days(mf)     
		print("selected minuteframe in days")
		print(mfd)
		for tf in self.timeframe:
			for day in range(int(tf[0]),int(tf[1]+1.)):
				print(day)
				try:
					tmpdat=loadtxt(path+"c"+str(self.year%100)+"%.3i.cra"%(day),skiprows=13)
					
					#return tmpdat
					time_LOBT=tmpdat.T[0]        
					utime_LOBT=unique(time_LOBT)
					utime_DOY=LOBT_to_DOY_array(utime_LOBT)
					#return utime_DOY
					time_indices=searchsorted(utime_LOBT,time_LOBT)
					time_DOY=utime_DOY[time_indices]
					#return time_DOY
		    
					if len(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,0])>0:
						self.times_cra.append(time_DOY[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))])
						self.secs_cra.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,0])
						self.step_cra.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,1])
						self.vvps.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,2])
						self.halt.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,3])
						self.speed.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,4])
						self.fsr.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,5])
						self.dcr.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,6])
						self.tcr.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,7])
						self.ssr.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,8])
						self.hpr.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,9])
						self.her.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,10])
				except IOError:
					print("day %i does not exist within cra data!"%(day))
		       
		self.times_cra=concatenate(self.times_cra)
		self.secs_cra=concatenate(self.secs_cra)
		self.step_cra=concatenate(self.step_cra)
		self.vvps=concatenate(self.vvps)
		self.halt=concatenate(self.halt)
		self.speed=concatenate(self.speed)
		self.fsr=concatenate(self.fsr)
		self.dcr=concatenate(self.dcr)
		self.tcr=concatenate(self.tcr)
		self.ssr=concatenate(self.ssr)
		self.hpr=concatenate(self.hpr)
		self.her=concatenate(self.her)
		    
		#self.time_cra=unique(self.times_cra)
		t1=clock()
		       
		print("cra data loaded")
		print("cra loading time: %.5f"%(t1-t0)) 
		       
		return True    

		
	def load_CTOFMR(self,path="/home/asterix/janitzek/ctof/reconstruct_PHA/"):
		path="/data/missions/soho/celias/lv1/ctof/cmr/1996/"        
			
		t0=clock()
		
		self.times_mr=[]
		self.secs_mr=[]
		self.mr_number=[]
		self.vsw_mr=[]
		self.mr0=[]
		self.mr1=[]
		self.mr2=[]
		self.mr3=[]
		self.mr4=[]
		self.mr5=[]
		self.mr6=[]
		self.mr7=[]
		self.mr8=[]
		self.mr9=[]
		self.mr10=[]
		self.mr11=[]
		self.mr12=[]
		self.mr13=[]        
		self.mr14=[]
		self.mr15=[]
		self.mr16=[]
		self.mr17=[]
		self.mr18=[]
		self.mr19=[]
		self.mr20=[]
		
		mf=array(self.minute_frame)
		mfd=min_in_days(mf)     
		print("selected minuteframe in days")
		print(mfd)
		
		
		
		
		#tf0=array([self.timeframe[0][0],self.timeframe[0][1]])
		tf1=arange(self.timeframe[0][0],self.timeframe[0][1],10)
		tf2=concatenate([tf1,tf1,array([self.timeframe[0][1]])])
		tf3=tf2[1:]
		tf4=sort(tf3)
		ndiv=int(((self.timeframe[0][1]-1)-self.timeframe[0][0])/10)+1
		#return ndiv,tf4
		timeframe=split(tf4,ndiv)
		#return tf4,ndiv,timeframe
		
		
		
		
		for tf in timeframe:
			
			self.times_mr_tf=[]
			self.secs_mr_tf=[]
			self.mr_number_tf=[]
			self.vsw_mr_tf=[]
			self.vth_mr_tf=[]
			
			self.mr0_tf=[]
			self.mr1_tf=[]
			self.mr2_tf=[]
			self.mr3_tf=[]
			self.mr4_tf=[]
			self.mr5_tf=[]
			self.mr6_tf=[]
			self.mr7_tf=[]
			self.mr8_tf=[]
			self.mr9_tf=[]
			self.mr10_tf=[]
			self.mr11_tf=[]
			self.mr12_tf=[]
			self.mr13_tf=[]        
			self.mr14_tf=[]
			self.mr15_tf=[]
			self.mr16_tf=[]
			self.mr17_tf=[]
			self.mr18_tf=[]
			self.mr19_tf=[]
			self.mr20_tf=[]
			
			
			
			
			for day in range(int(tf[0]),int(tf[1]+1.)):
				print(day)
				
				#return loadtxt(path+"c"+str(self.year%100)+"%.3i.cmr"%(day),skiprows=13)
				
				try:
					tmpdat=loadtxt(path+"c"+str(self.year%100)+"%.3i.cmr"%(day),skiprows=13)
					
					#convert LOBT time to DOY
					time_LOBT=tmpdat.T[0]
					
					if day>tf[0]:
						mask_valid=invert(in1d(time_LOBT,concatenate(self.secs_mr_tf)))#sort out time stamps that occurred twice in subsequent days
					else:
						mask_valid=array([True]*len(time_LOBT))
					tmpdat=tmpdat[mask_valid]
					time_DOY=LOBT_to_DOY_array(time_LOBT)[[mask_valid]]
					
					
					time_LOBT_valid=time_LOBT[mask_valid]	
					utime_LOBT_valid=unique(time_LOBT_valid)
					
					"""
					utime_LOBT=unique(time_LOBT)
					utime_DOY=LOBT_to_DOY_array(utime_LOBT)
					time_indices=searchsorted(utime_LOBT_valid,time_LOBT)
					time_DOY=utime_DOY[time_indices]
							"""
			
					#return tmpdat[:,0]
			
					if len(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,0])>0:
						self.times_mr_tf.append(time_DOY[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))])
						self.secs_mr_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,0])
						self.mr_number_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,1])
						self.vsw_mr_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,2])
						#self.vth_mr_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,3])
						#return 
						#print "test vth", self.vth_mr_tf, min(self.vth_mr_tf[0]),max(self.vth_mr_tf[0]),len(self.vth_mr_tf[0])                        
						
						self.mr0_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,3])
						self.mr1_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,4])
						self.mr2_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,5])
						self.mr3_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,6])
						self.mr4_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,7])
						self.mr5_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,8])
						self.mr6_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,9])
						self.mr7_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,10])
						self.mr8_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,11])
						self.mr9_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,12])
						self.mr10_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,13])
						self.mr11_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,14])
						self.mr12_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,15])
						self.mr13_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,16])
						self.mr14_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,17])
						self.mr15_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,18])
						self.mr16_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,19])
						self.mr17_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,20])
						self.mr18_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,21])
						self.mr19_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,22])
						self.mr20_tf.append(tmpdat[(time_DOY>(tf[0]+mfd[0]))*(time_DOY<(tf[1]-1+mfd[1]))][:,23])
				except IOError:
					print("day %i does not exist within cmr data!"%(day))

			self.times_mr_tf=concatenate(self.times_mr_tf)
			self.secs_mr_tf=concatenate(self.secs_mr_tf)
			self.mr_number_tf=concatenate(self.mr_number_tf)       
			self.vsw_mr_tf=concatenate(self.vsw_mr_tf)
			self.mr0_tf=concatenate(self.mr0_tf)        
			self.mr1_tf=concatenate(self.mr1_tf)
			self.mr2_tf=concatenate(self.mr2_tf)
			self.mr3_tf=concatenate(self.mr3_tf)
			self.mr4_tf=concatenate(self.mr4_tf)        
			self.mr5_tf=concatenate(self.mr5_tf)
			self.mr6_tf=concatenate(self.mr6_tf)
			self.mr7_tf=concatenate(self.mr7_tf)
			self.mr8_tf=concatenate(self.mr8_tf)        
			self.mr9_tf=concatenate(self.mr9_tf)
			self.mr10_tf=concatenate(self.mr10_tf)
			self.mr11_tf=concatenate(self.mr11_tf)
			self.mr12_tf=concatenate(self.mr12_tf)        
			self.mr13_tf=concatenate(self.mr13_tf)
			self.mr14_tf=concatenate(self.mr14_tf)
			self.mr15_tf=concatenate(self.mr15_tf)
			self.mr16_tf=concatenate(self.mr16_tf)        
			self.mr17_tf=concatenate(self.mr17_tf)
			self.mr18_tf=concatenate(self.mr18_tf)
			self.mr19_tf=concatenate(self.mr19_tf)
			self.mr20_tf=concatenate(self.mr20_tf)        
		
			self.times_mr.append(self.times_mr_tf)
			self.secs_mr.append(self.secs_mr_tf)
			self.mr_number.append(self.mr_number_tf)
			self.vsw_mr.append(self.vsw_mr_tf)
			self.mr0.append(self.mr0_tf)
			self.mr1.append(self.mr1_tf)
			self.mr2.append(self.mr2_tf)
			self.mr3.append(self.mr3_tf)
			self.mr4.append(self.mr4_tf)
			self.mr5.append(self.mr5_tf)
			self.mr6.append(self.mr6_tf)
			self.mr7.append(self.mr7_tf)
			self.mr8.append(self.mr8_tf)
			self.mr9.append(self.mr9_tf)
			self.mr10.append(self.mr10_tf)
			self.mr11.append(self.mr11_tf)
			self.mr12.append(self.mr12_tf)
			self.mr13.append(self.mr13_tf)
			self.mr14.append(self.mr14_tf)
			self.mr15.append(self.mr15_tf)
			self.mr16.append(self.mr16_tf)
			self.mr17.append(self.mr17_tf)
			self.mr18.append(self.mr18_tf)
			self.mr19.append(self.mr19_tf)
			self.mr20.append(self.mr20_tf)
		
		self.times_mr=concatenate(self.times_mr)
		self.secs_mr=concatenate(self.secs_mr)
		self.mr_number=concatenate(self.mr_number)       
		self.vsw_mr=concatenate(self.vsw_mr)
		self.mr0=concatenate(self.mr0)        
		self.mr1=concatenate(self.mr1)
		self.mr2=concatenate(self.mr2)
		self.mr3=concatenate(self.mr3)
		self.mr4=concatenate(self.mr4)        
		self.mr5=concatenate(self.mr5)
		self.mr6=concatenate(self.mr6)
		self.mr7=concatenate(self.mr7)
		self.mr8=concatenate(self.mr8)        
		self.mr9=concatenate(self.mr9)
		self.mr10=concatenate(self.mr10)
		self.mr11=concatenate(self.mr11)
		self.mr12=concatenate(self.mr12)        
		self.mr13=concatenate(self.mr13)
		self.mr14=concatenate(self.mr14)
		self.mr15=concatenate(self.mr15)
		self.mr16=concatenate(self.mr16)        
		self.mr17=concatenate(self.mr17)
		self.mr18=concatenate(self.mr18)
		self.mr19=concatenate(self.mr19)
		self.mr20=concatenate(self.mr20)        
		
		t1=clock()
		       
		print("cmr data loaded")
		print("cmr loading time: %.5f"%(t1-t0))
		uT,iT = unique(self.secs_mr,return_inverse=True)
		
		MR = zeros((uT.shape[0],21,508))
		MR[:,0,:]=self.mr0.reshape(uT.shape[0],508)
		MR[:,1,:]=self.mr1.reshape(uT.shape[0],508)
		MR[:,2,:]=self.mr2.reshape(uT.shape[0],508)
		MR[:,3,:]=self.mr3.reshape(uT.shape[0],508)
		MR[:,4,:]=self.mr4.reshape(uT.shape[0],508)
		MR[:,5,:]=self.mr5.reshape(uT.shape[0],508)
		MR[:,6,:]=self.mr6.reshape(uT.shape[0],508)
		MR[:,7,:]=self.mr7.reshape(uT.shape[0],508)
		MR[:,8,:]=self.mr8.reshape(uT.shape[0],508)
		MR[:,9,:]=self.mr9.reshape(uT.shape[0],508)
		MR[:,10,:]=self.mr10.reshape(uT.shape[0],508)
		MR[:,11,:]=self.mr11.reshape(uT.shape[0],508)
		MR[:,12,:]=self.mr12.reshape(uT.shape[0],508)
		MR[:,13,:]=self.mr13.reshape(uT.shape[0],508)
		MR[:,14,:]=self.mr14.reshape(uT.shape[0],508)
		MR[:,15,:]=self.mr15.reshape(uT.shape[0],508)
		MR[:,16,:]=self.mr16.reshape(uT.shape[0],508)
		MR[:,17,:]=self.mr17.reshape(uT.shape[0],508)
		MR[:,18,:]=self.mr18.reshape(uT.shape[0],508)
		MR[:,19,:]=self.mr19.reshape(uT.shape[0],508)
		MR[:,20,:]=self.mr20.reshape(uT.shape[0],508)
		
		self.uT_mr=uT
		self.iT_mr=iT		       
		self.MR=MR		       
		#self.usecs_mr=unique(self.secs_mr)
				       
		return True           
	
	def unpack_MR(self):
		pass
		
	
	
		
		
	def load_matrix_box_definitions(self,path='/home/asterix/janitzek/ctof/CTOF_matrix_boxes.txt'):
		'''
		docstring!
		'''
		
		mr_boxdef=loadtxt(path,skiprows=6).T
		self.mrbox_number=mr_boxdef[0].astype("int")
		self.mrbox_mpc_ll=mr_boxdef[1]#lower_left box corner mass channel
		self.mrbox_m_ll=mr_boxdef[2]#lower_left box corner mass-per-charge channel
		self.mrbox_mpc_ur=mr_boxdef[3]#upper_right box corner mass channel
		self.mrbox_m_ur=mr_boxdef[4]#upper_right box corner mass-per-charge channel
		
		return True
		

	def load_matrix_shifts(self,path='/home/asterix/janitzek/ctof/CTOF_matrix_shifts.txt'):
		'''
		Method loads and initializes self.shift that is used in 'calc_mr_centerstep' to calculate the right center step for each matrix box.
		'''
		mr_shiftdata=loadtxt(path,skiprows=6).T
		#self.mrbox_number=mr_shiftdata[0]#same box  numbers as in 'load_matrix_box_definitions'
		self.mrbox_shift=mr_shiftdata[1] 
		
		return True

		
		
		
	def scale_cradata(self):
		self.fsr=2.5*self.fsr
		self.dcr=2.5*self.dcr
		self.tcr=2.5*self.tcr
		self.ssr=2.5*self.ssr
		self.hpr=2.5*self.hpr
		self.her=2.5*self.her        


	def sort_cradata(self):#document that method, check timestamp coincidence, for larger data sets only take first method!
		
		t0=clock()
		
		#sort cra data into the right order
		#checked with two different approaches and works, Make faster!. 
		self.times_cra_sorted=array([])
		self.step_cra_sorted=array([])
		self.vvps_sorted=array([])
		self.halt_sorted=array([])
		self.speed_sorted=array([])
		self.fsr_sorted=array([])
		self.dcr_sorted=array([])
		self.tcr_sorted=array([])
		self.ssr_sorted=array([])
		self.hpr_sorted=array([])
		self.her_sorted=array([])
		utimes_cra=unique(self.times_cra)
		l_timestamps_valid=[]
		#Methode checken! 
		
		doys=arange(int(utimes_cra[0]),int(utimes_cra[-1])+1,1)
		#print "doys", doys
		j=doys[0]
		
		last_timestamp_doybefore=-999
		#print "last_timestamp_doybefore",last_timestamp_doybefore
		
		l_times_sorted=[]
		
		while j<(doys[-1]+1):
			
			self.times_cra_sorted_doy=array([])
			self.step_cra_sorted_doy=array([])
			self.vvps_sorted_doy=array([])
			self.halt_sorted_doy=array([])
			self.speed_sorted_doy=array([])
			self.fsr_sorted_doy=array([])
			self.dcr_sorted_doy=array([])
			self.tcr_sorted_doy=array([])
			self.ssr_sorted_doy=array([])
			self.hpr_sorted_doy=array([])
			self.her_sorted_doy=array([])
		
			doymask_cra=(self.times_cra>=j)*(self.times_cra<(j+1))
			self.times_doy=self.times_cra[doymask_cra]
			self.step_doy=self.step_cra[doymask_cra]
			self.vvps_doy=self.vvps[doymask_cra]
			self.halt_doy=self.halt[doymask_cra]
			self.speed_doy=self.speed[doymask_cra]
			self.fsr_doy=self.fsr[doymask_cra]
			self.dcr_doy=self.dcr[doymask_cra]
			self.tcr_doy=self.tcr[doymask_cra]
			self.ssr_doy=self.ssr[doymask_cra]
			self.hpr_doy=self.hpr[doymask_cra]
			self.her_doy=self.her[doymask_cra]
			
			'''
			if j>doys[0]:
				doymask_cra_doybefore=((self.times_cra>=(j-1))*(self.times_cra<j))
				last_timestamp_doybefore=self.times_cra[doymask_cra_doybefore][-1]
			else:
				last_timestamp_doybefore=-999
			print "last_timestamp_doybefore",last_timestamp_doybefore    
			'''
			
			utimes_doy=unique(self.times_doy)
			i=0
			
			#if j==151:
			#    return utimes_doy
			
			l_times_sorted_doy=[]
			
			while i<len(utimes_doy):
				
				t10=clock()
				if len(self.times_doy[self.times_doy==utimes_doy[i]])>120:#CTOF nominal step number, incl. 4 pure processing steps 
					times_time=self.times_doy[self.times_doy==utimes_doy[i]][0:120]
					step_time=self.step_doy[self.times_doy==utimes_doy[i]][0:120]
					vvps_time=self.vvps_doy[self.times_doy==utimes_doy[i]][0:120]
					halt_time=self.halt_doy[self.times_doy==utimes_doy[i]][0:120]
					speed_time=self.speed_doy[self.times_doy==utimes_doy[i]][0:120]
					fsr_time=self.fsr_doy[self.times_doy==utimes_doy[i]][0:120]
					dcr_time=self.dcr_doy[self.times_doy==utimes_doy[i]][0:120]
					tcr_time=self.tcr_doy[self.times_doy==utimes_doy[i]][0:120]
					ssr_time=self.ssr_doy[self.times_doy==utimes_doy[i]][0:120]
					hpr_time=self.hpr_doy[self.times_doy==utimes_doy[i]][0:120]
					her_time=self.her_doy[self.times_doy==utimes_doy[i]][0:120] 
					
				
				else:
					times_time=self.times_doy[self.times_doy==utimes_doy[i]]
					step_time=self.step_doy[self.times_doy==utimes_doy[i]]
					vvps_time=self.vvps_doy[self.times_doy==utimes_doy[i]]
					halt_time=self.halt_doy[self.times_doy==utimes_doy[i]]
					speed_time=self.speed_doy[self.times_doy==utimes_doy[i]]
					fsr_time=self.fsr_doy[self.times_doy==utimes_doy[i]]
					dcr_time=self.dcr_doy[self.times_doy==utimes_doy[i]]
					tcr_time=self.tcr_doy[self.times_doy==utimes_doy[i]]
					ssr_time=self.ssr_doy[self.times_doy==utimes_doy[i]]
					hpr_time=self.hpr_doy[self.times_doy==utimes_doy[i]]
					her_time=self.her_doy[self.times_doy==utimes_doy[i]]
				
				t11=clock()
				
				inds=arange(0,len(step_time))
				
			
				m_wt=(step_time>min(step_time))*(inds<(where(step_time==min(step_time))[0]))#steps sorted out where steps are higher than minimum step, but step indices are lower!
				#return step_time,inds,(inds<(where(step_time==min(step_time))[0])),m_wt
					
				#print 'times_time test'
				#print times_time[0]
			
				if i>0:
					if (utimes_doy[i]-utimes_doy[i-1])*24*60<6:#improve with histogram
						times_time[m_wt]=utimes_doy[i-1]
						l_timestamps_valid.append(utimes_doy[i-1])
						#print "times test"
						#print times_time[0]
						#if (times_time[0]>151.158)*(times_time[1]<151.164):
						#    return step_time, invert(m_wt)
							
							
							
						
						#return m_wt,times_time[m_wt],times_time
					else:
						#print "times test, timegap"l_times_sorted
						#print times_time[0]
						#if (times_time[0]>151.158)*(times_time[1]<151.164):
						#    return step_time, invert(m_wt)
						
						times_time=times_time[invert(m_wt)]
						step_time=step_time[invert(m_wt)]
						vvps_time=vvps_time[invert(m_wt)]
						halt_time=halt_time[invert(m_wt)]
						speed_time=speed_time[invert(m_wt)]
						fsr_time=fsr_time[invert(m_wt)]
						dcr_time=dcr_time[invert(m_wt)]
						tcr_time=tcr_time[invert(m_wt)]
						ssr_time=ssr_time[invert(m_wt)]    
						hpr_time=hpr_time[invert(m_wt)]    
						her_time=her_time[invert(m_wt)]    
				
				elif i==0 and (utimes_doy[i]-last_timestamp_doybefore)*24*60<6:#improve with histogram
					times_time[m_wt]=utimes_doy[i-1]
					l_timestamps_valid.append(utimes_doy[i-1])

				
				else:
					#print "times test, zero"
					#print times_time[0]
					#if (times_time[0]>151.158)*(times_time[1]<151.164):
					#    return step_time, invert(m_wt)
					times_time=times_time[invert(m_wt)]
					step_time=step_time[invert(m_wt)]
					vvps_time=vvps_time[invert(m_wt)]
					halt_time=halt_time[invert(m_wt)]
					speed_time=speed_time[invert(m_wt)]
					fsr_time=fsr_time[invert(m_wt)]
					dcr_time=dcr_time[invert(m_wt)]
					tcr_time=tcr_time[invert(m_wt)]
					ssr_time=ssr_time[invert(m_wt)]    
					hpr_time=hpr_time[invert(m_wt)]    
					her_time=her_time[invert(m_wt)]    
				
					
				
				#return invert(m_wt),step_time
				
				t12=clock()
				
				#return times_time,step_time, i
				self.times_cra_sorted_doy=concatenate([self.times_cra_sorted_doy,times_time])   
				self.step_cra_sorted_doy=concatenate([self.step_cra_sorted_doy,step_time])              
				self.vvps_sorted_doy=concatenate([self.vvps_sorted_doy,vvps_time])    
				self.halt_sorted_doy=concatenate([self.halt_sorted_doy,halt_time])    
				self.speed_sorted_doy=concatenate([self.speed_sorted_doy,speed_time])    
				self.fsr_sorted_doy=concatenate([self.fsr_sorted_doy,fsr_time])    
				self.dcr_sorted_doy=concatenate([self.dcr_sorted_doy,dcr_time])    
				self.tcr_sorted_doy=concatenate([self.tcr_sorted_doy,tcr_time])    
				self.ssr_sorted_doy=concatenate([self.ssr_sorted_doy,ssr_time])    
				self.hpr_sorted_doy=concatenate([self.hpr_sorted_doy,hpr_time])    
				self.her_sorted_doy=concatenate([self.her_sorted_doy,her_time])    
				#return invert(m_wt),step_time
				
				#print 'times_time[0],len(times_time)'
				#print times_time[0],len(times_time)
				l_times_sorted_doy.append(times_time[0])

				i=i+1
			
				t13=clock()
				
			self.times_cra_sorted=concatenate([self.times_cra_sorted,self.times_cra_sorted_doy])   
			self.step_cra_sorted=concatenate([self.step_cra_sorted,self.step_cra_sorted_doy])              
			self.vvps_sorted=concatenate([self.vvps_sorted,self.vvps_sorted_doy])    
			self.halt_sorted=concatenate([self.halt_sorted,self.halt_sorted_doy])    
			self.speed_sorted=concatenate([self.speed_sorted,self.speed_sorted_doy])    
			self.fsr_sorted=concatenate([self.fsr_sorted,self.fsr_sorted_doy])    
			self.dcr_sorted=concatenate([self.dcr_sorted,self.dcr_sorted_doy])    
			self.tcr_sorted=concatenate([self.tcr_sorted,self.tcr_sorted_doy])    
			self.ssr_sorted=concatenate([self.ssr_sorted,self.ssr_sorted_doy])    
			self.hpr_sorted=concatenate([self.hpr_sorted,self.hpr_sorted_doy])    
			self.her_sorted=concatenate([self.her_sorted,self.her_sorted_doy]) 
			
			#change last time stamp for the time gap calculation!
			if len(utimes_doy)>0:#only if there are any valid timestamps in this doy
				last_timestamp_doybefore=utimes_doy[i-1]
			else:
				last_timestamp_doybefore=-999
			#print "last_timestamp_doybefore",last_timestamp_doybefore
			
			l_times_sorted_doy=array(l_times_sorted_doy)
			l_times_sorted.append(l_times_sorted_doy)
			
			j=j+1
		
		l_times_sorted=ravel(array(l_times_sorted))
		
		t1=clock()
		
		#find error in sorting algorithm!
		#save (pickle) synchronized data
		#find error in histogramming algorithm!
		#load and synchronize matrix data
		
		
		#following part only for checking the sorting procedure; still has to be checked once with the whole data set!
		
		'''
		self.newsteps=array([])
		self.newsteps_time=array([])
		k=0
		while k<len(utimes_cra)-1:
		    #print utimes_cra[k]
		    mt0=(self.times_cra==utimes_cra[k])
		    mt1=(self.times_cra==utimes_cra[k+1])
		    steptime0=self.step_cra[mt0][0:120]
		    steptime1=self.step_cra[mt1][0:120]
		    ind0=where(steptime0==min(self.step_cra[mt0]))[0]
		    ind1=where(steptime1==min(self.step_cra[mt1]))[0]
		    try:
		        corsteps0=steptime0[ind0:]
		        falsesteps1=steptime1[0:ind1]
		    except TypeError:
		        print "TypeError"
		        return ind0,ind1
		    if (utimes_cra[k+1]-utimes_cra[k])*24*60<6:#improve with histogram           
		        newsteps0=concatenate([corsteps0,falsesteps1])
		    else:
		        newsteps0=corsteps0
		    #if utimes_cra[k]==self.times_cra_sorted[5394]:
		    #    return mt0,mt1,ind0,ind1, corsteps0,falsesteps1,newsteps0
		    #print k,corsteps0,falsesteps1,newsteps0
		    #newsteps=concatenate([newsteps,newsteps0])
		    self.newsteps=concatenate([self.newsteps,newsteps0])   
		    self.newsteps_time=concatenate([self.newsteps_time,array([utimes_cra[k]]*len(newsteps0))])
		    k=k+1
		'''    
		t2=clock()   
		
		
		
		#l_timestamps_valid=array(l_timestamps_valid)
		print("method runtimes [s]: t2-t0")
		print("t1-t0:", t1-t0)
		print("t2-t1:", t2-t1)
		print("t11-t10:", t11-t0)
		print("t12-t11:", t12-t11)
		print("t13-t12:", t13-t12)
		
		#return l_times_sorted
		return True



	def pickle_CTOFdata_piecewise(self,filename,path="/home/asterix/janitzek/ctof/CTOFdata_pickled/"):
		PIK = path+filename
		data_PHA=self.load_CTOFfull()
		data_PM=self.load_PM()
		#return data_PHA,data_PM
		data_PM_sync=[]
		for day_PHA in data_PHA:
			mask_day=[(data_PM>=day_PHA[0][0])*(data_PM<=day_PHA[0][-1])]
			return data
			day_PM=data_PM[mask_day]
			day_PM_sync=self.sync_CTOFPM(day_PHA,day_PM)
			return data_PM_sync
		data_PM_sync=[].append(day_PM_sync)
		return data_PM_sync
		
		with open(PIK, "wb") as f:
			pickle.dump(len(data), f)
			for day in data:
				pickle.dump(day, f,protocol=-1)

		        
		        
	def unpickle_CTOFdata_piecewise(self,filename,path="/home/asterix/janitzek/ctof/CTOFdata_pickled/"):
		PIK = path+filename
		data = []
		with open(PIK, "rb") as f:
			for _ in range(pickle.load(f)):
				data.append(pickle.load(f))
		l=len(concatenate(data[0]))
		data_out=zeros((len(data),l)) 
		i=0
		while i<len(data):
			data_out[i]=concatenate(data[i])
			i=i+1
		
		#print "No such data available, Data must contain 'PHA' and/or 'CRA'!"  

		return data_out

        
	def sync_CTOFPM_PHA(self,pmdat,tol = 308):#tol in seconds
		"""
		method to synchronize pm data with CTOF PHA data.
		docstring, method commented in detail within the code!
		tolerance taken from pm time-delta histogram (histogram of the differences of successive PM timestamps). 
		Maximum time delay between data = tol/2 = 154 seconds (=synchronization accuracy).
		"""
	   
		#fill data gadef ps in pm data with dummy values
		tol_day=tol/(24.*3600)
		i = 0
		pmtimes_nogaps=array([])
		pmvel_nogaps=array([])
		pmvth_nogaps=array([])
		pmdens_nogaps=array([])
		
		while i < len(pmdat.time)-1:
			#print i
			tfill=arange(pmdat.time[i],pmdat.time[i+1],tol_day)
			#print pmdat.vel[i], len(tfill)-1
			velfill=concatenate([array([pmdat.vel[i]]),array([-999]*(len(tfill)-1))])
			vthfill=concatenate([array([pmdat.vth[i]]),array([-999]*(len(tfill)-1))])
			densfill=concatenate([array([pmdat.dens[i]]),array([-999]*(len(tfill)-1))])
			#print tfill,velfill,vthfill,densfill
			
			pmtimes_nogaps=append(pmtimes_nogaps,tfill)
			pmvel_nogaps=append(pmvel_nogaps,velfill)
			pmvth_nogaps=append(pmvth_nogaps,vthfill)
			pmdens_nogaps=append(pmdens_nogaps,densfill)                       
			i=i+1
		pmdat.time=append(pmtimes_nogaps,pmdat.time[-1])    
		pmdat.vel=append(pmvel_nogaps,pmdat.vel[-1])
		pmdat.vth=append(pmvth_nogaps,pmdat.vth[-1])
		pmdat.dens=append(pmdens_nogaps,pmdat.dens[-1])
		
		#shift pm-time, so that CTOF-data will be synchronized to the closest PM-timestamp
		pmtime_shifted=(pmdat.time[1:]+pmdat.time[:-1])/2.
		pmdat.time_shifted=concatenate([array([pmdat.time[0]-tol_day/2.]),pmtime_shifted])
		
		#synchronize CTOF-data to PM-timestamps by using numpy.searchsorted

		#return searchsorted(pmdat.time,self.times, side = "right")
		zero_shift=searchsorted(pmdat.time,self.times, side = "right")[0]         
		time_indices = searchsorted(pmdat.time_shifted,self.times, side = "right")-zero_shift
		pmdat.time_sync = pmdat.time[time_indices]
		pmdat.time_shifted_sync = pmdat.time_shifted[time_indices]
		pmdat.vel_sync = pmdat.vel[time_indices]
		pmdat.vth_sync = pmdat.vth[time_indices]
		pmdat.dens_sync = pmdat.dens[time_indices]
		print("PM data synchronized with CTOF PHA data")
		return True
		

		
	def sync_CTOF_CMR(self,tol_cmr = 363,tol_pha=308):#tol in seconds, derived from timestamp histogram
		"""
		method to synchronize pm data with CTOF PHA data.
		docstring, method commented in detail within the code!
		tolerance taken from pm time-delta histogram (histogram of the differences of successive PM timestamps). 
		Maximum time delay between data = tol/2 = 154 seconds (=synchronization accuracy).
		"""
	   
		t0=clock()
		
		#fill gaps in pha data
		tol_day_pha=tol_pha/(24.*3600)
		i = 0
		pha_times_nogaps=array([])
	   
	   
		utimes_pha=unique(self.times)
		while i < len(utimes_pha)-1:
			pha_time_fill=arange(utimes_pha[i],utimes_pha[i+1],tol_day_pha)#no additional gap-filling data needed but timestamps for searchsorted!
			pha_times_nogaps=append(pha_times_nogaps,pha_time_fill)                      
			i=i+1
		
		t1=clock()
		
		#fill gaps in cmr data
		tol_day_cmr=tol_cmr/(24.*3600)
		
		cmr_times_nogaps_alldoys=[]
		cmr_number_nogaps_alldoys=[]
		cmr_vsw_nogaps_alldoys=[]
		cmr_mr0_nogaps_alldoys=[]
		cmr_mr1_nogaps_alldoys=[]
		cmr_mr2_nogaps_alldoys=[]
		cmr_mr3_nogaps_alldoys=[]
		cmr_mr4_nogaps_alldoys=[]
		cmr_mr5_nogaps_alldoys=[]
		cmr_mr6_nogaps_alldoys=[]
		cmr_mr7_nogaps_alldoys=[]
		cmr_mr8_nogaps_alldoys=[]
		cmr_mr9_nogaps_alldoys=[]
		cmr_mr10_nogaps_alldoys=[]
		cmr_mr11_nogaps_alldoys=[]
		cmr_mr12_nogaps_alldoys=[]
		cmr_mr13_nogaps_alldoys=[]
		cmr_mr14_nogaps_alldoys=[]
		cmr_mr15_nogaps_alldoys=[]
		cmr_mr16_nogaps_alldoys=[]
		cmr_mr17_nogaps_alldoys=[]
		cmr_mr18_nogaps_alldoys=[]
		cmr_mr19_nogaps_alldoys=[]
		cmr_mr20_nogaps_alldoys=[]
	   
		#iterate over days to avoid large arrays and therefore speed calculation up
		utimes_cmr=unique(self.times_mr)
		doys=arange(int(utimes_cmr[0]),int(utimes_cmr[-1])+1,1)
		#print "doys sorted",doys
		l=doys[0]
		while l<(doys[-1]+1):
			#print "doy"
			#print l
			
			#set doy (=day of year) mask
			#doymask_PHA=(self.times>=l)*(self.times<l+1)
			
			udoymask_cmr=(utimes_cmr>=l)*(utimes_cmr<l+1)
			doymask_cmr=(self.times_mr>=l)*(self.times_mr<l+1)
			
			utimes_cmr_doy=utimes_cmr[udoymask_cmr]
			
			self.times_cmr_doy=self.times_mr[doymask_cmr]
			self.number_cmr_doy=self.mr_number[doymask_cmr]
			self.vsw_cmr_doy=self.vsw_mr[doymask_cmr]
			self.mr0_cmr_doy=self.mr0[doymask_cmr]
			self.mr1_cmr_doy=self.mr1[doymask_cmr]
			self.mr2_cmr_doy=self.mr2[doymask_cmr]
			self.mr3_cmr_doy=self.mr3[doymask_cmr]
			self.mr4_cmr_doy=self.mr4[doymask_cmr]
			self.mr5_cmr_doy=self.mr5[doymask_cmr]
			self.mr6_cmr_doy=self.mr6[doymask_cmr]
			self.mr7_cmr_doy=self.mr7[doymask_cmr]
			self.mr8_cmr_doy=self.mr8[doymask_cmr]
			self.mr9_cmr_doy=self.mr9[doymask_cmr]
			self.mr10_cmr_doy=self.mr10[doymask_cmr]
			self.mr11_cmr_doy=self.mr11[doymask_cmr]
			self.mr12_cmr_doy=self.mr12[doymask_cmr]
			self.mr13_cmr_doy=self.mr13[doymask_cmr]
			self.mr14_cmr_doy=self.mr14[doymask_cmr]
			self.mr15_cmr_doy=self.mr15[doymask_cmr]
			self.mr16_cmr_doy=self.mr16[doymask_cmr]
			self.mr17_cmr_doy=self.mr17[doymask_cmr]
			self.mr18_cmr_doy=self.mr18[doymask_cmr]
			self.mr19_cmr_doy=self.mr19[doymask_cmr]
			self.mr20_cmr_doy=self.mr20[doymask_cmr]
			
			
			
			k = 0
			while k<len(utimes_cmr_doy)-1:
			#while k < len(unique(self.times_cra_sorted))-1:
				cmr_time_fill=arange(unique(self.times_cmr_doy)[k],unique(self.times_cmr_doy)[k+1],tol_day_cmr)#close gaps with 5-minute time stamps
				if len(cmr_time_fill)>1:
					ind_last=where(self.times_cmr_doy==unique(self.times_cmr_doy)[k])[0][-1]+1
					cmr_times_nogaps_doy=insert(self.times_cmr_doy,ind_last,cmr_time_fill[1:])          
					cmr_number_nogaps_doy=insert(self.number_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_vsw_nogaps_doy=insert(self.vsw_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr0_nogaps_doy=insert(self.mr0_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr1_nogaps_doy=insert(self.mr1_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr2_nogaps_doy=insert(self.mr2_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr3_nogaps_doy=insert(self.mr3_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr4_nogaps_doy=insert(self.mr4_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr5_nogaps_doy=insert(self.mr5_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr6_nogaps_doy=insert(self.mr6_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr7_nogaps_doy=insert(self.mr7_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr8_nogaps_doy=insert(self.mr8_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr9_nogaps_doy=insert(self.mr9_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr10_nogaps_doy=insert(self.mr10_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr11_nogaps_doy=insert(self.mr11_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr12_nogaps_doy=insert(self.mr12_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr13_nogaps_doy=insert(self.mr13_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr14_nogaps_doy=insert(self.mr14_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr15_nogaps_doy=insert(self.mr15_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr16_nogaps_doy=insert(self.mr16_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr17_nogaps_doy=insert(self.mr17_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr18_nogaps_doy=insert(self.mr18_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr19_nogaps_doy=insert(self.mr19_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					cmr_mr20_nogaps_doy=insert(self.mr20_cmr_doy,ind_last,array([-999]*len(cmr_time_fill[1:])))
					k=k+len(cmr_time_fill)+1
				else:
					cmr_times_nogaps_doy=self.times_cmr_doy
					cmr_number_nogaps_doy=self.number_cmr_doy
					cmr_vsw_nogaps_doy=self.vsw_cmr_doy
					cmr_mr0_nogaps_doy=self.mr0_cmr_doy
					cmr_mr1_nogaps_doy=self.mr1_cmr_doy
					cmr_mr2_nogaps_doy=self.mr2_cmr_doy
					cmr_mr3_nogaps_doy=self.mr3_cmr_doy
					cmr_mr4_nogaps_doy=self.mr4_cmr_doy
					cmr_mr5_nogaps_doy=self.mr5_cmr_doy
					cmr_mr6_nogaps_doy=self.mr6_cmr_doy
					cmr_mr7_nogaps_doy=self.mr7_cmr_doy
					cmr_mr8_nogaps_doy=self.mr8_cmr_doy
					cmr_mr9_nogaps_doy=self.mr9_cmr_doy
					cmr_mr10_nogaps_doy=self.mr10_cmr_doy
					cmr_mr11_nogaps_doy=self.mr11_cmr_doy
					cmr_mr12_nogaps_doy=self.mr12_cmr_doy
					cmr_mr13_nogaps_doy=self.mr13_cmr_doy
					cmr_mr14_nogaps_doy=self.mr14_cmr_doy
					cmr_mr15_nogaps_doy=self.mr15_cmr_doy
					cmr_mr16_nogaps_doy=self.mr16_cmr_doy
					cmr_mr17_nogaps_doy=self.mr17_cmr_doy
					cmr_mr18_nogaps_doy=self.mr18_cmr_doy
					cmr_mr19_nogaps_doy=self.mr19_cmr_doy
					cmr_mr20_nogaps_doy=self.mr20_cmr_doy
					k=k+1
			
			cmr_times_nogaps_alldoys.append(cmr_times_nogaps_doy)   
			cmr_number_nogaps_alldoys.append(cmr_number_nogaps_doy)
			cmr_vsw_nogaps_alldoys.append(cmr_vsw_nogaps_doy)
			cmr_mr0_nogaps_alldoys.append(cmr_mr0_nogaps_doy)
			cmr_mr1_nogaps_alldoys.append(cmr_mr1_nogaps_doy)
			cmr_mr2_nogaps_alldoys.append(cmr_mr2_nogaps_doy)
			cmr_mr3_nogaps_alldoys.append(cmr_mr3_nogaps_doy)
			cmr_mr4_nogaps_alldoys.append(cmr_mr4_nogaps_doy)
			cmr_mr5_nogaps_alldoys.append(cmr_mr5_nogaps_doy)
			cmr_mr6_nogaps_alldoys.append(cmr_mr6_nogaps_doy)
			cmr_mr7_nogaps_alldoys.append(cmr_mr7_nogaps_doy)
			cmr_mr8_nogaps_alldoys.append(cmr_mr8_nogaps_doy)
			cmr_mr9_nogaps_alldoys.append(cmr_mr9_nogaps_doy)
			cmr_mr10_nogaps_alldoys.append(cmr_mr10_nogaps_doy)
			cmr_mr11_nogaps_alldoys.append(cmr_mr11_nogaps_doy)
			cmr_mr12_nogaps_alldoys.append(cmr_mr12_nogaps_doy)
			cmr_mr13_nogaps_alldoys.append(cmr_mr13_nogaps_doy)
			cmr_mr14_nogaps_alldoys.append(cmr_mr14_nogaps_doy)
			cmr_mr15_nogaps_alldoys.append(cmr_mr15_nogaps_doy)
			cmr_mr16_nogaps_alldoys.append(cmr_mr16_nogaps_doy)
			cmr_mr17_nogaps_alldoys.append(cmr_mr17_nogaps_doy)
			cmr_mr18_nogaps_alldoys.append(cmr_mr18_nogaps_doy)
			cmr_mr19_nogaps_alldoys.append(cmr_mr19_nogaps_doy)
			cmr_mr20_nogaps_alldoys.append(cmr_mr20_nogaps_doy)
			l=l+1

		cmr_times_nogaps=concatenate(cmr_times_nogaps_alldoys) 
		cmr_number_nogaps=concatenate(cmr_number_nogaps_alldoys)
		cmr_vsw_nogaps=concatenate(cmr_vsw_nogaps_alldoys)
		cmr_mr0_nogaps=concatenate(cmr_mr0_nogaps_alldoys)
		cmr_mr1_nogaps=concatenate(cmr_mr1_nogaps_alldoys)
		cmr_mr2_nogaps=concatenate(cmr_mr2_nogaps_alldoys)
		cmr_mr3_nogaps=concatenate(cmr_mr3_nogaps_alldoys)
		cmr_mr4_nogaps=concatenate(cmr_mr4_nogaps_alldoys)
		cmr_mr5_nogaps=concatenate(cmr_mr5_nogaps_alldoys)
		cmr_mr6_nogaps=concatenate(cmr_mr6_nogaps_alldoys)
		cmr_mr7_nogaps=concatenate(cmr_mr7_nogaps_alldoys)
		cmr_mr8_nogaps=concatenate(cmr_mr8_nogaps_alldoys)
		cmr_mr9_nogaps=concatenate(cmr_mr9_nogaps_alldoys)
		cmr_mr10_nogaps=concatenate(cmr_mr10_nogaps_alldoys)
		cmr_mr11_nogaps=concatenate(cmr_mr11_nogaps_alldoys)
		cmr_mr12_nogaps=concatenate(cmr_mr12_nogaps_alldoys)
		cmr_mr13_nogaps=concatenate(cmr_mr13_nogaps_alldoys)
		cmr_mr14_nogaps=concatenate(cmr_mr14_nogaps_alldoys)
		cmr_mr15_nogaps=concatenate(cmr_mr15_nogaps_alldoys)
		cmr_mr16_nogaps=concatenate(cmr_mr16_nogaps_alldoys)
		cmr_mr17_nogaps=concatenate(cmr_mr17_nogaps_alldoys)
		cmr_mr18_nogaps=concatenate(cmr_mr18_nogaps_alldoys)
		cmr_mr19_nogaps=concatenate(cmr_mr19_nogaps_alldoys)
		cmr_mr20_nogaps=concatenate(cmr_mr20_nogaps_alldoys)
		
		t2=clock()
		
		#shift pha times about half a timestamp to get rid of 1-minute shift effect in cra data
		pha_times_shifted=(unique(pha_times_nogaps[1:]+pha_times_nogaps[:-1]))/2.
	   
		#cut off cra data before firts pha timestamp to apply searchsorted for synchronizing cra times with pha times (see next step)   
		m=cmr_times_nogaps>pha_times_shifted[0]
		cmr_times_nogaps=cmr_times_nogaps[m]
		#cmr_number_nogaps=cmr_number_nogaps[m]
		#cmr_vsw_nogaps=cmr_vsw_nogaps[m]
		#cmr_mr0_nogaps=cmr_mr0_nogaps[m]
		
		
		#synchronizing cra times with pha times
		ss=searchsorted(pha_times_shifted,cmr_times_nogaps)-1
		cmr_times_sync=unique(pha_times_nogaps)[ss]#check for gaps or double assignments!, why only 281 entries?
		
		#m_times=(self.times>min(cmr_times_sync))*(self.times<max(cmr_times_sync))
		#ss1=searchsorted(cmr_times_sync,self.times[m_times])
		
		self.times_cmr_sync=cmr_times_sync
		self.number_cmr_sync=cmr_number_nogaps[m]
		self.vsw_cmr_sync=cmr_vsw_nogaps[m]
		
		self.mr0_sync=cmr_mr0_nogaps[m]
		self.mr1_sync=cmr_mr1_nogaps[m]
		self.mr2_sync=cmr_mr2_nogaps[m]
		self.mr3_sync=cmr_mr3_nogaps[m]
		self.mr4_sync=cmr_mr4_nogaps[m]
		self.mr5_sync=cmr_mr5_nogaps[m]
		self.mr6_sync=cmr_mr6_nogaps[m]
		self.mr7_sync=cmr_mr7_nogaps[m]
		self.mr8_sync=cmr_mr8_nogaps[m]
		self.mr9_sync=cmr_mr9_nogaps[m]
		self.mr10_sync=cmr_mr10_nogaps[m]
		self.mr11_sync=cmr_mr11_nogaps[m]
		self.mr12_sync=cmr_mr12_nogaps[m]
		self.mr13_sync=cmr_mr13_nogaps[m]
		self.mr14_sync=cmr_mr14_nogaps[m]
		self.mr15_sync=cmr_mr15_nogaps[m]
		self.mr16_sync=cmr_mr16_nogaps[m]
		self.mr17_sync=cmr_mr17_nogaps[m]
		self.mr18_sync=cmr_mr18_nogaps[m]
		self.mr19_sync=cmr_mr19_nogaps[m]
		self.mr20_sync=cmr_mr20_nogaps[m]
		
	   

		matrix_rates1=array([self.mr0_sync,self.mr1_sync,self.mr2_sync,self.mr3_sync,self.mr4_sync,self.mr5_sync,self.mr6_sync,self.mr7_sync,self.mr8_sync,self.mr9_sync,self.mr10_sync])
		matrix_rates2=array([self.mr11_sync,self.mr12_sync,self.mr13_sync,self.mr14_sync,self.mr15_sync,self.mr16_sync,self.mr17_sync,self.mr18_sync,self. mr19_sync, self.mr20_sync])
		self.matrix_rates=concatenate([matrix_rates1,matrix_rates2])
		
		
		t3=clock()
		
		print("PHA-CMR data synchronized")
		print("synchronization time (PHA and CMR data):",t3-t0)
		
		return True
		
		
		
		
		
		
		

	def sync_CTOFPM(self,pmdat,tol_cra = 363,tol_pha = 308):#tol in seconds
		"""
		method to synchronize pm data with CTOF PHA, CRA and CMR (CHK) data, based on sync_CTOFPM
		docstring, method commented in detail within the code!
		Maximum time delay between data = tol/2 = 154 seconds (=synchronization accuracy).
		"""
	 
		t0=clock()
		
		#fill gaps in pha data
		tol_day_pha=tol_pha/(24.*3600)
		i = 0
		pha_times_nogaps=array([])
		pha_steps_nogaps=array([])
		pha_vvps_nogaps=array([])
		pha_halt_nogaps=array([])
		pha_speed_nogaps=array([])
		pha_fsr_nogaps=array([])
		pha_dcr_nogaps=array([])
		pha_tcr_nogaps=array([])
		pha_ssr_nogaps=array([])
		pha_hpr_nogaps=array([])
		pha_her_nogaps=array([])
		
		
		
		#fill gaps in PHA data
		utimes_pha=unique(self.times)
		while i < len(utimes_pha)-1:
			pha_time_fill=arange(utimes_pha[i],utimes_pha[i+1],tol_day_pha)#no additional gap-filling data needed but timestamps for searchsorted!
			pha_times_nogaps=append(pha_times_nogaps,pha_time_fill)
			i=i+1
		
		#fill gaps in cra data
		tol_day_cra=tol_cra/(24.*3600)
		i = 0
		print("former timestamp array length")
		print(len(unique(self.times_cra_sorted)),len(self.times_cra_sorted))
		while i < len(unique(self.times_cra_sorted))-1:
			cra_time_fill=arange(unique(self.times_cra_sorted)[i],unique(self.times_cra_sorted)[i+1],tol_day_cra)#close gaps with 5-minute time stamps
			if len(cra_time_fill)>1:
				print(i,unique(self.times_cra_sorted)[i],cra_time_fill[1:],len(cra_time_fill[1:]))
				ind_last=where(self.times_cra_sorted==unique(self.times_cra_sorted)[i])[0][-1]+1
				#return True
				self.times_cra_sorted=insert(self.times_cra_sorted,ind_last+1,cra_time_fill[1:])          
				#if len(cra_time_fill[1:])==0:
				#    return ind_last,cra_time_fill[1:]

			
				#print self.step_cra_sorted,shape(self.step_cra_sorted),array([-999]*len(cra_time_fill)),shape(array([-999]*len(cra_time_fill)))
				self.step_cra_sorted=insert(self.step_cra_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.vvps_sorted=insert(self.vvps_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.halt_sorted=insert(self.halt_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.speed_sorted=insert(self.speed_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.fsr_sorted=insert(self.fsr_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.dcr_sorted=insert(self.dcr_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.tcr_sorted=insert(self.tcr_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.ssr_sorted=insert(self.ssr_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.hpr_sorted=insert(self.hpr_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				self.her_sorted=insert(self.her_sorted,ind_last,array([-999]*len(cra_time_fill[1:])))
				i=i+len(cra_time_fill)+1
			else:
				i=i+1
		    #append last value   
		#return True
		
		t1=clock()
		
		#shift pha times to the middle to get rid of 1-minute shift effect in cra data
		pha_times_shifted=(unique(pha_times_nogaps[1:]+pha_times_nogaps[:-1]))/2.
		
		#return pha_times_nogaps
		
		
		#cut off cra time stamps below first tmestamp of cra data 
		cra_times_nogaps=self.times_cra_sorted
		cra_steps_nogaps=self.step_cra_sorted
		cra_vvps_nogaps=self.vvps_sorted
		cra_halt_nogaps=self.halt_sorted
		cra_speed_nogaps=self.speed_sorted
		cra_fsr_nogaps=self.fsr_sorted
		cra_dcr_nogaps=self.dcr_sorted
		cra_tcr_nogaps=self.tcr_sorted
		cra_ssr_nogaps=self.ssr_sorted
		cra_hpr_nogaps=self.hpr_sorted
		cra_her_nogaps=self.her_sorted
	   
		#return pha_times_shifted
		m=cra_times_nogaps>pha_times_shifted[0]
		cra_times_nogaps=cra_times_nogaps[m]
		cra_steps_nogaps=cra_steps_nogaps[m]
		
		#return m,cra_halt_nogaps,cra_vvps_nogaps
		cra_vvps_nogaps=cra_vvps_nogaps[m]
		
		cra_halt_nogaps=cra_halt_nogaps[m]
		cra_speed_nogaps=cra_speed_nogaps[m]
		cra_fsr_nogaps=cra_fsr_nogaps[m]
		cra_dcr_nogaps=cra_dcr_nogaps[m]
		cra_tcr_nogaps=cra_tcr_nogaps[m]
		cra_ssr_nogaps=cra_ssr_nogaps[m]
		cra_hpr_nogaps=cra_hpr_nogaps[m]
		cra_her_nogaps=cra_her_nogaps[m]
		

		ss=searchsorted(pha_times_shifted,cra_times_nogaps)-1
		cra_times_sync=unique(pha_times_nogaps)[ss]
		
		self.step_cra_sync_slow=zeros((len(self.times)))-999
		self.vvps_sync_slow=zeros((len(self.times)))-999
		self.halt_sync_slow=zeros((len(self.times)))-999
		self.speed_sync_slow = zeros((len(self.times)))-999
		self.fsr_sync_slow = zeros((len(self.times)))-999
		self.dcr_sync_slow = zeros((len(self.times)))-999
		self.tcr_sync_slow = zeros((len(self.times)))-999
		self.ssr_sync_slow = zeros((len(self.times)))-999
		self.hpr_sync_slow = zeros((len(self.times)))-999
		self.her_sync_slow = zeros((len(self.times)))-999
		
		t2=clock()
	   
		
		i=0
		utimes_cra_sync=unique(cra_times_sync)#check whether there are really no missed utime_pha except for at the end!
		#return utimes_cra_sync
		while i<len(utimes_cra_sync): 
			
			t20=clock()
			
			print(i, utimes_cra_sync[i],len(utimes_cra_sync))
			inds=self.times==utimes_cra_sync[i]# what happens in c.times gaps?
			inds_cra=cra_times_sync==utimes_cra_sync[i]
			#if utime[i]==150.16421299999999:
			#    return inds, inds_cra
			print("test_inds")
			print(len(where(inds==True)[0]))
			
			t21=clock()
			print("t21-t20")
			print(t21-t20)
			
			if len(where(inds==True)[0])>0:#timesteps where no pha data is recorded is sorted out here
				if min(cra_steps_nogaps[inds_cra])!=-999:#no dummy value allowed (up to here: one value==dummy => all values==dummy).
					
					t22=clock()
					
					test_variable=0
					#print "min_test"
					#print len(cra_steps_nogaps[inds_cra]), len(self.step[inds])
					
					steps_pha_sliced=self.step[inds]
					steps_cra_sliced=cra_steps_nogaps[inds_cra]
					vvps_sliced=cra_vvps_nogaps[inds_cra]
					halt_sliced=cra_halt_nogaps[inds_cra]
					speed_sliced=cra_speed_nogaps[inds_cra]
					fsr_sliced=cra_fsr_nogaps[inds_cra]
					dcr_sliced=cra_dcr_nogaps[inds_cra]
					tcr_sliced=cra_tcr_nogaps[inds_cra]
					ssr_sliced=cra_ssr_nogaps[inds_cra]
					hpr_sliced=cra_hpr_nogaps[inds_cra]
					her_sliced=cra_her_nogaps[inds_cra]
					
					
					t23=clock()
					print("t23-t22")
					print(t23-t22)
					
					if min(steps_cra_sliced)>min(steps_pha_sliced):
						
						steps_cra_sliced=concatenate([array([min(steps_cra_sliced)-1]),steps_cra_sliced])
						test_variable=1
						
					
					ss_steps=searchsorted(steps_cra_sliced,steps_pha_sliced)
					#check length of searchsorted arrays!
					
					if test_variable==1:
						#steps_cra_sliced=concatenate([array([-999]),steps_cra_sliced])                
						steps_cra_sliced[0]=-999
					if max(steps_cra_sliced)<max(steps_pha_sliced):
						print("True Test")
						steps_cra_sliced=concatenate([steps_cra_sliced,array([-999])])
						vvps_sliced=concatenate([vvps_sliced,array([-999])])
						halt_sliced=concatenate([halt_sliced,array([-999])])
						speed_sliced=concatenate([speed_sliced,array([-999])])
						fsr_sliced=concatenate([fsr_sliced,array([-999])])
						dcr_sliced=concatenate([dcr_sliced,array([-999])])
						tcr_sliced=concatenate([tcr_sliced,array([-999])])
						ssr_sliced=concatenate([ssr_sliced,array([-999])])
						hpr_sliced=concatenate([hpr_sliced,array([-999])])
						her_sliced=concatenate([her_sliced,array([-999])])
						
					t24=clock()
					print("t24-t23")
					print(t24-t23)
					
					#self.inds[inds]=inds
					#self.inds_steps[inds]=inds_steps
					#return ss_steps,cra_steps_nogaps[inds_cra]
					try: 
						print("test yes")
						self.step_cra_sync_slow[inds]=steps_cra_sliced[ss_steps]#what happens if c.step too high for cra step?
						self.vvps_sync_slow[inds]=vvps_sliced[ss_steps]
						self.halt_sync_slow[inds]=halt_sliced[ss_steps]
						self.speed_sync_slow[inds]=speed_sliced[ss_steps]
						self.fsr_sync_slow[inds]=fsr_sliced[ss_steps]
						self.dcr_sync_slow[inds]=dcr_sliced[ss_steps]
						self.tcr_sync_slow[inds]=tcr_sliced[ss_steps]
						self.ssr_sync_slow[inds]=ssr_sliced[ss_steps]
						self.hpr_sync_slow[inds]=hpr_sliced[ss_steps]
						self.her_sync_slow[inds]=her_sliced[ss_steps]
					
					
					except IndexError:
						print("test no")
						print("IndexError")
						return steps_cra_sliced,steps_pha_sliced,ss_steps,test_variable
					
					self.step_cra_sync_slow[inds]=concatenate([array([-999]),cra_steps_nogaps[inds_cra]])[ss_steps]
					self.vvps_sync_slow[inds]=concatenate([array([-999]),cra_vvps_nogaps[inds_cra]])[ss_steps]
					self.halt_sync_slow[inds]=concatenate([array([-999]),cra_halt_nogaps[inds_cra]])[ss_steps]
					self.speed_sync_slow[inds]=concatenate([array([-999]),cra_speed_nogaps[inds_cra]])[ss_steps]
					self.fsr_sync_slow[inds]=concatenate([array([-999]),cra_fsr_nogaps[inds_cra]])[ss_steps]
					self.dcr_sync_slow[inds]=concatenate([array([-999]),cra_dcr_nogaps[inds_cra]])[ss_steps]
					self.tcr_sync_slow[inds]=concatenate([array([-999]),cra_tcr_nogaps[inds_cra]])[ss_steps]
					self.ssr_sync_slow[inds]=concatenate([array([-999]),cra_ssr_nogaps[inds_cra]])[ss_steps]
					self.hpr_sync_slow[inds]=concatenate([array([-999]),cra_hpr_nogaps[inds_cra]])[ss_steps]
					self.her_sync_slow[inds]=concatenate([array([-999]),cra_her_nogaps[inds_cra]])[ss_steps]
					
					t25=clock()
					print("t25-t24")
					print(t25-t24)
					
					
					
				else:
					t26=clock()
					
					self.step_cra_sync_slow[inds]=array([-999]*len(ss_steps))
					self.vvps_sync_slow[inds]=array([-999]*len(ss_steps))
					self.halt_sync_slow[inds]=array([-999]*len(ss_steps))
					self.speed_sync_slow[inds]=array([-999]*len(ss_steps))
					self.fsr_sync_slow[inds]=array([-999]*len(ss_steps))
					self.dcr_sync_slow[inds]=array([-999]*len(ss_steps))
					self.tcr_sync_slow[inds]=array([-999]*len(ss_steps))
					self.ssr_sync_slow[inds]=array([-999]*len(ss_steps))
					self.hpr_sync_slow[inds]=array([-999]*len(ss_steps))
					self.her_sync_slow[inds]=array([-999]*len(ss_steps))
			
					t27=clock()
					print("t27-t26")
					print(t27-t26)
					
					
					
					
					
			i=i+1

			
			t3=clock()
		
		print("synchronization method runtimes:")
		print("total method runtime:", t3-t0)
		print(t1-t0)
		print(t2-t1)
		print(t3-t2)
		
		print("PM data synchronized with CTOF PHA and CRA data")       
		return True#addapt other quantities than step and times correctly! 

		
		
		
		
		
	def sync_CTOFPM_fast(self,pmdat,tol_cra = 363,tol_pha = 308):#tol in seconds
		"""
		method to synchronize PM data with CTOF PHA, CRA data, based on sync_CTOFPM
		docstring, method commented in detail within the code!
		Maximum time delay between data = tol/2 = 154 seconds (=synchronization accuracy).
		"""
	 
		t0=clock()
		
		#fill gaps in pha data
		tol_day_pha=tol_pha/(24.*3600)
		i = 0
		pha_times_nogaps=array([])
		pha_steps_nogaps=array([])
		pha_vvps_nogaps=array([])
		pha_halt_nogaps=array([])
		pha_speed_nogaps=array([])
		pha_fsr_nogaps=array([])
		pha_dcr_nogaps=array([])
		pha_tcr_nogaps=array([])
		pha_ssr_nogaps=array([])
		pha_hpr_nogaps=array([])
		pha_her_nogaps=array([])
	   
		utimes_pha=unique(self.times)
		while i < len(utimes_pha)-1:
			pha_time_fill=arange(utimes_pha[i],utimes_pha[i+1],tol_day_pha)#no additional gap-filling data needed but timestamps for searchsorted!
			pha_times_nogaps=append(pha_times_nogaps,pha_time_fill)                      
			i=i+1
		
		t_check=clock()
		
		#fill gaps in cra data
		tol_day_cra=tol_cra/(24.*3600)
		
		cra_times_nogaps_alldoys=[]
		cra_steps_nogaps_alldoys=[]
		cra_vvps_nogaps_alldoys=[]
		cra_halt_nogaps_alldoys=[]
		cra_speed_nogaps_alldoys=[]
		cra_fsr_nogaps_alldoys=[]
		cra_dcr_nogaps_alldoys=[]
		cra_tcr_nogaps_alldoys=[]
		cra_ssr_nogaps_alldoys=[]
		cra_hpr_nogaps_alldoys=[]
		cra_her_nogaps_alldoys=[]   
		
		#iterate over days to avoid large arrays and therefore speed calculation up
		utimes_cra_sorted=unique(self.times_cra_sorted)
		doys=arange(int(utimes_cra_sorted[0]),int(utimes_cra_sorted[-1])+1,1)
		print("doys sorted",doys)
		l=doys[0]
		while l<(doys[-1]+1):
			print("doy")
			print(l)
			
			#set doy (=day of year) mask
			#doymask_PHA=(self.times>=l)*(self.times<l+1)
			
			udoymask_cra_sorted=(utimes_cra_sorted>=l)*(utimes_cra_sorted<l+1)
			doymask_cra_sorted=(self.times_cra_sorted>=l)*(self.times_cra_sorted<l+1)
			
			utimes_cra_sorted_doy=utimes_cra_sorted[udoymask_cra_sorted]
			
			self.times_cra_sorted_doy=self.times_cra_sorted[doymask_cra_sorted]
			self.step_cra_sorted_doy=self.step_cra_sorted[doymask_cra_sorted]
			self.vvps_sorted_doy=self.vvps_sorted[doymask_cra_sorted]
			self.halt_sorted_doy=self.halt_sorted[doymask_cra_sorted]
			self.speed_sorted_doy=self.speed_sorted[doymask_cra_sorted]
			self.fsr_sorted_doy=self.fsr_sorted[doymask_cra_sorted]
			self.dcr_sorted_doy=self.dcr_sorted[doymask_cra_sorted]
			self.tcr_sorted_doy=self.tcr_sorted[doymask_cra_sorted]
			self.ssr_sorted_doy=self.ssr_sorted[doymask_cra_sorted]
			self.hpr_sorted_doy=self.hpr_sorted[doymask_cra_sorted]
			self.her_sorted_doy=self.her_sorted[doymask_cra_sorted]
			
			k = 0
			while k<len(utimes_cra_sorted_doy)-1:
			#while k < len(unique(self.times_cra_sorted))-1:
				cra_time_fill=arange(unique(self.times_cra_sorted_doy)[k],unique(self.times_cra_sorted_doy)[k+1],tol_day_cra)#close gaps with 5-minute time stamps
				if len(cra_time_fill)>1:
					ind_last=where(self.times_cra_sorted_doy==unique(self.times_cra_sorted_doy)[k])[0][-1]+1
					cra_times_nogaps_doy=insert(self.times_cra_sorted_doy,ind_last,cra_time_fill[1:])          
					cra_steps_nogaps_doy=insert(self.step_cra_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_vvps_nogaps_doy=insert(self.vvps_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_halt_nogaps_doy=insert(self.halt_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_speed_nogaps_doy=insert(self.speed_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_fsr_nogaps_doy=insert(self.fsr_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_dcr_nogaps_doy=insert(self.dcr_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_tcr_nogaps_doy=insert(self.tcr_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_ssr_nogaps_doy=insert(self.ssr_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_hpr_nogaps_doy=insert(self.hpr_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					cra_her_nogaps_doy=insert(self.her_sorted_doy,ind_last,array([-999]*len(cra_time_fill[1:])))
					k=k+len(cra_time_fill)+1
				else:
					cra_times_nogaps_doy=self.times_cra_sorted_doy
					cra_steps_nogaps_doy=self.step_cra_sorted_doy
					cra_vvps_nogaps_doy=self.vvps_sorted_doy
					cra_halt_nogaps_doy=self.halt_sorted_doy
					cra_speed_nogaps_doy=self.speed_sorted_doy
					cra_fsr_nogaps_doy=self.fsr_sorted_doy
					cra_dcr_nogaps_doy=self.dcr_sorted_doy
					cra_tcr_nogaps_doy=self.tcr_sorted_doy
					cra_ssr_nogaps_doy=self.ssr_sorted_doy
					cra_hpr_nogaps_doy=self.hpr_sorted_doy
					cra_her_nogaps_doy=self.her_sorted_doy
					k=k+1
			
			cra_times_nogaps_alldoys.append(cra_times_nogaps_doy)   
			cra_steps_nogaps_alldoys.append(cra_steps_nogaps_doy)
			cra_vvps_nogaps_alldoys.append(cra_vvps_nogaps_doy)   
			cra_halt_nogaps_alldoys.append(cra_halt_nogaps_doy)   
			cra_speed_nogaps_alldoys.append(cra_speed_nogaps_doy)
			cra_fsr_nogaps_alldoys.append(cra_fsr_nogaps_doy)   
			cra_dcr_nogaps_alldoys.append(cra_dcr_nogaps_doy)
			cra_tcr_nogaps_alldoys.append(cra_tcr_nogaps_doy)   
			cra_ssr_nogaps_alldoys.append(cra_ssr_nogaps_doy)
			cra_hpr_nogaps_alldoys.append(cra_hpr_nogaps_doy)
			cra_her_nogaps_alldoys.append(cra_her_nogaps_doy)   
			l=l+1

		cra_times_nogaps=concatenate(cra_times_nogaps_alldoys) 
		cra_steps_nogaps=concatenate(cra_steps_nogaps_alldoys) 
		cra_vvps_nogaps=concatenate(cra_vvps_nogaps_alldoys) 
		cra_halt_nogaps=concatenate(cra_halt_nogaps_alldoys) 
		cra_speed_nogaps=concatenate(cra_speed_nogaps_alldoys) 
		cra_fsr_nogaps=concatenate(cra_fsr_nogaps_alldoys) 
		cra_dcr_nogaps=concatenate(cra_dcr_nogaps_alldoys) 
		cra_tcr_nogaps=concatenate(cra_tcr_nogaps_alldoys) 
		cra_ssr_nogaps=concatenate(cra_ssr_nogaps_alldoys) 
		cra_hpr_nogaps=concatenate(cra_hpr_nogaps_alldoys) 
		cra_her_nogaps=concatenate(cra_her_nogaps_alldoys) 
		
		t1=clock()
		
		#shift pha times about half a timestamp to get rid of 1-minute shift effect in cra data
		pha_times_shifted=(unique(pha_times_nogaps[1:]+pha_times_nogaps[:-1]))/2.
	   
		#cut off cra data before firts pha timestamp to apply searchsorted for synchronizing cra times with pha times (see next step)   
		m=cra_times_nogaps>pha_times_shifted[0]
		cra_times_nogaps=cra_times_nogaps[m]
		cra_steps_nogaps=cra_steps_nogaps[m]
		cra_vvps_nogaps=cra_vvps_nogaps[m]
		cra_halt_nogaps=cra_halt_nogaps[m]
		cra_speed_nogaps=cra_speed_nogaps[m]
		cra_fsr_nogaps=cra_fsr_nogaps[m]
		cra_dcr_nogaps=cra_dcr_nogaps[m]
		cra_tcr_nogaps=cra_tcr_nogaps[m]
		cra_ssr_nogaps=cra_ssr_nogaps[m]
		cra_her_nogaps=cra_her_nogaps[m]
		
		#synchronizing cra times with pha times
		ss=searchsorted(pha_times_shifted,cra_times_nogaps)-1
		cra_times_sync=unique(pha_times_nogaps)[ss]#check fpr gaps or double assignments!
		utimes_cra_sync=unique(cra_times_sync)
		
		#return utimes_cra_sync, cra_times_sync
		#self.cra_times_sync_short=cra_times_sync
		
		t2=clock()        
	   
		#iterate over days to avoid large arrays and therefore speed calculation up        
		self.step_cra_sync_alldoys=[]
		self.vvps_sync_alldoys=[]
		self.halt_sync_alldoys=[]
		self.speed_sync_alldoys=[]
		self.fsr_sync_alldoys=[]
		self.dcr_sync_alldoys=[]
		self.tcr_sync_alldoys=[]
		self.ssr_sync_alldoys=[]
		self.hpr_sync_alldoys=[]
		self.her_sync_alldoys=[]
	   
		doys=arange(int(utimes_cra_sync[0]),int(utimes_cra_sync[-1])+1,1)
		j=doys[0]
		while j<(doys[-1]+1):
			print("doy")
			print(j)
			
			#set doy (=day of year) mask
			doymask_PHA=(self.times>=j)*(self.times<j+1)
			udoymask_cra=(utimes_cra_sync>=j)*(utimes_cra_sync<j+1)
			doymask_cra=(cra_times_sync>=j)*(cra_times_sync<j+1)
			utimes_cra_sync_doy=utimes_cra_sync[udoymask_cra]

			#introduce synchronized data product, still daily sliced! 
			self.step_cra_sync_doy=zeros((len(self.times[doymask_PHA])))-999
			self.vvps_sync_doy=zeros((len(self.times[doymask_PHA])))-999
			self.halt_sync_doy=zeros((len(self.times[doymask_PHA])))-999
			self.speed_sync_doy = zeros((len(self.times[doymask_PHA])))-999
			self.fsr_sync_doy = zeros((len(self.times[doymask_PHA])))-999
			self.dcr_sync_doy = zeros((len(self.times[doymask_PHA])))-999
			self.tcr_sync_doy = zeros((len(self.times[doymask_PHA])))-999
			self.ssr_sync_doy = zeros((len(self.times[doymask_PHA])))-999
			self.hpr_sync_doy = zeros((len(self.times[doymask_PHA])))-999
			self.her_sync_doy = zeros((len(self.times[doymask_PHA])))-999
		
			#slice CRA data daily
			cra_times_sync_doy=cra_times_sync[doymask_cra]
			cra_steps_nogaps_doy=cra_steps_nogaps[doymask_cra]   
			cra_vvps_nogaps_doy=cra_vvps_nogaps[doymask_cra]
			cra_halt_nogaps_doy=cra_halt_nogaps[doymask_cra]
			cra_speed_nogaps_doy=cra_speed_nogaps[doymask_cra]
			cra_fsr_nogaps_doy=cra_fsr_nogaps[doymask_cra]
			cra_dcr_nogaps_doy=cra_dcr_nogaps[doymask_cra]
			cra_tcr_nogaps_doy=cra_tcr_nogaps[doymask_cra]
			cra_ssr_nogaps_doy=cra_ssr_nogaps[doymask_cra]
			cra_hpr_nogaps_doy=cra_hpr_nogaps[doymask_cra]
			cra_her_nogaps_doy=cra_her_nogaps[doymask_cra]
						
			#slice PHA data daily
			self.times_doy=self.times[doymask_PHA]
			self.step_doy=self.step[doymask_PHA]
			
			#iterate over all timestamps within each doy
			i=0
			while i<len(utimes_cra_sync_doy): 
				
				print('timestamp')
				print(i, utimes_cra_sync_doy[i])
				
				#get indices of each time stamp to slice PHA and cra data per timestamp
				inds=self.times_doy==utimes_cra_sync_doy[i]
				inds_cra=cra_times_sync_doy==utimes_cra_sync_doy[i]
				
				if len(where(inds==True)[0])>0:#timesteps where no pha data is recorded is sorted out here, no need to include these timestamps anymore after timestamp synchronization is done!,noPHA
					
					if min(cra_steps_nogaps_doy[inds_cra])!=-999:#no dummy value allowed (since up to here: one value==dummy => all values==dummy)., no cra data available here
						
						test_variable=0# test variable needed because minimum of steps_cra_sliced can change (see below)!
						
						#timestamp-slice pha data
						times_pha_sliced=self.times_doy[inds]
						steps_pha_sliced=self.step_doy[inds]
						
						#timestamp-slice cra data
						steps_cra_sliced=cra_steps_nogaps_doy[inds_cra]
						vvps_sliced=cra_vvps_nogaps_doy[inds_cra]
						halt_sliced=cra_halt_nogaps_doy[inds_cra]
						speed_sliced=cra_speed_nogaps_doy[inds_cra]
						fsr_sliced=cra_fsr_nogaps_doy[inds_cra]
						dcr_sliced=cra_dcr_nogaps_doy[inds_cra]
						tcr_sliced=cra_tcr_nogaps_doy[inds_cra]
						ssr_sliced=cra_ssr_nogaps_doy[inds_cra]
						hpr_sliced=cra_hpr_nogaps_doy[inds_cra]
						her_sliced=cra_her_nogaps_doy[inds_cra]
						
						if min(steps_cra_sliced)>min(steps_pha_sliced):#set those cra data products to dummy value where no cra_step data exists (lower steps)
							steps_cra_sliced=concatenate([array([min(steps_cra_sliced)-1]),steps_cra_sliced])
							test_variable=1
						
						ss_steps=searchsorted(steps_cra_sliced,steps_pha_sliced)
						
						if test_variable==1:# test variable needed because minimum of steps_cra_sliced has changed!
							#print "True Test"
							steps_cra_sliced[0]=-999
							vvps_sliced=concatenate([array([-999]),vvps_sliced])
							halt_sliced=concatenate([array([-999]),halt_sliced])
							speed_sliced=concatenate([array([-999]),speed_sliced])
							fsr_sliced=concatenate([array([-999]),fsr_sliced])
							dcr_sliced=concatenate([array([-999]),dcr_sliced])
							tcr_sliced=concatenate([array([-999]),tcr_sliced])
							ssr_sliced=concatenate([array([-999]),ssr_sliced])
							hpr_sliced=concatenate([array([-999]),hpr_sliced])
							her_sliced=concatenate([array([-999]),her_sliced])

						
						if max(steps_cra_sliced)<max(steps_pha_sliced):#set those cra data products to dummy value where no cra_step data exists (higher steps)
							#print "True Test"
							steps_cra_sliced=concatenate([steps_cra_sliced,array([-999])])
							vvps_sliced=concatenate([vvps_sliced,array([-999])])
							halt_sliced=concatenate([halt_sliced,array([-999])])
							speed_sliced=concatenate([speed_sliced,array([-999])])
							fsr_sliced=concatenate([fsr_sliced,array([-999])])
							dcr_sliced=concatenate([dcr_sliced,array([-999])])
							tcr_sliced=concatenate([tcr_sliced,array([-999])])
							ssr_sliced=concatenate([ssr_sliced,array([-999])])
							hpr_sliced=concatenate([hpr_sliced,array([-999])])
							her_sliced=concatenate([her_sliced,array([-999])])
							

						self.step_cra_sync_doy[inds]=steps_cra_sliced[ss_steps]
						self.vvps_sync_doy[inds]=vvps_sliced[ss_steps]
						self.halt_sync_doy[inds]=halt_sliced[ss_steps]
						self.speed_sync_doy[inds]=speed_sliced[ss_steps]
						self.fsr_sync_doy[inds]=fsr_sliced[ss_steps]
						self.dcr_sync_doy[inds]=dcr_sliced[ss_steps]
						self.tcr_sync_doy[inds]=tcr_sliced[ss_steps]
						self.ssr_sync_doy[inds]=ssr_sliced[ss_steps]
						self.hpr_sync_doy[inds]=hpr_sliced[ss_steps]
						self.her_sync_doy[inds]=her_sliced[ss_steps]
						
					else:#dummy value sequence is inserted for cra data values if cra data is missing
						self.step_cra_sync_doy[inds]=array([-999]*len(ss_steps))
						self.vvps_sync_doy[inds]=array([-999]*len(ss_steps))
						self.halt_sync_doy[inds]=array([-999]*len(ss_steps))
						self.speed_sync_doy[inds]=array([-999]*len(ss_steps))
						self.fsr_sync_doy[inds]=array([-999]*len(ss_steps))
						self.dcr_sync_doy[inds]=array([-999]*len(ss_steps))
						self.tcr_sync_doy[inds]=array([-999]*len(ss_steps))
						self.ssr_sync_doy[inds]=array([-999]*len(ss_steps))
						self.hpr_sync_doy[inds]=array([-999]*len(ss_steps))
						self.her_sync_doy[inds]=array([-999]*len(ss_steps))
					
				else:#nothing is done if PHA data is missing
					print('timestamp',i, utimes_cra_sync_doy[i])
					print('no PHA data measured in this timestamp')
					
				i=i+1
						
			self.step_cra_sync_alldoys.append(self.step_cra_sync_doy)            
			self.vvps_sync_alldoys.append(self.vvps_sync_doy)
			self.halt_sync_alldoys.append(self.halt_sync_doy)
			self.speed_sync_alldoys.append(self.speed_sync_doy)
			self.fsr_sync_alldoys.append(self.fsr_sync_doy)
			self.dcr_sync_alldoys.append(self.dcr_sync_doy)
			self.tcr_sync_alldoys.append(self.tcr_sync_doy)
			self.ssr_sync_alldoys.append(self.ssr_sync_doy)
			self.hpr_sync_alldoys.append(self.hpr_sync_doy)
			self.her_sync_alldoys.append(self.her_sync_doy)
			j=j+1
			
		self.step_cra_sync=concatenate(self.step_cra_sync_alldoys)    
		self.vvps_sync=concatenate(self.vvps_sync_alldoys)    
		self.halt_sync=concatenate(self.halt_sync_alldoys)    
		self.speed_sync=concatenate(self.speed_sync_alldoys)    
		self.fsr_sync=concatenate(self.fsr_sync_alldoys)    
		self.dcr_sync=concatenate(self.dcr_sync_alldoys)    
		self.tcr_sync=concatenate(self.tcr_sync_alldoys)    
		self.ssr_sync=concatenate(self.ssr_sync_alldoys)    
		self.hpr_sync=concatenate(self.hpr_sync_alldoys)    
		self.her_sync=concatenate(self.her_sync_alldoys)    
		   
		t3=clock()
		
		print("synchronization method runtimes:")
		print("total method runtime:", t3-t0)
		print(t1-t0)
		print(t2-t1)
		print(t3-t2)
		print("PM data synchronized with CTOF PHA and CRA data")       
		
		return t_check-t0,t1-t_check,t2-t1,t3-t2
		
		#return True#
		
		
		
		
		
		
		
		
		
		
		
		
		
	def prepare_CTOFdata(self,days=arange(174,176,1),mbounds="upplow_exc"):
		t0=clock()
		#self.load_CTOFfull()
		#print "PHA_loaded"
		t1=clock()
		#self.load_CTOFcra()
		#print "CRA_loaded"
		t2=clock()
		#self.scale_cradata()
		#print "CRA_scaled"
		t3=clock()
		#self.sort_cradata()
		#print "CRA_sorted"
		t4=clock()
		self.load_CTOFMR()
		print("CMR_loaded")
		t5=clock()
		#self.sync_CTOF_CMR()
		#print "CMR synchronized with CTOF PHA"
		#return True
		
		pmdat=self.load_PM()
		self.get_vsw_quick()
		print("PM_loaded")
		t6=clock()
		
		self.load_cmr_heftical()
		self.sync_mrdata()
		self.clean_mrdata()
		
		self.create_PRcounts_PHA(savedata=False)
		self._get_centerstep_shift()
		self.calc_centerstep_hefti(round_step=False)
		#self.get_epq_rates()
		self.set_mrbox_PRmask()
		#self.get_epq_rates_improved()
		self.get_epq_rates_PRdaywise(days)
		self.get_brf()
		self.get_baserates()
		print("data preparation finished")
		t7=clock()
		print("cmr data loading time:", t5-t4)
		print("pm loading time", t6-t5)
		print("data preparation time:", t7-t6)
		print("total time:",t7-t0)
		return True
	
		
	def plot_mrhists(self,mr_box,timestamp):#timestamp in doy,mrbox=matrix box number 0,1,2...,508
		
		mr_channels=arange(0,21,1)
		mr_mask=(self.number_cmr_sync==mr_box)*(self.times_cmr_sync==timestamp)
		mr_counts=zeros((max(mr_channels)+1))
		print(len(mr_counts))
		
		i=0
		while i<=max(mr_channels):
			print(i,max(mr_channels))
			print(len(mr_mask),len(self.matrix_rates[i]))
			try:
				mr_counts[i]=self.matrix_rates[i][mr_mask]
			except ValueError: 
				print("value error: following sequence should have length 1")
				return self.matrix_rates[i][mr_mask]
				
			i+=1
			
		plt.figure()
		plt.bar(mr_channels,mr_counts,width=1,color='b')
		plt.axis([0,max(mr_channels),0,1.1*max(mr_counts)])
		plt.show()

	def _add_mmpq(self):
		self.mpq=self.tof_to_mpq_hefti(tof=self.tof,step=self.step)#tof in ch, 
		self.m=self.tofE_to_m_hefti(tof=self.tof,ESSD=self.energy)#tof in ch, E_SSD in ch 
		self.mpq = self.mpq.astype(int)
		self.m = self.m.astype(int)

	###########new base rate factor reconstruction (together with Lars) sterts here!
	def plot_mmpq(self,PR=None):	
		PR_mask=(self.range==PR)
		x=arange(0,150,1)
		y=arange(0,150,1)
		if PR!=None:
			h,bx,by=histogram2d(self.mpq[PR_mask],self.m[PR_mask],[x,y])
		else:
			h,bx,by=histogram2d(self.mpq,self.m,[x,y])
		fig, ax = plt.subplots(1,1)
		p=ax.pcolormesh(bx,by,h.T,vmin=1.)
		ax.set_xlabel("mpq [ch]")
		ax.set_ylabel("m [ch]")
		p.cmap.set_under('w')

		plt.show()


	def create_PRcounts_PHA(self,savedata=False):

		utimes=unique(self.times)
		timestamp_extend=utimes[-1]+5./(24*3600)#add artifcial timestamp for histogramming
		utimes_extend=concatenate([utimes,array([timestamp_extend])])
		steps_extend=arange(0,117,1)
		pranges_extend=arange(0,7,1)
		bins=array([pranges_extend, utimes_extend, steps_extend])		
		
		#masks result from mpq- and m-onboard algorithm and mrbox definition
		intof=(self.tof>=90)*(self.tof<=1022)
		inE=(self.energy<=254)#too low energies (mostly) land in regular boxes
		inmpq=(self.mpq>5)*(self.mpq<80)
		inm=(self.m<61)
		inmask=(intof*inE*inmpq*inm)
		data_PHA=array([self.range[inmask],self.times[inmask],self.step[inmask]])
		#data_PHA=array([self.range,self.times,self.step])
		PRcounts_PHA,Bins=histogramdd(data_PHA.T, bins)
		
		if savedata==True:
			path="baserate_factors_new/"
			i=0
			while i<6:
				PRcounts_PHA_flat=ravel(PRcounts_PHA[i])
				steps_grid,times_grid=meshgrid(steps_extend[:-1],utimes)
				times_flat,steps_flat=ravel(times_grid),ravel(steps_grid)
				#return PRcounts_PHA[i],PRcounts_PHA_flat,times_flat,steps_flat
				outdata=array([times_flat,steps_flat,PRcounts_PHA_flat])
				outfile="PRcounts_PHA_PR%i"%(i)	
				with open(path+outfile, 'wb') as f:
					f.write("timestamp [DOY 1996]	E/q-step\n")
					savetxt(f, outdata.T, fmt='%.5f', delimiter=' ', newline='\n')
					f.close()	
				i=i+1

		self.PHA_PR=PRcounts_PHA
		return 


	def load_PRcounts_PHA(self,PR=1,path="baserate_factors_new/"):
		infile="PRcounts_PHA_PR%i"%(PR)
		PRcounts_PHA=loadtxt(path+infile,skiprows=1,unpack=True)
		return PRcounts_PHA


	def sync_mrdata(self):
		usecs_pha=unique(self.secs)
		usecs_mr=unique(self.secs_mr)
		usecs = unique(append(usecs_pha,usecs_mr))
		
		uimr = searchsorted(usecs,usecs_mr)-1
		uipha = searchsorted(usecs,usecs_pha,side="right")-1
		uiints = intersect1d(uimr,uipha)		
		dtusecs = append(diff(usecs),0)
		uiint_mask = (dtusecs[uiints]>285)*(dtusecs[uiints]<289)#nominal time shift between pha data and mr data is 1 CTOF cycle-15 secs =287 secs 
		
		vuitimes = uiints[uiint_mask]
		vuphatimes = usecs[vuitimes]
		vumrtimes = usecs[vuitimes+1]
		umr_mask=in1d(usecs_mr,vumrtimes)		
		upha_mask=in1d(usecs_pha,vuphatimes)
		fmr_mask=in1d(self.secs_mr,vumrtimes)		
		fpha_mask=in1d(self.secs,vuphatimes)
		
		self.utimes=self.times[upha_mask]
		self.times=self.times[fpha_mask]
		self.secs=self.secs[fpha_mask]
		self.range=self.range[fpha_mask]
		self.energy=self.energy[fpha_mask]
		self.tof=self.tof[fpha_mask]
		self.step=self.step[fpha_mask]
		self.mpq=self.mpq[fpha_mask]
		self.m=self.m[fpha_mask]
		
		self.vsw=self.vsw[fpha_mask]		
		self.vth=self.vth[fpha_mask]		
		self.dsw=self.dsw[fpha_mask]	
		
		self.times_mr=self.times_mr[fmr_mask]
		self.secs_mr=self.secs_mr[fmr_mask]
		self.mr_number=self.mr_number[fmr_mask]
		self.vFe_mr=self.vsw_mr[fmr_mask]#vsw_mr is actually onboard estimated iron speed
		self.MR=self.MR[umr_mask]#this is the actual matrix rate count data
		
		#just for testing, should be deleted later
		self.umr_mask=umr_mask	
		self.upha_mask=upha_mask
		self.fmr_mask=fmr_mask		
		self.fpha_mask=fpha_mask
		return True
						
	def clean_mrdata(self):					
		self.MRA=zeros((self.MR.shape[0],self.MR.shape[1],self.MR.shape[2]+1))
		boxmask=zeros((shape(self.MR))).astype(bool)
		boxmask_1d=(self.mrbox_shift!=-1)
		boxmask[:,:,]=boxmask_1d
		self.MRC=self.MR*boxmask#clean matrix rates, with counts only in the identfied boxes (rest boxes: counts=0) 
		self.MRC[self.MRC==-1]=0#storage failure in these boxes, happens rarely (2% of timestamps) 
		self.MRC[self.MRC==-2]=0#-2 is used as dummy value if matrix rate cannot be filled because the central step is too high or low, so 0 is the correct count number for these cases
		 
		MRR=self.MR[invert(boxmask)]#counts in rest boxes, that are not identified in the Hefti matrix rate definition scheme 	
		nR=(self.mrbox_shift[invert(boxmask_1d)]).size
		MRR=MRR.reshape(self.MR.shape[0],self.MR.shape[1],nR)
		MRR[MRR==-1]=0
		MRR[MRR==-2]=0
		self.MRR=sum(MRR,axis=2)
		self.MRA[:,:,:-1]=self.MRC
		self.MRA[:,:,-1]=self.MRR#all counts in the rest boxes are in the last box with box-index i=508 now!
		 
		 
	def _get_centerstep_shift(self,bn=None):
		if bn == None:
			self.centersteps_shift = self.mrbox_shift[self.mrbox_number].astype(int)
			return self.centersteps_shift
		else:
			return self.mrbox_shift[bn]
			
		
	def calc_centerstep_hefti(self,v_Fe=None,round_step=False,cs_shift=None):
		"""
		we found out from comparison of the "-2" dummy value in the matrix rates, that the centerstep shift is rounded and not truncated, thus the False flag is correct.
		"""
		bbVpMult=49.8516
		VpOff=-164
		if v_Fe == None:
				self.uT,self.iT,self.riT = unique(self.times_mr,return_index=True,return_inverse=True)
				v_Fe = self.vFe_mr[self.iT]
				#v_Fe=self.vFe_mr.reshape(len(self.vFe_mr)/508,508).T[0]
				#return v_Fe_old,v_Fe
				
				self.centersteps = zeros((v_Fe.size,self.centersteps_shift.size),dtype=int)
				for i,v in enumerate(v_Fe):
					if round_step==True:
						self.centersteps[i] = 116-(int(log(v_Fe[i])*bbVpMult+0.5)+VpOff-(self.centersteps_shift+1)/2)	
					else:
						self.centersteps[i] = 116-(int(log(v_Fe[i])*bbVpMult)+VpOff-(self.centersteps_shift+1)/2)	
		else:
			cs=116-(int(log(v_Fe)*bbVpMult)+VpOff-(cs_shift+1)/2)
		
			return v_Fe, cs_shift, cs 
	
	def get_epq_rates(self):#old
		self.mrshifts = [range(-116,-32),[-32,-31,-30,-29,-28,-27,-26,-25],[-24,-23,-22,-21,-20,-19,-18,-17],[-16,-15,-14,-13],[-12,-11,-10,-9],[-7,-8],[-5,-6],[-3,-4],[-2],[-1],[0],[1],[2],[3,4],[5,6],[7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20,21,22,23,24],[25,26,27,28,29,30,31,32],range(33,116)]
		self.epq_rates = zeros((self.centersteps.shape[0],116, self.centersteps.shape[1]))
		self.iB = array(range(508)*self.uT.shape[0])
		self.iCEpq = self.centersteps[self.riT,self.iB]
		for M in range(21):
			iM = ones(self.iB.size,dtype=int)*M
			norm = zeros(self.iB.size,dtype=float64)
			for s in self.mrshifts[M]:
				iEpq = self.iCEpq + s
				mask = ((iEpq>=0)*(iEpq<116))
				norm[mask]+=1
			for s in self.mrshifts[M]:
				iEpq = self.iCEpq + s
				mask = ((iEpq>=0)*(iEpq<116))
				#self.epq_rates[self.riT[mask],iEpq[mask],self.iB[mask]]+=self.MR[self.riT[mask],iM[mask],self.iB[mask]]/norm[mask]#old, not cleaned matrix rates still in it
				self.epq_rates[self.riT[mask],iEpq[mask],self.iB[mask]]+=self.MRC[self.riT[mask],iM[mask],self.iB[mask]]/norm[mask]



	def get_epq_rates_improved(self):#old
		self.mrshifts = [range(-116,-32),[-32,-31,-30,-29,-28,-27,-26,-25],[-24,-23,-22,-21,-20,-19,-18,-17],[-16,-15,-14,-13],[-12,-11,-10,-9],[-7,-8],[-5,-6],[-3,-4],[-2],[-1],[0],[1],[2],[3,4],[5,6],[7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20,21,22,23,24],[25,26,27,28,29,30,31,32],range(33,116)]
		
		self.epq_rates = zeros((self.centersteps.shape[0],116, self.centersteps.shape[1]))
		self.iB = array(range(508)*self.uT.shape[0])
		
		mask_pha=(sum(self.PHA_PR,axis=0)[self.riT]).astype(bool)
		self.iCEpq = self.centersteps[self.riT,self.iB]
		for M in range(21):
			iM = ones(self.iB.size,dtype=int)*M
			norm = zeros(self.iB.size,dtype=float64)
			for s in self.mrshifts[M]:
				iEpq = self.iCEpq + s
				mask_mr = ((iEpq>=0)*(iEpq<116))
				iEpq_indval=1*iEpq
				iEpq_indval[iEpq_indval>115]=0#is ok since mpha is multiplied with mask_mr later, anyways 
				iEpq_indval[iEpq_indval<0]=0#is ok since mpha is multiplied with mask_mr later, anyways
				mm=(arange(len(iEpq_indval)),iEpq_indval)
				mpha=mask_pha[mm]		
				mask=mask_mr*mpha
				norm[mask]+=1
				
				
			for s in self.mrshifts[M]:
				iEpq = self.iCEpq + s
				mask_mr = ((iEpq>=0)*(iEpq<116))
				iEpq_indval=1*iEpq
				iEpq_indval[iEpq_indval>115]=0#is ok since mpha is multiplied with mask_mr later, anyways 
				iEpq_indval[iEpq_indval<0]=0#is ok since mpha is multiplied with mask_mr later, anyways
				mm=(arange(len(iEpq_indval)),iEpq_indval)
				mpha=mask_pha[mm]		
				mask=mask_mr*mpha
				
				self.epq_rates[self.riT[mask],iEpq[mask],self.iB[mask]]+=self.MRC[self.riT[mask],iM[mask],self.iB[mask]]/norm[mask]
				


	def get_epq_rates_PRwise(self,uT,centersteps,MRC,PHA_PR,sumPR=True):#should be used via  get_epq_rates_PRdaywise
		self.mrshifts = [range(-116,-32),[-32,-31,-30,-29,-28,-27,-26,-25],[-24,-23,-22,-21,-20,-19,-18,-17],[-16,-15,-14,-13],[-12,-11,-10,-9],[-7,-8],[-5,-6],[-3,-4],[-2],[-1],[0],[1],[2],[3,4],[5,6],[7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20,21,22,23,24],[25,26,27,28,29,30,31,32],range(33,116)]
		epq_rates = zeros((6,centersteps.shape[0],116, centersteps.shape[1]))
		iB = array(range(508)*uT.shape[0])
		riT=ravel(array([arange(len(uT))]*508).T)
		
		
		pranges=[0,1,2,3,4,5]
		for p in pranges:
			print("Priority Range:", p)
			mask_pha=PHA_PR[p][riT].astype(bool)
			
			iCEpq = centersteps[riT,iB]
			for M in range(21):
				iM = ones(iB.size,dtype=int)*M
				norm = zeros(iB.size,dtype=float64)
				for s in self.mrshifts[M]:
					iEpq = iCEpq + s
					mask_mr = ((iEpq>=0)*(iEpq<116))
					iEpq_indval=1*iEpq
					iEpq_indval[iEpq_indval>115]=0#is ok since mpha is multiplied with mask_mr later, anyways 
					iEpq_indval[iEpq_indval<0]=0#is ok since mpha is multiplied with mask_mr later, anyways
					mm=(arange(len(iEpq_indval)),iEpq_indval)
					mpha=mask_pha[mm]		
					mask=mask_mr*mpha
					norm[mask]+=1.
				for s in self.mrshifts[M]:
					iEpq = iCEpq + s
					mask_mr = ((iEpq>=0)*(iEpq<116))
					iEpq_indval=1*iEpq
					iEpq_indval[iEpq_indval>115]=0#is ok since mpha is multiplied with mask_mr later, anyways 
					iEpq_indval[iEpq_indval<0]=0#is ok since mpha is multiplied with mask_mr later, anyways
					mm=(arange(len(iEpq_indval)),iEpq_indval)
					mpha=mask_pha[mm]		
					mask=mask_mr*mpha
					
					epq_rates[p][riT[mask],iEpq[mask],iB[mask]]+=(MRC*self.mrbox_prmask[p])[riT[mask],iM[mask],iB[mask]]/(norm[mask])
		
		if sumPR==True:
			epq_rates=sum(epq_rates,axis=0)
		return epq_rates	


	def get_epq_rates_PRdaywise(self,days):#days can have any arbitrary distance,but must have len=2 at least!
		days=concatenate([days,array([days[-1]+1])])
		t0=clock()
		epq_rates=[]
		i=0
		while i<(len(days)-1):	
			t00=clock()
			udmask=(self.uT>=days[i])*(self.uT<(days[i+1]))
			uT=self.uT[udmask]
			MRC=self.MRC[udmask]
			PHA_PR=(self.PHA_PR.transpose(1,0,2))[udmask].transpose(1,0,2)
			centersteps=self.centersteps[udmask]
			epq_rates_day=self.get_epq_rates_PRwise(uT,centersteps,MRC,PHA_PR,sumPR=True)#improve here!
			epq_rates.append(epq_rates_day)
			i=i+1
			t01=clock()
			print("calc. time for day %i: %.2f seconds"%(days[i],t01-t00))
			
		#return epq_rates 	
		self.epq_rates=concatenate(epq_rates)
		t1=clock()
		print("total calc. time: %.2f seconds"%(t1-t0))


			
	def set_mrbox_PRmask(self):		
			pranges=[0,1,2,3,4,5]
			PRmasks=zeros((len(pranges),len(self.mrbox_number)))#508 mrboxes * 6 Priority Ranges
			i=0 
			for prange in pranges:
				PRmask=zeros((len(self.mrbox_number)))
				if prange==0:
					mlim_low=0#check: also DC data also included here
					mlim_up=7
				elif prange==1:
					mlim_low=52
					mlim_up=61
				elif prange==2:
					mlim_low=47
					mlim_up=52
				elif prange==3:
					mlim_low=41
					mlim_up=47
				elif prange==4:
					mlim_low=33
					mlim_up=41
				elif prange==5:
					mlim_low=7
					mlim_up=33
					
				for mrbox in self.mrbox_number:
				#for mrbox in boxes:
					
					#full boxes
					if self.mrbox_m_ll[mrbox]>=mlim_low and self.mrbox_m_ur[mrbox]<=mlim_up:#PR only depends on onboard calculated mass channel
						fracin=1.
						PRmask[mrbox]=fracin
					#treat boxes that lie in two PRs (fulfilled condition: no (identified) boxes exist that lie in more than two PRs)	
					elif self.mrbox_m_ll[mrbox]<mlim_up and self.mrbox_m_ur[mrbox]>mlim_up:
						chin=mlim_up-self.mrbox_m_ll[mrbox]
						chbox=float(self.mrbox_m_ur[mrbox]-self.mrbox_m_ll[mrbox])
						fracin=chin/chbox
						PRmask[mrbox]=fracin
					elif self.mrbox_m_ll[mrbox]<mlim_low and self.mrbox_m_ur[mrbox]>mlim_low:
						chin=self.mrbox_m_ur[mrbox]-mlim_low
						chbox=float(self.mrbox_m_ur[mrbox]-self.mrbox_m_ll[mrbox])
						fracin=chin/chbox
						PRmask[mrbox]=fracin
					else:
						fracin=0.
						PRmask[mrbox]=fracin
				PRmasks[i]=PRmask
				i=i+1		
			self.mrbox_prmask=PRmasks
	
	
		
	def get_brf_from_saveddata(self,timerange=[174,220],path="baserate_factors_new/"):#timerange only for testing, PHA data loading outsourcen	
		pranges=[0,1,2,3,4,5]
		cphas=[]
		cepqs=[]
		brfs=[]
		for i in pranges:
			print("calc brf for PR:", i)
			
			#get PHA PR-counts
			p=self.load_PRcounts_PHA(PR=i,path=path)
			ptime=p[0]
			upsteps=unique(p[1])
			pcounts=p[2]
			timemask=(ptime>=timerange[0])*(ptime<timerange[-1])
			cpha=pcounts[timemask].reshape(len(pcounts[timemask])/len(upsteps),len(upsteps))#116=len(epq-steps)
			cpha=cpha[self.upha_mask,:]
			cpha[cpha==0]=1.
			
			#get MR PR-counts
			cepq=sum(self.epq_rates[:,:,self.mrbox_prmask[i]],axis=2)
			cepq=cepq[self.umr_mask,:]
			#if i==1:
			#	return cpha,cepq
			
			#calculate base rate factors
			brf=cepq/cpha.astype(float)
			brfs.append(brf)
			cphas.append(cpha)	
			cepqs.append(cepq)
		self.brfs=array(brfs)	
		self.cphas=array(cphas)
		self.cepqs=array(cepqs)
		
	
	def get_brf(self,timerange=[174,220],path="baserate_factors_new/",minbrf=False):#timerange only for testing, PHA data loading outsourcen	
		pranges=[0,1,2,3,4,5]
		cphas=[]
		cepqs=[]
		brfs=[]
		for i in pranges:
			#get PHA PR-counts
			cpha=self.PHA_PR[i]
			cpha_denom=cpha*1.
			cpha_denom[cpha_denom==0]=1.
			
			#get MR PR-counts
			cepq=sum(self.epq_rates[:,:,]*self.mrbox_prmask[i],axis=2)
			#cepq=sum(self.epq_rates[i][:,:,]*self.mrbox_prmask[i],axis=2)#use if sumPR==False
			
			
			#calculate base rate factors
			brf=cepq/cpha_denom.astype(float)
			if minbrf==True:
				brf[(brf>0)*(brf<1)]=1.
			brfs.append(brf)
			cphas.append(cpha)	
			cepqs.append(cepq)
		self.brfs=array(brfs)	
		self.cphas=array(cphas)
		self.cepqs=array(cepqs)
		return True
	
	def plot_brfs(self,PR,utime_ind=0,Xlims=[0,116.01],Ylim_factor=1.1,figx=6.8,figy=7.,fontsize=32,ticklabelsize=20,marker_lw=2, cycle=0,ytitle_pos=1.,legsize=25,title_fix=False,save_figure=False,figpath="/home/asterix/janitzek/ctof/achapter2_finalplots/brcor/",figname="test",PHA_color="gray",MR_color="r"):
		
		y1=self.cphas[PR][utime_ind]
		y2=self.cepqs[PR][utime_ind]
		y2[y1==0]=0#prelim!
		fig,ax=plt.subplots(1,1,figsize=(figx, figy))
		ax.bar(arange(0,116,1),y1,width=1.,alpha=0.5,color=PHA_color,label=r"$\rm{PHA \ counts}$")
		ax.bar(arange(0,116,1),y2,width=1.,alpha=0.5,color=MR_color,label=r"$\rm{MR \ counts}$")
		ax.legend(loc="upper left",prop={'size': legsize})
		ax.set_xlabel(r"$\rm{E}/\rm{q-step}$",fontsize=fontsize)
		ax.set_ylabel(r"$\rm{counts \ per \ step}$",fontsize=fontsize)
		ax.set_title(r"$\rm{PR: \ \ %i \ , \ CTOF-Cycle: \ 174\ -\ %i}$"%(PR,utime_ind),fontsize=fontsize,y=1.01)
		if title_fix:
			#ax.set_title(r"$ \rm{DOY: \ %i \ 1996, \ CTOF-Cycle: \ %i, \ E/q-step:\ %i}$"%(timerange[0],cycle,step),fontsize=fontsize,y=1.01)
			pass
		ax.tick_params(axis="x", labelsize=ticklabelsize)
		ax.tick_params(axis="y", labelsize=ticklabelsize)
		ax.set_xlim(Xlims[0],Xlims[1])
		ax.set_ylim(0,Ylim_factor*max(y2))
		
		if save_figure==True:
			plt.savefig(figpath+figname,bbox_inches='tight')
		else:
			plt.show()	


	def pcmask_step(self,step,q12=90,q23=90,q34=90,q45=90):
		"""
		q=percentile
		used mask: m=d.pcmask_step(20,q12=95,q23=85,q34=95,q45=90)
		only works reliable for steps 30-70, find out why!
		""" 
		brf=self.brfs*1.
		brf[brf<1.]=1.#see above
		p12=percentile(brf[2]/brf[1].astype(float),q=q12,axis=0)[step]
		p23=percentile(brf[3]/brf[2].astype(float),q=q23,axis=0)[step]
		p34=percentile(brf[4]/brf[3].astype(float),q=q34,axis=0)[step]	
		p45=percentile(brf[5]/brf[4].astype(float),q=q45,axis=0)[step]

		r12=(brf[2,:,step]/brf[1,:,step].astype(float))
		r23=(brf[3,:,step]/brf[2,:,step].astype(float))
		r34=(brf[4,:,step]/brf[3,:,step].astype(float))
		r45=(brf[5,:,step]/brf[4,:,step].astype(float))
		
		mr12=mean(r12)
		mr23=mean(r23)
		mr34=mean(r34)
		mr45=mean(r45)
		
		qfactor_mr12=q12/mr12
		qfactor_mr23=q23/mr23
		qfactor_mr34=q34/mr34
		qfactor_mr45=q45/mr45
		
		print("PR_12: mean,qfactor=",mr12,qfactor_mr12)
		print("PR_23: mean,qfactor=",mr23,qfactor_mr23) 
		print("PR_34: mean,qfactor=",mr34,qfactor_mr34) 
		print("PR_45: mean,qfactor=",mr45,qfactor_mr45) 
		
		#return r12,r23,r34,r45,p12,p23,p34,p45
		
		m1=(r12<=p12)*(r23<=p23)*(r34<=p34)*(r45<p45)#take also other direction! 
		#return m1 
		utimes=unique(self.times)
		m2=in1d(self.times,utimes[m1])
		mask=(self.step==step)*m2
		return mask


	def pcmask_absbrf(self,step,q0=100,q1=95,q2=95,q3=95,q4=95,q5=95):
		"""
		q=percentile
		q0=100 recommended when PR0 is not of interest
		used mask for reference plots: m=d.pcmask_absbrf(step=90,q1=90,q2=90,q3=70,q4=90,q5=90)
		""" 
		brf=self.brfs*1.
		#brf[brf<1.]=1.#see above
		b0=brf[0,:,step]
		b1=brf[1,:,step]
		b2=brf[2,:,step]
		b3=brf[3,:,step]
		b4=brf[4,:,step]
		b5=brf[5,:,step]
		
		"""
		p0=percentile(b0[b0>1],q=q0)
		p1=percentile(b1[b1>1],q=q1)
		p2=percentile(b2[b2>1],q=q2)
		p3=percentile(b3[b3>1],q=q3)
		p4=percentile(b4[b4>1],q=q4)
		p5=percentile(b5[b5>1],q=q5)
		"""
		p0=percentile(b0,q=q0)
		p1=percentile(b1,q=q1)
		p2=percentile(b2,q=q2)
		p3=percentile(b3,q=q3)
		p4=percentile(b4,q=q4)
		p5=percentile(b5,q=q5)
		print(p0,p1,p2,p3,p4,p5)
		
		m=(brf[0,:,step]<p0)*(brf[1,:,step]<p1)*(brf[2,:,step]<p2)*(brf[3,:,step]<p3)*(brf[4,:,step]<p4)*(brf[5,:,step]<p5)
		utimes=unique(self.times)
		mask=in1d(self.times,utimes[m])
		return mask


	
	def get_baserates(self):
		self.range[self.range==6]=0#work around, since PR=6 only happens about 10 times
		pha=array([self.range,self.times,self.step])
		ut,it=unique(self.times,return_inverse=True)
		self.baserate=self.brfs[pha[0].astype(int),it.astype(int),pha[2].astype(int)]
		return True
			
	def calc_velmoments(self,m,q,mrboxes):
		#replace mr box by list of mrboxes, introduce phase space correction
		#calculate other moments
		mrbox = list(mrboxes)
		steps=arange(0,116,1)
		vels=list(step_to_vel(steps,q_e=q,m_amu=m))	
		vels*=self.MR.shape[0]
		vels=array(vels)
		vels=vels.reshape((self.MR.shape[0],len(steps)))
		counts = self.epq_rates[:,:,mrbox]
		counts = counts.sum(axis=2)
		counts[counts<0]=0#should not be necessary if get_epqrates works correctly(!)
		norms=counts.sum(axis=1).astype("float")
		norms[norms==0]=1.
		vmeans=sum((counts*vels),axis=1)/norms
		
		#phase space corrected counts (combined correction for 1D-velocity and -position space)
		counts_pscor2=1./vels**2*counts
		norms_pscor2=counts_pscor2.sum(axis=1).astype("float")
		norms_pscor2[norms_pscor2==0]=1.
		vmeans_pscor2=sum((counts_pscor2*vels),axis=1)/norms_pscor2
		
		return vmeans,vmeans_pscor2



















	def compare_PHA_TCR(self,step,filter_products=[],countrange=arange(0,100,1)):#method will be shifted to CTOF_paramfit
		
		t0=clock()
		#tcr_true=2.5*self.tcr_sync
		m_step=self.step_cra_sync==step
		m_energy=self.energy>0
		    
		
		#Histogram PHA and TCR counts
		
		#PHA: all times are histogrammed, from this we get the number of PHA counts h0 at each CTOF cycle, which is then histogrammed over the countrange
		timebins=concatenate([unique(self.times[m_step*m_energy]),array([368,369])])#368,369 are days that are never reached in a year and therefore upper boundary of the histogram
		h=histogram(self.times[m_step*m_energy],timebins)
		#return h
		h0=h[0][:-1]
		
		
		
		plt.figure()
		H_PHA=plt.hist(h0,countrange,alpha=1.0,color="b",label="PHA")
		
		print('sum(N_PHA)')
		print(sum(H_PHA[0]))
		
		#return H_PHA
		
		plt.xlabel("Number of counts")
		plt.ylabel("Number of occurences")
		plt.legend(loc="upper right")
		plt.show()    
		
		
		
		#TCR
		
		"""
		m_step_TCR=self.step_cra_sync==step        
		#return self.times,m_step_TCR*m_energy
		
		tinc=(self.times[m_step_TCR*m_energy][1:]-self.times[m_step_TCR*m_energy][:-1])>0
		m_unique=concatenate([array([True]),tinc])
		#return tinc
		
		
		utcr_step=self.tcr_sync[m_step_TCR*m_energy][m_unique]
		
		
		
		#take into account times at which no counts occur within the selected ESA step
		utimes=unique(self.times)
		timestamps_nonzerocounts=unique(self.times[m_step_TCR*m_energy])
		inds_zerocounts=searchsorted(timestamps_nonzerocounts,setdiff1d(utimes,timestamps_nonzerocounts))
		print "comptest"
		print len(utimes)
		print setdiff1d(utimes,unique(self.times[m_step_TCR*m_energy][1:])) 
		
		utcr_step=insert(utcr_step,inds_zerocounts,array([0]*len(inds_zerocounts)))
		#return timestamps_nonzerocounts,setdiff1d(utimes,timestamps_nonzerocounts),inds_zerocounts,utcr_step    
		"""
		
		#return m_step*m_energy*m_utime
		
		N_tcr_SE=self.tcr_sync[m_step*m_energy]
		
		m_utime_SE=concatenate([array([True]),(self.times[m_step*m_energy][1:]-self.times[m_step*m_energy][:-1])>0])
		#return N_tcr_SE, m_utime_SE
		
		N_tcr=N_tcr_SE[m_utime_SE]
		
		#return N_tcr 
		
		plt.figure()
		H_TCR=plt.hist(N_tcr,countrange,alpha=1.0,color="r",label="TCR")
		
		print('sum(N_TCR)')
		print(sum(H_TCR[0]))
		
		
		plt.xlabel("Number of counts")
		plt.ylabel("Number of occurences")
		plt.legend(loc="upper right")
		plt.show()
		
		"""
		C_diff=N_tcr-h0
		
		plt.figure()
		H_C=plt.hist(C_diff,arange(-countrange[-1],countrange[-1],countrange[1]-countrange[0]),alpha=1.0,color="m",label="Diff")
		plt.xlabel("absolute difference of counts")
		plt.ylabel("Number of occurences")
		plt.legend(loc="upper right")
		plt.show()
		"""
		
		
		
		
		
		
		
		
		
		t1=clock()
		
		"""
		#same method as above solved with loop (slow) for correctness check
		
		#Histogram difference between PHA and TCR counts
		N_PHAs=zeros((len(unique(self.times))))
		N_TCRs=zeros((len(unique(self.times))))
		N_diffs=zeros((len(unique(self.times))))
		i=0
		utimes=unique(self.times)
		while i<len(utimes):
		    
		    m_times=self.times==utimes[i]
		    #PHA
		    if len(self.times[m_step_PHA*m_times*m_energy])>0:
		        N_PHA=len(self.times[m_step_PHA*m_times*m_energy])
		    else:
		        N_PHA=0
		    #TCR
		    if len(self.tcr_sync[m_step_TCR*m_times*m_energy])>0:
		        N_TCR=self.tcr_sync[m_step_TCR*m_times*m_energy][0]#all entrys of the masked sample have the same TCR value.
		    else:
		        N_TCR=0
		    N_diff=N_TCR-N_PHA    
		    N_PHAs[i]=N_PHA    
		    N_TCRs[i]=N_TCR
		    N_diffs[i]=N_diff
		    print i,utimes[i] 
		    i=i+1
		
		
		
		
		
		
		
		
		
		
	  
		#PHA hist
		plt.figure()
		H_P=plt.hist(N_PHAs,countrange,alpha=1.0,color="b",label="N_PHA")
		plt.xlabel("count rate (absolute)")
		plt.ylabel("Number of occurences")
		plt.legend(loc="upper right")
		plt.title("%i"%step)
		plt.show()
		
		#TCR hist
		plt.figure()
		H_T=plt.hist(N_TCRs,countrange,alpha=1.0,color="r",label="N_TCR")
		plt.xlabel("count rate (absolute)")
		plt.ylabel("Number of occurences")
		plt.legend(loc="upper right")
		plt.title("%i"%step)
		plt.show()

		t2=clock()
		
		#PHA-TCR hist
		plt.figure()
		H_diff=plt.hist(N_diffs,arange(-countrange[-1],countrange[-1],countrange[1]-countrange[0]),alpha=1.0,color="m",label="N_TCR-N_PHA")
		plt.xlabel("count rate difference (absolute)")
		plt.ylabel("Number of occurences")
		plt.legend(loc="upper right")
		plt.title("%i"%step)
		plt.show()
		"""
		
		print(t1-t0)
		#print t2-t1
		
		
		#return H_PHA,H_TCR,H_P,H_T, utcr_step, N_TCRs, tinc, self.times[m_step_TCR*m_energy][1:], H_C, H_diff 
		#return h0,N_PHAs
		return True
        
        
        
        
	def reconstruct_PHA(self):
		
		#cut out O6 box in ET-data
		#convert tof and energy channels to m/q ,m channels
		#cut out corresponding box in m/q, m matrix
		#calculate base rates from comparison (within the limited step resolution)
		#document everything in lab book and here!        
		
		
		
		
		return True
		
		
	def tof_to_mpq_hefti(self,tof,step):#tof in ch, 
		"""
		Algorithm to convert measured CTOF tofch at a given Energy-per-charge (ESA) step to mass-per-charge channel mpc. 
		It is entirely taken from Hefti PHD (1997), all table and formula numeration apply to this thesis.
		Minimal tof: 90 ch, maximal tof: 1022 ch, all other tof channels are converted to m/q = 126 ch (for tof < 90 ch) and m/q = 127 ch (for tof > 1022 ch).
		If a too small (large) m/q value is calculated it is assigned to channel 0 (125).
		Regular m/q range = [1ch, 125ch[
		Algorithm successfully tested with example from Heft PHD thesis: self.tof_to_mpq_hefti(tof=259ch,step=70) = 29 ch
		"""

		mpc=zeros((len(tof)))

		tofmask_valid=[(tof>=90)*(tof<=1022)]
		tof_valid=tof[tofmask_valid]
		step_valid=step[tofmask_valid]
		tof_low=tof[(tof<90)]
		tof_high=tof[(tof>1022)]
		inds_tofvalid=in1d(tof,tof_valid)
		inds_toflow=in1d(tof,tof_low)
		inds_tofhigh=in1d(tof,tof_high)

		#table A.2
		A00,A01,A02 = -294.5961,31.4464,-0.2448
		A10,A11,A12 = 60.1827,-1.3393,0.1219
		A20,A21,A22 = -0.026322,0.002940,0.000599
		
		#table A.3
		lnTADC=-1.632172#=approx log(tof_offset = 0.1955 ch)
		U_0=0.33109#lowest possible E/q in keV
		U_step=1.04092#factor by which E/q is increased from step to step
		U_acc=22.810#in kV, post-acceleration voltage in step5, probably the one applied from DOY 150-220 
		
		
		u=log(tof_valid)+lnTADC
		#def U_ESA(step):
		
		step_max=116
		w=log(U_0*U_step**(step_max-step_valid)+U_acc)#deviation to Hefti PHD is due to the fact that in the CTOF PHA data the step nomenclature of Aellig PHD (1998) is used. (see also "calc_mr_centerstep")
		
		A0=A00+A01*w+A02*w**2
		A1=A10+A11*w+A12*w**2
		A2=A20+A21*w+A22*w**2
		
		mpc_valid=A0+A1*u+A2*u**4#in ch (compare Figure A1)
		
		
		mpc_valid[mpc_valid<1]=0.
		mpc_valid[mpc_valid>125]=125.
		mpc_low=zeros((len(tof_low)))+126.
		mpc_high=zeros((len(tof_high)))+127.
		
		mpc[inds_tofvalid]=mpc_valid
		mpc[inds_toflow]=mpc_low
		mpc[inds_tofhigh]=mpc_high
		
		return mpc
		

	def tofE_to_m_hefti(self,tof,ESSD):#tof in ch, E_SSD in ch 
		"""
		Algorithm to convert measured CTOF tofch and SSD-Energy channel to mass channel m. 
		It is entirely taken from Hefti PHD (1997), all table and formula numeration apply to this thesis.
		Minimal E_SSD = 2 ch, maximal E_SSD = 254 ch, all other ESSD channels are converted to m = 62 ch (for E_SSD < 2 ch) and m = 63 ch (for E_SSD > 254 ch).
		If a too small (large) m value is calculated it is assigned to channel 0 (61).
		Regular m/q range = [1ch, 61ch[ 
		Algorithm successfully tested with example from Heft PHD thesis: self.tofE_to_m_hefti(tof=259,ESSD=44) = 39
		"""
		
		m=zeros((len(ESSD)))

		ESSDmask_valid=[(ESSD>=2)*(ESSD<=254)]
		ESSD_valid=ESSD[ESSDmask_valid]
		tof_valid=tof[ESSDmask_valid]
		ESSD_low=ESSD[(ESSD<2)]
		ESSD_high=ESSD[(ESSD>254)]
		inds_ESSDvalid=in1d(ESSD,ESSD_valid)
		inds_ESSDlow=in1d(ESSD,ESSD_low)
		inds_ESSDhigh=in1d(ESSD,ESSD_high)


		#table A.3
		lnTADC=-1.632172#=approx log(tof_offset = 0.1955 ch)
		#lnTADC=-1.632194
		
		#table A.4
		B0=-20.755203
		B1=-2.605667
		B2=0.529373
		B3=-14.045425
		B4=0.0
		B5=0.790695
		B6=3.744827
		
		#table A.5
		lnEADC=0.855
		#lnEADC=0.855011
		
		u=log(tof_valid)+lnTADC
		v=log(ESSD_valid)+lnEADC
		
		m_valid=B0+B1*v+B2*v**2+B3*u+B4*u**2+B5*u**3+B6*v*u#in ch (compare Figure A1)
		#m_valid+=0.5
		
		m_valid[m_valid<1]=0.
		m_valid[m_valid>61]=61.#note that m=61 is already in the overspill box
		m_low=zeros((len(ESSD_low)))+62.
		m_high=zeros((len(ESSD_high)))+63.
		
		m[inds_ESSDvalid]=m_valid
		m[inds_ESSDlow]=m_low
		m[inds_ESSDhigh]=m_high
		
		return m 
		
		
	def calc_mr_centerstep(self,mr_box,timestamp):
		"""
		Algorithm to calculate the central step (=cstep) around which the matrix rates are calculated as the scheme in table A.6 in Hefti PHD (1997) explains in detail.  
		The Algorithm is entirely taken from Hefti PHD (1997), all table and formula numeration apply to this thesis.
		The onboard calculated iron velocity is saved in the CTOF *.cmr files for each timestamp together with the respective 21 matrix rates for each of the 508 matrix boxes in 
		m/q-m space (see figure A.1).
		timestamps out of CTOF PHA timestamps,mr_box=0,1,2,...507
		"""
		
		#table A.7
		bbVpMult=49.8516#scaling factor for v_Fe
		VpOff=-164#offset for v_Fe
		
		
		#calculate shift for selected matrix box, see table A.2
		def K(mr_box):
			K=self.mrbox_shift[self.mrbox_number==mr_box]
			#print "calculated shift for mr_box %i is K=%i"%(mr_box,K)
			return K
		
		mask_v=(self.number_cmr_sync==mr_box)*(self.times_cmr_sync==timestamp)
		
		v_Fe=self.vsw_cmr_sync[mask_v]#in km/s
		
		#print 'timestamp:', timestamp
		#print 'v_Fe:', v_Fe
		
		#v_Fe=410.73#only used for comparison ewith example (see next line)
		
		#print log(v_Fe)*bbVpMult+VpOff-around(K(mr_box)/2.+0.1),timestamp,mr_box
		try:
			cstep_hefti=float(log(v_Fe)*bbVpMult+VpOff-around(K(mr_box)/2.+0.1))#corrections in last term are inserted in to force an up-rounding as required from comparison with examp. given in Hefti PHD
			#from Hefti documentation not clear whether one has to take the rounded or truncated float here, therefore for instance the pure float is left.
		except TypeError:
			print(log(v_Fe)*bbVpMult+VpOff-around(K(mr_box)/2.+0.1),timestamp,mr_box)
			return -1.			 
		
		step_max=116
		cstep=step_max-cstep_hefti
		#Note that in the hefti thesis the steps are defined the inverse way as in the Aellig PHD thesis (1998), which is the base for the Janitzek in-flight calbration): step_aellig=119
		
		return cstep 
		
		
		
		
		
		

	def relate_mpc_tof(self,steprange=[0,116],tof=arange(90,1023,1)):
		"""
		docstring!
		"""
		steps=arange(steprange[0],steprange[-1]+1,1)
		
		stepgrid,tofgrid=meshgrid(steps,tof)
		tg,sg=ravel(tofgrid),ravel(stepgrid)
		mpc=self.tof_to_mpq_hefti(tof=tg,step=sg)
		
		#data_inverted=transpose(array([sg,mpc,tg]))
		#data_inverted=array([sg,mpc,tg])
		
		mpc_valid=arange(1,82,1)
		tof_valid=zeros((len(steps),len(mpc_valid)))    
		i=0
		for step in steps:
			mpc_step=mpc[sg==step]
			tof_step=tg[sg==step]
			#return mpc_step,tof_step
			tof_valid_step=interp(x=mpc_valid,xp=mpc_step,fp=tof_step)
			tof_valid[i]=tof_valid_step
			i+=1
		
		self.tof_valid_table=array([steps,mpc_valid,tof_valid])
		#save this table to file (with default method input)
		
		return True
		
		
	def relate_m_E(self,tofrange=[90,1022],energy=arange(2,255,1)):
		"""
		docstring!
		"""
		tofs=arange(tofrange[0],tofrange[-1]+1,1)
		
		tofgrid,Egrid=meshgrid(tofs,energy)
		Eg,tg=ravel(Egrid),ravel(tofgrid)
		m=self.tofE_to_m_hefti(tof=tg,ESSD=Eg)
		
		m_valid=arange(1,62,1)
		E_valid=zeros((len(tofs),len(m_valid)))    
		i=0
		for tof in tofs:
			m_tof=m[tg==tof]
			E_tof=Eg[tg==tof]
			#return mpc_step,tof_step
			E_valid_step=interp(x=m_valid,xp=m_tof,fp=E_tof)
			E_valid[i]=E_valid_step
			i+=1
		       
		self.E_valid_table=array([tofs,m_valid,E_valid])
		#save this table to file (with default method input)

		return True


	def find_mpcm_bounds(self,mrbox_number):
		"""
		docstring!
		"""
		mpc_lowbound=float(self.mrbox_mpc_ll[self.mrbox_number==mrbox_number])
		mpc_upbound=float(self.mrbox_mpc_ur[self.mrbox_number==mrbox_number])
		m_lowbound=float(self.mrbox_m_ll[self.mrbox_number==mrbox_number])
		m_upbound=float(self.mrbox_m_ur[self.mrbox_number==mrbox_number])
	   
		mpc_bounds=[mpc_lowbound,mpc_upbound]
		m_bounds=[m_lowbound,m_upbound]
		
		return mpc_bounds,m_bounds



	def find_tofEbox(self,step,mrbox_number):#check method more extensively
		""" 
		mpc_bounds, mbounds: lower and upper boundary of mass_per_charge and mass in channels
		""" 
		
		#calculate corresponding tof channels 
		
		mpc_bounds,m_bounds=self.find_mpcm_bounds(mrbox_number) 
		
		ind_step_lb=float(where(self.tof_valid_table[0]==step)[0])
		ind_mpc_lb=float(where(self.tof_valid_table[1]==mpc_bounds[0])[0])
		tof_lowbound=round(self.tof_valid_table[2][ind_step_lb,ind_mpc_lb])
		print(self.tof_valid_table[0][ind_step_lb],self.tof_valid_table[1][ind_mpc_lb])
		
		ind_step_ub=float(where(self.tof_valid_table[0]==step)[0])
		ind_mpc_ub=float(where(self.tof_valid_table[1]==mpc_bounds[-1])[0])
		tof_upbound=round(self.tof_valid_table[2][ind_step_ub,ind_mpc_ub])
		print(self.tof_valid_table[0][ind_step_lb],self.tof_valid_table[1][ind_mpc_lb])
		
		ind_tof_ub=float(where(self.E_valid_table[0]==tof_upbound)[0])
		ind_m_lb=float(where(self.E_valid_table[1]==m_bounds[0])[0])
		E_lowbound=round(self.E_valid_table[2][ind_tof_ub,ind_m_lb])
		print(self.E_valid_table[0][ind_tof_ub],self.E_valid_table[1][ind_m_lb])
		
		ind_tof_lb=float(where(self.E_valid_table[0]==tof_lowbound)[0])
		#return m_bounds[-1]
		#return where(self.E_valid_table[1]==m_bounds[-1])[0]
		ind_m_ub=float(where(self.E_valid_table[1]==m_bounds[-1])[0])
		E_upbound=round(self.E_valid_table[2][ind_tof_lb,ind_m_ub])
		print(self.E_valid_table[0][ind_tof_lb],self.E_valid_table[1][ind_m_ub])
		
		tof_bounds=[tof_lowbound,tof_upbound]
		E_bounds=[E_lowbound,E_upbound]
		
		return tof_bounds,E_bounds 
		

	def load_cmr_heftical(self):#maybe only used for now
		self.load_matrix_box_definitions()
		self.load_matrix_shifts()
		self.relate_mpc_tof()
		self.relate_m_E()
		return True

	def calc_tcr_from_mr(self,timestamp):
		mr_sums_all=zeros((21))
		mr_sums_tcr=zeros((21))#measured SSD energies below channel 2, these end all up in mass channel 62 and therefore in the excluded boxes
		boxes_energy_underflow=concatenate([arange(418,422,1),arange(488,506,1)])
		mask_energy_underflow=invert(in1d(self.number_cmr_sync,boxes_energy_underflow))
		i=0
		while i<21:
			mr_sums_all[i]=sum(self.matrix_rates[i][self.times_cmr_sync==timestamp])
			mr_sums_tcr[i]=sum(self.matrix_rates[i][(self.times_cmr_sync==timestamp)*mask_energy_underflow])
			print(i,mr_sums_all[i],mr_sums_tcr[i],len(self.matrix_rates[i][self.times_cmr_sync==timestamp]),len(self.matrix_rates[i][(self.times_cmr_sync==timestamp)*mask_energy_underflow]) )
			i+=1    
		mrcounts_all=sum(mr_sums_all)    
		tcr_mr=sum(mr_sums_tcr)
		return mrcounts_all,tcr_mr    

	def countfilter_PHA(self,timerange,step,tofrange,Erange,Plot=False):#timerange=[timestamp_min,timestamp_max] in DOY, tofrange=[tofmin,tofmax[ in ch, Erange=[Emin,Emax[ in ch     
		"""
		docstring!
		"""
		mask_time=(self.times>=timerange[0])*(self.times<=timerange[-1])
		mask_step=self.step==step
		mask_tof=(self.tof>=tofrange[0])*(self.tof<tofrange[-1])
		mask_E=(self.energy>=Erange[0])*(self.energy<Erange[-1])
		mask_all=mask_time*mask_step*mask_tof*mask_E
		
		counts_filtered=len(self.times[mask_all])
		
		print(self.times[mask_all],self.tof[mask_all],self.energy[mask_all])
		
		#integrate plot_option of ET-matrix for quick-look comparison 
		#plot histogram for checking
		if Plot==True:
			ax=self.plot_ETmatrix(timerange=timerange,step=step,tofrange=[100,600],Erange=[1,100],bintof=1,binE=1)
			ax.plot(tofrange,[Erange[0],Erange[0]],color='r',linewidth=3.0)
			ax.plot(tofrange,[Erange[-1],Erange[-1]],color='r',linewidth=3.0)
			ax.plot([tofrange[0],tofrange[0]],Erange,color='r',linewidth=3.0)
			ax.plot([tofrange[-1],tofrange[-1]],Erange,color='r',linewidth=3.0)
		
		return counts_filtered 


	def plot_ETmatrix(self,timerange,step,tofrange,Erange,bintof,binE):
		
		mask_time=(self.times>=timerange[0])*(self.times<=timerange[-1])
		mask_step=self.step==step
		mask_tof=(self.tof>=tofrange[0])*(self.tof<=tofrange[-1])
		mask_E=(self.energy>=Erange[0])*(self.energy<=Erange[-1])
		
		mask_all=mask_time*mask_step*mask_tof*mask_E
		#mask_all=mask_step*mask_tof*mask_E
		
		
		tof_masked=self.tof[mask_all]
		energy_masked=self.energy[mask_all]
		
		#return tof_masked,energy_masked,timerange 
		
		fig, ax = plt.subplots(1,1)
		h1,bx1,by1=histogram2d(tof_masked,energy_masked,[arange(tofrange[0],tofrange[-1],bintof),arange(Erange[0],Erange[-1],binE)])      
		Plot=ax.pcolor(bx1,by1,h1.T,vmin=1, vmax=max(ravel(h1)))
		Plot.cmap.set_under('w')
		ax.set_xlabel("tof [ch]")       
		ax.set_ylabel("energy [ch]")
		ax.set_title("timerange [DOY 1996]: %.5f - %.5f, step: %i"%(timerange[0],timerange[-1],step))
		self.plot=fig,ax
		cb1 = fig.colorbar(Plot)
		cb1.set_label("Counts per Bin")             
		self.plot=fig, ax
		return ax        
		



	def compare_PHAMR_counts(self,mrbox_number,prange,timestamp,plot_PHA_boxes=False,mbounds="upplow_exc"):    
		"""
		after Hefti PHD thesis page 88 this regular matrix rate binning applies only if the "compressed mode" is not active, so when v_Fe>348 km/s.
		For estimated value of v_Fe at each timestamp see cmr files. 
		"""
		
		#self.assign_mrbox_prange(mbounds=mbounds,prange=prange)
		
		cstep=round(self.calc_mr_centerstep(mr_box=mrbox_number,timestamp=timestamp))#step is rounded here, should be discussed and checked!
		 
		mr_steps=array([0,cstep-32,cstep-24,cstep-16,cstep-12,cstep-8,cstep-6,cstep-4,cstep-2, cstep-1,cstep,cstep+1,cstep+2,cstep+3, cstep+5,cstep+7,cstep+9,cstep+13,cstep+17,cstep+25,cstep+32 ,116])
		mr_steps[mr_steps<0]=0
		mr_steps[mr_steps>116]=116
		mr_steps_diff=mr_steps[1:]-mr_steps[:-1]
		print(mr_steps,len(mr_steps),mr_steps_diff)
		
		#cstep 20 [i+1,Halt] is missing so far!
		
		#steps_PHA=zeros((len(mr_steps)-1))
		counts_PHA=zeros((len(mr_steps)-1))
		#steps_MR=zeros((len(mr_steps)-1))
		counts_MR=zeros((len(mr_steps)-1))
		#PHA data step-count histogram
		mr_step=mr_steps[0]
		j=0
		while mr_step < mr_steps[-1]:
			print('mr_step,j')
			print(mr_step,j)
			
			substep_counts_PHA=array([])
			substep=0
			while substep<mr_steps_diff[j]:
				print('substep')
				print(substep)
				#steps_PHA[i]=step
				tof_bounds,E_bounds=self.find_tofEbox(step=mr_step+substep,mrbox_number=mrbox_number)
				substep_counts_PHA=append(substep_counts_PHA,self.countfilter_PHA(timerange=[timestamp,timestamp],step=mr_step+substep,tofrange=tof_bounds,Erange=E_bounds,Plot=plot_PHA_boxes))
				print("substep_counts_PHA,j,mr_step+substep",substep_counts_PHA,j,mr_step+substep)
				substep+=1    
				
			counts_PHA[j]=sum(substep_counts_PHA)        
			print("counts_PHA[j],j", counts_PHA[j],j)
			j+=1
			mr_step=mr_steps[j]
		
		#MR data step-count histogram
		mr_step=mr_steps[0]
		j=0
		#self.mrbox_prange
		while mr_step < mr_steps[-1]:    
			stepcounts_MR=self.matrix_rates[j][(self.number_cmr_sync==mrbox_number)*(self.times_cmr_sync==timestamp)]
			counts_MR[j]=stepcounts_MR
			j+=1
			mr_step=mr_steps[j]
		
		mrstep_binmids=(mr_steps[1:]+mr_steps[:-1])/2.
		mrstep_widths=mr_steps[1:]-mr_steps[:-1]
		#return mr_steps,counts_PHA,mrstep_widths
		
		plt.figure()
		plt.title("matrix_box: %i, timestamp: %f"%(mrbox_number,timestamp))
		#plt.bar(mr_steps[:-1],counts_MR,width=mrstep_widths,color="r",alpha=0.5,label="MR, %s"%(mbounds))
		plt.bar(mr_steps[:-1],counts_MR,width=mrstep_widths,color="r",alpha=0.5,label="MR")
		plt.bar(mr_steps[:-1],counts_PHA,width=mrstep_widths,color="b",alpha=0.5,label="PHA")
		#plt.plot(mrstep_binmids,counts_PHA,linewidth=2.0,marker="o",color="b",alpha=1.0)
		#plt.plot(mrstep_binmids,counts_MR,linewidth=2.0,marker="o",color="r",alpha=1.0)    
		plt.xlabel("ESA step")
		plt.ylabel("counts")
		plt.legend()
		plt.axis([0,120,0,1.1*max(counts_MR)])
		plt.show()
		
		#velocity histogram (up to now) only valid for O6 (mrbox_number=235, and vicinity)
		if mrbox_number==235:#235 only for checking
			v_O6=getionvel(16/6.,mr_steps)
			v_O6_binmids=getionvel(16/6.,mrstep_binmids)
			v_O6_widths=v_O6[1:]-v_O6[:-1]
			
			plt.figure()
			plt.title("matrix_box: %i, timestamp: %f"%(mrbox_number,timestamp))
			plt.bar(v_O6[:-1],counts_MR,width=v_O6_widths,color="r",alpha=0.5,label="MR")
			plt.bar(v_O6[:-1],counts_PHA,width=v_O6_widths,color="b",alpha=0.5,label="PHA")
			#plt.plot(v_O6_binmids,counts_PHA,linewidth=2.0,marker="o",color="b",alpha=1.0)
			#plt.plot(v_O6_binmids,counts_MR,linewidth=2.0,marker="o",color="r",alpha=1.0)    
			plt.xlabel("v_O6 [km/s]")
			plt.ylabel("counts")
			plt.legend()
			plt.axis([0,1600,0,1.1*max(counts_MR)])#velocity maximum calculated from v_O6 at step 0
			plt.show()  
			
		if mrbox_number==45:# onlyfor checking
			
			v_Fe13=getionvel(56/13.,mr_steps)
			v_Fe13_binmids=getionvel(56/13.,mrstep_binmids)
			v_Fe13_widths=v_Fe13[1:]-v_Fe13[:-1]
			
			plt.figure()
			plt.title("matrix_box: %i, timestamp: %f"%(mrbox_number,timestamp))
			plt.bar(v_Fe13[:-1],counts_MR,width=v_Fe13_widths,color="r",alpha=0.5,label="MR")
			plt.bar(v_Fe13[:-1],counts_PHA,width=v_Fe13_widths,color="b",alpha=0.5,label="PHA")
			#plt.plot(v_O6_binmids,counts_PHA,linewidth=2.0,marker="o",color="b",alpha=1.0)
			#plt.plot(v_O6_binmids,counts_MR,linewidth=2.0,marker="o",color="r",alpha=1.0)    
			plt.xlabel("v_Fe13 [km/s]")
			plt.ylabel("counts")
			plt.legend()
			plt.axis([0,1600,0,1.1*max(counts_MR)])#velocity maximum calculated from v_Fe13 at step 0
			plt.show()

			
		print("calculated center step:", cstep)
		print("stepbins:",mr_steps,len(mr_steps),mr_steps_diff)
		
		return mr_steps,counts_PHA,counts_MR
		
		
	def compare_PHAMR_Ioncounts(self,timestamp,ion,m,q,Nsigma=2.0,steps=arange(0,116,1),plot_PHA_boxes=False,multiply_br=False,mbounds="upplow_exc",Plot=False):
		
		utimes=unique(self.times)
		hists=zeros((len(steps),len(utimes)-1))
		mpq_bounds_low=zeros(len(steps))
		mpq_bounds_high=zeros(len(steps))	
		m_bounds_low=zeros(len(steps))
		m_bounds_high=zeros(len(steps))	
		j=0
		
		#ax=self.plot_pranges(step,tofs=arange(100,701,1),energies=arange(2,101,1))[3]
		for step in steps:
		
			#PHA
			tofpos=tof(step,m,q)
			Epos=ESSD(tofpos,m)
			tofsigma=tofsig(step,m,q)
			Esigma=Esig(Epos)
			print(tofpos,Epos,tofsigma,Esigma )
			tofbounds=[tofpos-Nsigma*tofsigma,tofpos+Nsigma*tofsigma]
			Ebounds=[Epos-Nsigma*Esigma,Epos+Nsigma*Esigma]
			
			#cut-out and sum PHA data
			
			#self.times=zeros((0))
			#self.range=zeros((0))
			#self.energy=zeros((0))
			#self.tof=zeros((0))
			#self.step=zeros((0))
			
			stepmask=(self.step==step)
			tofmask=(self.tof>=tofbounds[0])*(self.tof<tofbounds[-1])
			Emask=(self.energy>=Ebounds[0])*(self.energy<Ebounds[-1])
			
			tofs_step=self.tof[stepmask*tofmask*Emask]
			energies_step=self.energy[stepmask*tofmask*Emask]
			times_step=self.times[stepmask*tofmask*Emask]
			h=histogram(times_step,utimes)
			hists[j]=h[0]			
			
			if Plot==True:
				plt.figure()
				plt.plot(h[1][:-1],h[0])
				plt.xlabel("time")
				plt.ylabel("PHA counts")
				plt.legend()
				plt.show()
			
			#MR
			mpq_low=self.tof_to_mpq_hefti(tof=tofbounds[0],step=step)#tof in ch,
			mpq_high=self.tof_to_mpq_hefti(tof=tofbounds[1],step=step)#tof in ch,
			m_low=self.tofE_to_m_hefti(tof=tofbounds[1],ESSD=Ebounds[0])#tof in ch, E_SSD in ch 
			m_high=self.tofE_to_m_hefti(tof=tofbounds[0],ESSD=Ebounds[1])#tof in ch, E_SSD in ch 
			mpq_bounds_low[j]=mpq_low
			mpq_bounds_high[j]=mpq_high
			m_bounds_low[j]=m_low
			m_bounds_high[j]=m_high
			
			mrbox_bounds=array([mpq_bounds_low,mpq_bounds_high,m_bounds_low,m_bounds_high])
			#mmpq_bounds_Si7=[39.,43.,41,51]
			j=j+1
		
		VDFs_PHA=hists.T	
		mmpq_bounds_Si7=[39.,43.,41,51]	

		step_min=vel_to_step(m=m,q=q,v=700.)-1
		step_max=vel_to_step(m=m,q=q,v=250.)+1
		stepmask_vel=(steps>=step_min)*(steps<step_max)


		mb_low=mean(m_bounds_low[stepmask_vel])
		mb_high=mean(m_bounds_high[stepmask_vel])
		mpqb_low=mean(mpq_bounds_low[stepmask_vel])
		mpqb_high=mean(mpq_bounds_high[stepmask_vel])
		
		boxmask_mpc=(self.mrbox_mpc_ll>=mpqb_low)*(self.mrbox_mpc_ll<mpqb_high)#discuss the boundaries for accuracy
		boxmask_m=(self.mrbox_m_ll>=mb_low)*(self.mrbox_m_ll<mb_high)
		
		
		mrbox_numbers_valid=self.mrbox_number[boxmask_mpc*boxmask_m]
		
		
		#mrbox_numbers_valid=[201]#201 for Si7+ after Hefti 
		
		a=[]
		b=[]
		for mrbox_number in mrbox_numbers_valid:
			
			steps_mr,counts_PHA,counts_mr=self.compare_PHAMR_counts(mrbox_number,prange=0,timestamp=timestamp,plot_PHA_boxes=False,mbounds="upplow_exc")
			a.append(steps_mr)
			b.append(counts_mr)
		
		print(mrbox_numbers_valid)
		a=array(a)
		b=array(b)
		
		VDF_PHA=VDFs_PHA[utimes==timestamp]
		
		return a,b,VDF_PHA,VDFs_PHA
			
		
		
		
		return VDFs_PHA,mrbox_bounds,steps[stepmask_vel],steps[stepmask_vel],mb_high,mb_low,mpqb_low,mpqb_high
		
			
		
		
		
		
		
		
		
		
		
		"""
		def compare_PHAMR_counts(self,mrbox_number,timestamp,plot_PHA_boxes=False):    
		
		#old version of compare_PHAMR_counts
		#after Hefti PHD thesis page 88 this regular matrix rate binning applies only if the "compressed mode" is not active, so when v_Fe>348 km/s.
		#For estimated value of v_Fe at each timestamp see cmr files. 
		
		cstep=round(self.calc_mr_centerstep(mr_box=mrbox_number,timestamp=timestamp))#step is rounded here, should be discussed and checked!        
		#old version of
		
		
		steps_PHA=zeros((5))
		counts_PHA=zeros((5))
		steps_MR=zeros((5))
		counts_MR=zeros((5))
		    self.mrbox_prange
		
		step=cstep-2
		i=0
		while step<=cstep+2:
		    
		    #PHA data step-count histogram
		    steps_PHA[i]=step
		    tof_bounds,E_bounds=self.find_tofEbox(step=step,mrbox_number=mrbox_number)
		    stepcounts_PHA=self.countfilter_PHA(timerange=[timestamp,timestamp],step=step,tofrange=tof_bounds,Erange=E_bounds,Plot=plot_PHA_boxes)
		    counts_PHA[i]=stepcounts_PHA        
		    
		    #MR data step-count histogram
		    steps_MR[i]=step
		    stepcounts_MR=self.matrix_rates[8+i][(self.number_cmr_sync==mrbox_number)*(self.times_cmr_sync==timestamp)]
		    counts_MR[i]=stepcounts_MR
		    
		    step+=1
		    i+=1
		
		
		plt.figure()
		plt.title("matrix_box: %i, timestamp: %f"%(mrbox_number,timestamp))
		plt.bar(steps_PHA,counts_PHA,width=1.0,color="b",alpha=0.5,label="PHA")
		plt.bar(steps_MR,counts_MR,width=1.0,color="r",alpha=0.5,label="MR")    
		plt.legend()
		plt.show()
		
		
		return steps_PHA,steps_MR,counts_PHA,counts_MR
	"""        

	def plot_mrbox_hist(self,mr,timerange,logrates=False,plot_boxnumber=True,boxshift=False,plot_histogram=True, mark_Heftiboxes=False,markcolor="gray",Xlims=[0,80],Ylims=[0,65],figx=13.9,figy=7.,fontsize=32,ticklabelsize=20,marker_lw=2,cycle=0,ytitle_pos=1.,save_figure=False,figpath="/home/asterix/janitzek/ctof/achapter2_finalplots/brcor/",figname="test",figformat="png",plot_PRborders=False,DOY2=175):
		"""
		used example:  d.plot_mrbox_hist(mr=10,timerange=[utimes[0],utimes[1]],logrates=True,Xlims=[17,51],Ylims=[25,53],mark_Heftiboxes=True,markcolor="m",marker_lw=3,cycle=1,save_figure=True,figname="MRdata_O6box",ytitle_pos=1.01)
		docstring!
		"""
	   
		binx,biny=1,1
		x,y=arange(0,82,binx),arange(0,64,biny)
		ygrid,xgrid=meshgrid(y,x)
		xg,yg=ravel(xgrid),ravel(ygrid)
		#hg=zeros((len(y),len(x)))
		hg=zeros((len(xg)))
	   
		#self.mrbox_number
		#self.mrbox_mpc_ll#lower_left box corner mass channel
		#self.mrbox_m_ll=#lower_left box corner mass-per-charge channel
		#self.mrbox_mpc_ur=#upper_right box corner mass channel
		#self.mrbox_m_ur=#upper_ri
		
		#return where((self.mrbox_mpc_ll==25)*(self.mrbox_m_ll)==15)
		
		for i in self.mrbox_number:
			
			if self.mrbox_mpc_ll[i]>-1:
				
				xmin=self.mrbox_mpc_ll[i]
				xmax=self.mrbox_mpc_ur[i]
				ymin=self.mrbox_m_ll[i]
				ymax=self.mrbox_m_ur[i]
				
				#if i==456:
					#return xmin,xmax,ymin,ymax
				
				hg[(xg>=xmin)*(xg<xmax)*(yg>=ymin)*(yg<ymax)]=sum(self.matrix_rates[mr][(self.times_cmr_sync>=timerange[0])*(self.times_cmr_sync<=timerange[-1])*(self.number_cmr_sync==i)])
				print(i,sum(self.matrix_rates[mr][(self.times_cmr_sync>=timerange[0])*(self.times_cmr_sync<=timerange[-1])*(self.number_cmr_sync==i)]))
		
		bx=x[:-1]
		by=y[:-1]
		h=hg.reshape(len(x),len(y)).T
		
		#return bx,by,h,xg,yg,hg
		
		fig, ax = plt.subplots(1,1,figsize=(figx,figy))
		if plot_PRborders==True:
			ax.plot([Xlims[0],Xlims[1]],[7,7],linewidth=3,color="k")
			ax.plot([Xlims[0],Xlims[1]],[33,33],linewidth=3,color="k")
			ax.plot([Xlims[0],Xlims[1]],[41,41],linewidth=3,color="k")
			ax.plot([Xlims[0],Xlims[1]],[47,47],linewidth=3,color="k")
			ax.plot([Xlims[0],Xlims[1]],[52,52],linewidth=3,color="k")
		
		if logrates==True:
			
			if plot_histogram==True:
				Plot=ax.pcolor(bx,by,h,norm=LogNorm(vmin=1, vmax=max(hg)))
		
			i=0
			if plot_boxnumber==True:
					#for i in [0]:
					for i in self.mrbox_number:
						if (self.mrbox_mpc_ll[i]>=Xlims[0]) and (self.mrbox_mpc_ur[i]<=Xlims[-1]) and (self.mrbox_m_ll[i]>=Ylims[0]) and (self.mrbox_m_ur[i]<=Ylims[-1]):
						#if self.mrbox_mpc_ll[i]>=0 and self.mrbox_m_ll[i]>=0:
							print(i)
							#return self.mrbox_number[i]
							if boxshift==True:
								ax.text(self.mrbox_mpc_ll[i]+0.2,self.mrbox_m_ll[i]+0.2,"%i"%(self.mrbox_shift[i]))
							else:
								ax.text(self.mrbox_mpc_ll[i]+0.2,self.mrbox_m_ll[i]+0.2,"%i"%(self.mrbox_number[i]))
							ax.plot([self.mrbox_mpc_ll[i],self.mrbox_mpc_ur[i]],[self.mrbox_m_ll[i],self.mrbox_m_ll[i]],linestyle="-",color="k")
							ax.plot([self.mrbox_mpc_ll[i],self.mrbox_mpc_ur[i]],[self.mrbox_m_ur[i],self.mrbox_m_ur[i]],linestyle="-",color="k")
							ax.plot([self.mrbox_mpc_ll[i],self.mrbox_mpc_ll[i]],[self.mrbox_m_ll[i],self.mrbox_m_ur[i]],linestyle="-",color="k")
							ax.plot([self.mrbox_mpc_ur[i],self.mrbox_mpc_ur[i]],[self.mrbox_m_ll[i],self.mrbox_m_ur[i]],linestyle="-",color="k")
					
					if mark_Heftiboxes==True:
						i=0
						for i in [41,92,93,201,235]:
						#for i in [235]:
							ax.plot([self.mrbox_mpc_ll[i],self.mrbox_mpc_ur[i]],[self.mrbox_m_ll[i],self.mrbox_m_ll[i]],linewidth=marker_lw,linestyle="-",color=markcolor)
							ax.plot([self.mrbox_mpc_ll[i],self.mrbox_mpc_ur[i]],[self.mrbox_m_ur[i],self.mrbox_m_ur[i]],linewidth=marker_lw,linestyle="-",color=markcolor)
							ax.plot([self.mrbox_mpc_ll[i],self.mrbox_mpc_ll[i]],[self.mrbox_m_ll[i],self.mrbox_m_ur[i]],linewidth=marker_lw,linestyle="-",color=markcolor)
							ax.plot([self.mrbox_mpc_ur[i],self.mrbox_mpc_ur[i]],[self.mrbox_m_ll[i],self.mrbox_m_ur[i]],linewidth=marker_lw,linestyle="-",color=markcolor)
					
						
		else:
			Plot=ax.pcolor(bx,by,h,vmin=1, vmax=max(hg))
		
		
		
		ax.set_xlabel(r"$\rm{mass-per-charge \ [ch]}$",fontsize=fontsize)#before fonstize=20       
		ax.set_ylabel(r"$\rm{mass \ [ch]}$",fontsize=fontsize)#before 20  
		ax.tick_params(axis="x", labelsize=ticklabelsize)
		ax.tick_params(axis="y", labelsize=ticklabelsize)
		
		#self.plot=fig,ax
		if plot_histogram==True:
			Plot.cmap.set_under('w')
			cb1 = fig.colorbar(Plot)
			cb1.set_label(r"$\rm{counts \ per \ bin}$",fontsize=fontsize)#before 20              
			for ctick in cb1.ax.get_yticklabels():
				ctick.set_fontsize(ticklabelsize)
			
			#ax.set_title("timerange [DOY 1996]: [%.5f,%.5f], matrix rate: %i"%(timerange[0],timerange[-1],mr),fontsize=20)
			#ax.set_title(r"$\rm{DOY: \ 174 \ 1996, \ CTOF-Cycle: \ %i, \ Matrix \ Rate: \ %i}$"%(cycle,mr),fontsize=fontsize,y=ytitle_pos)
			ax.set_title(r"$\rm{DOY: \ 174 - %i \ 1996, \ Matrix \ Rate: \ %i}$"%(DOY2,mr),fontsize=fontsize,y=ytitle_pos)
		
		
		else:	
			if boxshift==True:
				ax.set_title(r"$\rm{Lookup \ Table \ for \ the \ CTOF \ Matrix \ Rates \ Center \ Step}$",fontsize=fontsize)
			else:
				ax.set_title(r"$\rm{CTOF \ Matrix \ Rates \ Box \ Definition}$",fontsize=fontsize)
		ax.set_xlim([Xlims[0],Xlims[-1]])
		ax.set_ylim([Ylims[0],Ylims[-1]])
		plt.show()
		
		if save_figure==True:
			plt.savefig(figpath+figname+"."+figformat,bbox_inches='tight')
		
		return bx,by,h,xg,yg,hg


	def plot_matrixrates(self,mrbox_number,timerange,Plot=True):
		"""
		docstring!
		"""
		#print timerange
		
		timemask=(self.times_cmr_sync>=timerange[0])*(self.times_cmr_sync<=timerange[-1])
		#timemask=(self.times>=timerange[0])*(self.times<=timerange[-1])
		boxmask=(self.number_cmr_sync==mrbox_number)
		
		#return timemask*boxmask 
		
		mr=arange(0,21,1.)
		counts=sum(self.matrix_rates.T[timemask*boxmask],axis=0)
		#return mr,counts 
	  
		if Plot==True:
			plt.figure()
			plt.title("timerange [DOY 1996]: [%.5f,%.5f], matrix box: %i"%(timerange[0],timerange[-1],mrbox_number))
			plt.bar(mr,counts,width=1.0,align="center")
			plt.xlabel("matrix rate",fontsize=20)
			plt.ylabel("counts per matrix rate",fontsize=20)
			plt.axis([0,21,0,1.3*max(counts)])
			plt.show()
			 
		return mr,counts 
	
	
		
	def MR_to_VDF(self,mrbox_number,timestamp,compression="high_res",Plot=True,figx=13.9,figy=7*2,fontsize=32,ticklabelsize=20,save_figure=False,figpath="/home/asterix/janitzek/ctof/achapter3_finalplots/",figname="test",Wspace=0.3,legsize=25,Ylim_factor=1.3,ypos_title=1.01,DOY=174,cycle=0,legendsize=25,Ncol=1,legendloc="upper right",element="O",charge=6,velrange=None):	
		"""
		used examples: 
		O6slow:
		ds.MR_to_VDF(235,uts[170],save_figure=False,figname="Derive_VDF_O6_slow",DOY=178,cycle=170,ypos_title=0.95,Ncol=1,legendloc="upper right",legendsize=20,element="O",charge=6,Ylim_factor=1.10,velrange=[300,440])
		O6fast:
		df.MR_to_VDF(235,utf[120],save_figure=False,figname="Derive_VDF_O6_fast",DOY=214,cycle=120,ypos_title=0.95,Ncol=1,legendloc="upper right",legendsize=20,element="O",charge=6,Ylim_factor=1.10,velrange=[330,800])
		Fe9fast:
		df.MR_to_VDF(92,utf[120],save_figure=True,figname="Derive_VDF_Fe9_fast",DOY=214,cycle=120,ypos_title=0.95,Ncol=1,legendloc="upper right",legendsize=20,element="Fe",charge=9,Ylim_factor=1.10,velrange=[290,800])
		Si7fast:
		df.MR_to_VDF(201,utf[120],save_figure=True,figname="Derive_VDF_Si7_fast",DOY=214,cycle=120,ypos_title=0.95,Ncol=1,legendloc="upper right",legendsize=20,element="Si",charge=7,Ylim_factor=1.10,velrange=[330,800])

		"""
		
		mr,counts_mr=self.plot_matrixrates(mrbox_number,timerange=[timestamp,timestamp],Plot=False)
		
		utimes=unique(self.times_cmr_sync)
		times=self.times_cmr_sync
		ss=searchsorted(times,utimes)
		vsw=self.vsw_cmr_sync
		#utimes=unique(self.times)
		#times=self.times
		#ss=searchsorted(times,utimes)
		#vsw=self.vsw
		
		uvsw=vsw[ss]
		vsw_ind=where(timestamp==utimes)[0]
		vp=uvsw[vsw_ind]#proton speed at given time stamp

		
		Boxcounts_mr,cstep,mr_steps,mr_counts,mr_steps_diff=self.calc_boxbaserates(mrbox=mrbox_number,timestamp=timestamp)
		counts=Boxcounts_mr	
		steps=arange(0,116,1.)
	
		if mrbox_number==235:
			m=16.
			q=6.
		elif mrbox_number==201:	
			m=28.
			q=7.
		elif mrbox_number in [41,92,93]:	
			m=56.
			q=9.
		
		mr_vels=step_to_vel(step=mr_steps, q_e=q, m_amu=m)
		mr_veldiffs=zeros((len(mr_vels)))+5.#improve!
		vels=step_to_vel(step=steps, q_e=q, m_amu=m)
		steps_extend=concatenate([steps,array([117.])])
		vels_extend=step_to_vel(step=steps_extend, q_e=q, m_amu=m)
		veldiffs=vels_extend[1:]-vels_extend[:-1]
		
		
		#ps2_correction(N,v,v_ref)
		indmax=where(counts==max(counts))[0]
		if len(shape(indmax))>0:
			indmax=indmax[0]
		velmax=vels[indmax]
		counts_cor=ps2_correction(N=counts,v=vels,v_ref=velmax)#check maximum error here!
		
		vmean=average(vels,weights=counts_cor)
		vmedian=median(vels[counts_cor>0])
		
		if Plot==True:
			
			fig, ax = plt.subplots(2,2,figsize=(figx,figy))
			fig.subplots_adjust(wspace=Wspace)
			fig.suptitle(r"$\rm{MR-Box: \ \ %i \ (%s}^{%i\plus} {), \ \ CTOF-Cycle: \ %i\ -\ %i}$"%(mrbox_number,element,charge,DOY,cycle), y=ypos_title,fontsize=fontsize)
			
			
			#ax.set_title(r"$\rm{PR: \ \ %i \ , \ CTOF-Cycle: \ 174\ -\ %i}$"%(PR,utime_ind),fontsize=fontsize,y=1.01)
			#ax.set_title("MRbox: %i, time: %.5f DOY 1996"%(mrbox_number,timestamp))
			
			#upper left
			ax[0][0].bar(mr,counts_mr,width=1.0,align="center",color="b", alpha=0.5, linewidth=2)
			ax[0][0].set_xlabel(r"$\rm{matrix \ rate}$",fontsize=fontsize)
			ax[0][0].set_ylabel(r"$\rm{counts}$",fontsize=fontsize)
			ax[0][0].set_xlim([0,20.01])
			ax[0][0].set_ylim([0.1,Ylim_factor*max(counts_mr)])
			ax[0][0].tick_params(axis="x", labelsize=ticklabelsize)
			ax[0][0].tick_params(axis="y", labelsize=ticklabelsize)
			
			#upper right
			ax[0][1].bar(steps-0.5,counts, width=1., color="w", alpha=0.5, linewidth=2)
			mr_counts_scale=mr_counts/mr_steps_diff
			ax[0][1].bar((mr_steps-0.5)[mr_counts_scale>0],mr_counts_scale[mr_counts_scale>0], width=mr_steps_diff[mr_counts_scale>0], color="r", alpha=0.5, linewidth=3,edgecolor="b")
			ax[0][1].set_xlabel(r"$\rm{E/q-step}$",fontsize=fontsize)
			ax[0][1].set_ylabel(r"$\rm{counts}$",fontsize=fontsize)
			ax[0][1].set_ylim([0.1,Ylim_factor*max(counts)])
			ax[0][1].tick_params(axis="x", labelsize=ticklabelsize)
			ax[0][1].tick_params(axis="y", labelsize=ticklabelsize)

			#lower left
			ax[1][0].bar(vels,counts, width=veldiffs, align="center", color="c", alpha=0.5, linewidth=2)
			ax[1][0].set_xlabel(r"$\rm{ion \ speed \ [km/s]}$",fontsize=fontsize)
			ax[1][0].set_ylabel(r"$\rm{counts}$",fontsize=fontsize)
			ax[1][0].set_ylim([0.1,Ylim_factor*max(counts)])
			ax[1][0].tick_params(axis="x", labelsize=ticklabelsize)
			ax[1][0].tick_params(axis="y", labelsize=ticklabelsize)
			if velrange!=None:
				ax[1][0].set_xlim(velrange[0],velrange[-1])
			
			#lower right
			ax[1][1].bar(vels,counts_cor, width=veldiffs, align="center", color="orange", alpha=0.5, linewidth=2)
			ax[1][1].set_xlabel(r"$\rm{ion \ speed \ [km/s]}$",fontsize=fontsize)
			ax[1][1].set_ylabel(r"$\rm{counts \ cor.}$",fontsize=fontsize)
			#ax[1][1].plot([vmean,vmean],[0,Ylim_factor*max(counts_cor)],linewidth=3,color="k",label=r"$\langle \rm{v}_{\rm{ion}} \rangle = \ %i \ \rm{km/s}$"%(round(vmean)))
			ax[1][1].plot([vmean,vmean],[0,Ylim_factor*max(counts_cor)],linewidth=3,color="k",label=r"$\langle \rm{v}_{\rm{ion}} \rangle$"%(round(vmean)))
			
			if velrange!=None:
				ax[1][1].set_xlim(velrange[0],velrange[-1])
			
			#ax[1][1].plot([vmedian,vmedian],[0,1.3*max(counts_cor)],linewidth=3,color="k",linestyle="--",label="median ion speed = %i km/s"%(round(vmedian)))
			#ax[1][1].plot([vp,vp],[0.1,Ylim_factor*max(counts_cor)],linewidth=3,color="r",label=r"$\langle \rm{v}_{\rm{p}} \rangle = \ %i \ \rm{km/s}$"%(round(vp)))
			ax[1][1].plot([vp,vp],[0.1,Ylim_factor*max(counts_cor)],linewidth=3,color="r",label=r"$\langle \rm{v}_{\rm{p}} \rangle$"%(round(vp)))
			ax[1][1].legend(loc=legendloc,prop={'size':legendsize},ncol=Ncol)
			ax[1][1].set_ylim([0.1,Ylim_factor*max(counts_cor)])
			ax[1][1].tick_params(axis="x", labelsize=ticklabelsize)
			ax[1][1].tick_params(axis="y", labelsize=ticklabelsize)

			if save_figure==True:
				plt.savefig(figpath+figname,bbox_inches='tight')
			else:
				plt.show()
		return mr,counts_mr,vp,vmean
		


	def plot_pranges(self,step,tofs=arange(100,701,1),energies=arange(2,101,1)):
		"""
		docstring!
		"""
		stepmask=(self.step==step)
		tof_step=self.tof[stepmask]
		energy_step=self.energy[stepmask]
		range_step=self.range[stepmask]
		stepdata = array([tof_step,energy_step,range_step]).T
		ustepdata=vstack({tuple(row) for row in stepdata}).T
		
		ranges=zeros((len(tofs),len(energies)))        
		i=0
		for tof in tofs:
			ranges_tof=ustepdata[2][ustepdata[0]==tof]
			energies_tof=ustepdata[1][ustepdata[0]==tof]
			#energies_tofexist_inds_ldata=where(in1d(energies_tof,energies)==True)[0]
			#energies_tofexist_inds_lrange=where(in1d(energies,energies_tof)==True)[0]

			if len(energies_tof)>0:#can only be considered if there is a measured count at this tof
				
				x=energies_tof
				y=energies
				
				index = np.argsort(x)
				sorted_x = x[index]
				sorted_index = np.searchsorted(sorted_x, y)
				yindex = np.take(index, sorted_index, mode="clip")
				mask = x[yindex] != y
				#result = np.ma.array(yindex, mask=mask)
				
				yindex[where(mask==True)[0]]=-1
				ec=concatenate([energies_tof,array([-1])])
				rc=concatenate([ranges_tof,array([-1])])
				
				final_energies=ec[yindex]
				final_ranges=rc[yindex]
			
				#if tof==211:
					#return energies,energies_tof,l,yindex
					#return energies_tof,energies,yindex,ec,rc,final_energies, final_ranges
			
			else:
				final_ranges=zeros((len(energies)))-1
			ranges[i]=final_ranges
			i=i+1
		
		fig, ax = plt.subplots(1,1)
		Plot=ax.pcolor(tofs,energies,ranges.T,vmin=0, vmax=max(ravel(ranges)))
		Plot.cmap.set_under('w')
		ax.set_xlabel("tof [ch]")       
		ax.set_ylabel("energy [ch]")
		ax.set_title("ESA step : %i"%(step))
		self.plot=fig,ax
		cb1 = fig.colorbar(Plot)
		cb1.set_label("priority range number")            
		plt.show()
		
		return tofs,energies,ranges,ax
	
	
	def check_IonPR(self,ion,step,m,q,Nsigma=2.): 
    
		ax=self.plot_pranges(step,tofs=arange(100,701,1),energies=arange(2,101,1))[3]
		tofpos=tof(step,m,q)
		Epos=ESSD(tofpos,m)
		tofsigma=tofsig(step,m,q)
		Esigma=Esig(Epos)
		print(tofpos,Epos,tofsigma,Esigma )
		tofbounds=[tofpos-Nsigma*tofsigma,tofpos+Nsigma*tofsigma]
		Ebounds=[Epos-Nsigma*Esigma,Epos+Nsigma*Esigma]
		ax.set_title("ion: %s"%(ion))
		ax.plot(tofbounds,[Ebounds[0],Ebounds[0]],color="k",linewidth=2,label="Nsigma=%.2f"%(Nsigma))
		ax.plot(tofbounds,[Ebounds[1],Ebounds[1]],color="k",linewidth=2)
		ax.plot([tofbounds[0],tofbounds[0]],Ebounds,color="k",linewidth=2)
		ax.plot([tofbounds[1],tofbounds[1]],Ebounds,color="k",linewidth=2)
		ax.legend()
		
		
		
	def plot_pranges_mpcm(self,step,tofs,energies,ranges,mpc=arange(1,81,1),m=arange(1,62,1)):#mpc, m in ch        
		"""
		correct indexing!
		"""
		
		#put into general load routine
		self.relate_mpc_tof()
		self.relate_m_E()
		mpcr=self.tof_to_mpq_hefti(step=array([step]*len(tofs)),tof=tofs)
		ranges_sorted=zeros((len(tofs),len(m)))
		i=0
		for tof in tofs:
			print(tof)
			mr_tof=self.tofE_to_m_hefti(tof=array([tof]*len(energies)),ESSD=energies)
			ind=searchsorted(mr_tof,m)-1#not correct yet! Find out, why?!
			
			ranges_tof=ranges[i]
			ranges_tof_sorted=ranges_tof[ind]
			#print ind,len(ind),ranges_tof,len(ranges_tof),ranges_tof[ind],len(ranges_tof[ind])
			#if tof==225:
			#    return m,mr_tof,ranges_tof,ranges_tof_sorted
			#return len(m),len(ind)
			ranges_sorted[i]=ranges_tof_sorted
			i=i+1
		
		
		fig, ax = plt.subplots(1,1)
		Plot=ax.pcolor(mpcr,m,ranges_sorted.T,vmin=0, vmax=max(ravel(ranges_sorted)))
		Plot.cmap.set_under('w')
		ax.set_xlabel("m/q [ch]")       
		ax.set_ylabel("m [ch]")
		ax.set_title("ESA step : %i"%(step))
		self.plot=fig,ax
		cb1 = fig.colorbar(Plot)
		cb1.set_label("priority range number")          
		plt.show()  
		

	def assign_mrbox_prange(self,mbounds="upplow_exc",prange=0):
		"""
		The priority range is assigned by the covered mass channels of a matrix a rate box.
		The fixed priority range mass boundaries are derived from PHA data flags and can be visualized with 
		the "plot_pranges"-routines. 
		Possible mbounds(=mass boundaries) box assignment options are: "upplow_exc", "upplow_inc", "low_inc", "upp_inc".
		upplow_exc: both upper AND lower box boundary have to lie within the priority range.
 		upplow_inc: either upper OR lower box boundary have to lie within the priority range.
		upp_inc: only the upper box boundary has to lie within the priority range.
		low_inc: only the lower box boundary has to lie within the priority range.
		Warning: Option upplow_inc will most likely lead to double assignments of boxes to more than one priority range!
		"""
		
		pranges=array([0,1,2,3,4,5])
		#pr_mrange=array([[0,7],[52,64],[47,52],[41,47],[33,41],[7,33]])#mass intervals are  half-open e.g: [0,8[ for priority range 1
		
		
		self.mrbox_prange=zeros((len(self.mrbox_mpc_ll)))-1
		
		if mbounds=="upplow_exc":
			self.mrbox_prange[self.mrbox_m_ur<7]=pranges[0]#always the same
			self.mrbox_prange[(self.mrbox_m_ll>=7)*(self.mrbox_m_ur<33)]=pranges[5]
			self.mrbox_prange[(self.mrbox_m_ll>=33)*(self.mrbox_m_ur<41)]=pranges[4]
			self.mrbox_prange[(self.mrbox_m_ll>=41)*(self.mrbox_m_ur<47)]=pranges[3]
			self.mrbox_prange[(self.mrbox_m_ll>=47)*(self.mrbox_m_ur<52)]=pranges[2]
			self.mrbox_prange[(self.mrbox_m_ll>=52)*(self.mrbox_m_ur<62)]=pranges[1]#without excess boxes
		
		"""
		elif mbounds=="low_inc":
			self.mrbox_prange[self.mrbox_m_ur<8]=pranges[0]#always the same
			self.mrbox_prange[(self.mrbox_m_ll>=8)*(self.mrbox_m_ll<34)]=pranges[5]
			self.mrbox_prange[(self.mrbox_m_ll>=34)*(self.mrbox_m_ll<42)]=pranges[4]
			self.mrbox_prange[(self.mrbox_m_ll>=42)*(self.mrbox_m_ll<48)]=pranges[3]
			self.mrbox_prange[(self.mrbox_m_ll>=48)*(self.mrbox_m_ll<53)]=pranges[2]
			self.mrbox_prange[(self.mrbox_m_ll>=53)*(self.mrbox_m_ur<62)]=pranges[1]#without excess boxes
		
		elif mbounds=="upp_inc":
			self.mrbox_prange[self.mrbox_m_ur<8]=pranges[0]#always the same
			self.mrbox_prange[(self.mrbox_m_ur>8)*(self.mrbox_m_ur<=34)]=pranges[5]
			self.mrbox_prange[(self.mrbox_m_ur>34)*(self.mrbox_m_ur<=42)]=pranges[4]
			self.mrbox_prange[(self.mrbox_m_ur>42)*(self.mrbox_m_ur<=48)]=pranges[3]
			self.mrbox_prange[(self.mrbox_m_ur>48)*(self.mrbox_m_ur<=53)]=pranges[2]
			self.mrbox_prange[(self.mrbox_m_ur>53)*(self.mrbox_m_ur<62)]=pranges[1]#without excess boxes
		
		
		elif mbounds=="upplow_inc":
			
			if prange==0:
				self.mrbox_prange[self.mrbox_m_ur<8]=pranges[0]#always the same
			elif prange==5:	
				self.mrbox_prange[(self.mrbox_m_ll>=8)*(self.mrbox_m_ll<=34)+(self.mrbox_m_ur>=8)*(self.mrbox_m_ur<=34)]=pranges[5]
			elif prange==4:
				self.mrbox_prange[(self.mrbox_m_ll>=34)*(self.mrbox_m_ll<=42)+(self.mrbox_m_ur>=34)*(self.mrbox_m_ur<=42)]=pranges[4]            
			elif prange==3:
				self.mrbox_prange[(self.mrbox_m_ll>=42)*(self.mrbox_m_ll<=48)+(self.mrbox_m_ur>=42)*(self.mrbox_m_ur<=48)]=pranges[3]            
			elif prange==2:
				self.mrbox_prange[(self.mrbox_m_ll>=48)*(self.mrbox_m_ll<=53)+(self.mrbox_m_ur>=48)*(self.mrbox_m_ur<=53)]=pranges[2]            
			elif prange==1:
				self.mrbox_prange[(self.mrbox_m_ll>53)*(self.mrbox_m_ur<62)+(self.mrbox_m_ur>53)*(self.mrbox_m_ur<62)]=pranges[1]#without excess boxes
		"""
		return True
		
	
	def calc_PRbaserates(self,prange,timestamp,mbounds="upplow_exc",ESA_stopstep=None):#check boundaries more exact!

		#select priority range assignment option for matrix boxes 
		self.assign_mrbox_prange(mbounds=mbounds,prange=prange)

		steprange=arange(0,116,1)
		pr_steps=array([])
		pr_counts=array([])		
		prboxes=self.mrbox_number[self.mrbox_prange==prange]
		Steps=zeros((len(prboxes),len(steprange)))
		PR_counts=zeros((len(prboxes),len(steprange)))

		print("timestamp",timestamp)
		print("boxes",prboxes)
		
		#prboxes=prboxes[14:15]
		
		i=0
		for mrbox in prboxes: 
				print("mrbox",mrbox)
			
				#calculate matrix rates center step
				cstep=around(self.calc_mr_centerstep(mrbox,timestamp))
			
				#select desired matrix rate counts
				timemask=(self.times_cmr_sync==timestamp)
				boxmask=(self.number_cmr_sync==mrbox)		
				mr=arange(0,21,1.)
				allcounts_mr=self.matrix_rates.T[timemask*boxmask][0]
				allcounts_mr[allcounts_mr<0]=0.
			
				#ESA_stopstep=cstep+5#test
			
				if ESA_stopstep!=None:
					

					#counts_full
					steps_full=arange(cstep-2,cstep+3,1)
					counts_full=array([allcounts_mr[8],allcounts_mr[9],allcounts_mr[10],allcounts_mr[11],allcounts_mr[12]])
					print("full",counts_full)
					
					#counts_half
					steps_half=array([[cstep-8,cstep-7],[cstep-6,cstep-5],[cstep-4,cstep-3],[cstep+3,cstep+4],[cstep+5,cstep+6],[cstep+7,cstep+8]])
					counts_half=array([zeros((2))+allcounts_mr[5]/2.,zeros((2))+allcounts_mr[6]/2.,zeros((2))+allcounts_mr[7]/2.,zeros((2))+allcounts_mr[13]/2.,zeros((2))+allcounts_mr[14]/2.,zeros((2))+allcounts_mr[15]/2])
					print(allcounts_mr)
					print("2_before",steps_half, counts_half)
					
					if ESA_stopstep in steps_half:
						indstop=concatenate(array(where(steps_half==ESA_stopstep)))
						stopmask=(steps_half[indstop[0]]<=ESA_stopstep)
						invstopmask=invert(stopmask)
						#print stopmask,invstopmask
						#print steps_half,counts_half,indstop[0],concatenate(array(where(steps_half==ESA_stopstep)))
						redist_counts=sum(counts_half[indstop[0]])/float(len(counts_half[indstop[0]][stopmask]))
						#print redist_counts
						counts_half[indstop[0]][stopmask]=redist_counts
						counts_half[indstop[0]][invstopmask]=0
						#return counts_half
					print("2_after",steps_half,counts_half)
					
					
					#counts_quart
					steps_quart=array([arange(cstep-16,cstep-12,1),arange(cstep-12,cstep-8,1),arange(cstep+9,cstep+13,1),arange(cstep+13,cstep+17,1)])
					counts_quart=array([zeros((4))+allcounts_mr[3]/4.,zeros((4))+allcounts_mr[4]/4.,zeros((4))+allcounts_mr[16]/4.,zeros((4))+allcounts_mr[17]/4.])
					print("4_before",steps_quart,counts_quart)
					
					if ESA_stopstep in steps_quart:
						indstop=concatenate(array(where(steps_quart==ESA_stopstep)))
						stopmask=(steps_quart[indstop[0]]<=ESA_stopstep)
						invstopmask=invert(stopmask)
						#print stopmask,invstopmask
						#print steps_quart,counts_half,indstop[0],concatenate(array(where(steps_quart==ESA_stopstep)))
						redist_counts=sum(counts_quart[indstop[0]])/float(len(counts_quart[indstop[0]][stopmask]))
						#print redist_counts
						counts_quart[indstop[0]][stopmask]=redist_counts
						counts_quart[indstop[0]][invstopmask]=0
						#return counts_quart
					print("4_after",steps_quart,counts_quart)
					
					
					
					#counts_eight
					steps_eight=array([arange(cstep-32,cstep-24),arange(cstep-24,cstep-16,1),arange(cstep+17,cstep+25),arange(cstep+25,cstep+33,1)])
					counts_eight=array([zeros((8))+allcounts_mr[1]/8.,zeros((8))+allcounts_mr[2]/8.,zeros((8))+allcounts_mr[18]/8.,zeros((8))+allcounts_mr[19]/8.])
					print("8_before",steps_eight,counts_eight)
					
					if ESA_stopstep in steps_eight:
						indstop=concatenate(array(where(steps_eight==ESA_stopstep)))
						stopmask=(steps_eight[indstop[0]]<=ESA_stopstep)
						invstopmask=invert(stopmask)
						#print stopmask,invstopmask
						#print steps_eight,counts_half,indstop[0],concatenate(array(where(steps_eight==ESA_stopstep)))
						redist_counts=sum(counts_eight[indstop[0]])/float(len(counts_eight[indstop[0]][stopmask]))
						#print redist_counts
						counts_eight[indstop[0]][stopmask]=redist_counts
						counts_eight[indstop[0]][invstopmask]=0
						#return counts_eight
					print("8_after",steps_eight,counts_eight)
					
					
					#counts_rest
					
					if (cstep-32)>0:
						len_lowrest=(cstep-32)-0.
					else:
						len_lowrest=0
					if (116.-(cstep+33))>0:
						len_uprest=116.-(cstep+33)
					else:
						len_uprest=0
					steps_rest=array([arange(cstep-32-len_lowrest,cstep-32),arange(cstep+33,cstep+33+len_uprest)])
					counts_rest=array([zeros((len_lowrest))+allcounts_mr[0]/len_lowrest,zeros((len_uprest))+allcounts_mr[20]/len_uprest])
					print("rest_before",steps_rest,counts_rest)
					
					if ESA_stopstep in steps_rest:
						indstop=concatenate(array(where(steps_rest==ESA_stopstep)))
						stopmask=(steps_rest[indstop[0]]<=ESA_stopstep)
						invstopmask=invert(stopmask)
						#print stopmask,invstopmask
						#print steps_rest,counts_half,indstop[0],concatenate(array(where(steps_rest==ESA_stopstep)))
						redist_counts=sum(counts_rest[indstop[0]])/float(len(counts_rest[indstop[0]][stopmask]))
						#print redist_counts
						counts_rest[indstop[0]][stopmask]=redist_counts
						counts_rest[indstop[0]][invstopmask]=0
						#return counts_rest
					print("rest_after",steps_rest,counts_rest)

					counts_mr=concatenate([counts_rest[0],counts_eight[0],counts_eight[1],counts_quart[0],counts_quart[1],counts_half[0],counts_half[1],counts_half[2],counts_full,counts_half[3],counts_half[4],counts_half[5],counts_quart[2],counts_quart[3],counts_eight[2],counts_eight[3],counts_rest[1]])	
					
					steps_mr=concatenate([steps_rest[0],steps_eight[0],steps_eight[1],steps_quart[0],steps_quart[1],steps_half[0],steps_half[1],steps_half[2],steps_full,steps_half[3],steps_half[4],steps_half[5],steps_quart[2],steps_quart[3],steps_eight[2],steps_eight[3],steps_rest[1]])
					
					#return counts_mr,steps_mr
					
					#elif ESA_stopstep in steps_quart:	

					
					
				
				else:
					#reconstruct counts-per-ESA-step from matrix rates counts (after Hefti PhD thesis [1995])
					counts_full=array([allcounts_mr[8],allcounts_mr[9],allcounts_mr[10],allcounts_mr[11],allcounts_mr[12]])
					counts_half=array([zeros((2))+allcounts_mr[5]/2.,zeros((2))+allcounts_mr[6]/2.,zeros((2))+allcounts_mr[7]/2.,zeros((2))+allcounts_mr[13]/2.,zeros((2))+allcounts_mr[14]/2.,zeros((2))+allcounts_mr[15]/2])	
					counts_quart=array([zeros((4))+allcounts_mr[3]/4.,zeros((4))+allcounts_mr[4]/4.,zeros((4))+allcounts_mr[16]/4.,zeros((4))+allcounts_mr[17]/4.])
					counts_eight=array([zeros((8))+allcounts_mr[1]/8.,zeros((8))+allcounts_mr[2]/8.,zeros((8))+allcounts_mr[18]/8.,zeros((8))+allcounts_mr[19]/8.])
					if (cstep-32)>0:
						len_lowrest=(cstep-32)-0.
					else:
						len_lowrest=0
					if (116.-(cstep+33))>0:
						len_uprest=116.-(cstep+33)
					else:
						len_uprest=0
					counts_rest=array([zeros((len_lowrest))+allcounts_mr[0]/len_lowrest,zeros((len_uprest))+allcounts_mr[20]/len_uprest])
			
					counts_mr=concatenate([counts_rest[0],counts_eight[0],counts_eight[1],counts_quart[0],counts_quart[1],counts_half[0],counts_half[1],counts_half[2],counts_full,counts_half[3],counts_half[4],counts_half[5],counts_quart[2],counts_quart[3],counts_eight[2],counts_eight[3],counts_rest[1]])		
			
				#reconstruct ESA-step from matrix rate step definition (after Hefti PhD thesis [1995])
				steps_full=arange(cstep-2,cstep+3,1)
				steps_half=array([arange(cstep-8,cstep-2,1),arange(cstep+3,cstep+9,1)])
				steps_quart=array([arange(cstep-16,cstep-8,1),arange(cstep+9,cstep+17,1)])
				steps_eight=array([arange(cstep-32,cstep-16,1),arange(cstep+17,cstep+33,1)])
				steps_rest=array([arange(cstep-32-len_lowrest,cstep-32),arange(cstep+33,cstep+33+len_uprest)])
				steps_mr=concatenate([steps_rest[0],steps_eight[0],steps_quart[0],steps_half[0],steps_full,steps_half[1],steps_quart[1],steps_eight[1],steps_rest[1]])
				steps_mr=steps_mr[(steps_mr>=0)*(steps_mr<=115)]
				counts_mr=counts_mr[(steps_mr>=0)*(steps_mr<=115)]
			
				PR_counts[i]=counts_mr	
				Steps[i]=steps_mr
				i+=1

		print("number of matrix rate boxes withinin selected priority range:", len(prboxes))
		PR_counts_sum=sum(PR_counts,axis=0)

		#return Steps,PR_counts,PR_counts_sum      
		return PR_counts_sum
		
	
	
	def calc_boxbaserates(self,mrbox,timestamp):

		print("timestamp",timestamp)
		
		#calculate matrix rates center step
		cstep=around(self.calc_mr_centerstep(mrbox,timestamp))
		#cstep=calc_centerstep_hefti(self)
		#calc_centerstep_hefti(self,v_Fe=None,round_step=False,cs_shift=None)
		
		#select desired matrix rate counts
		timemask=(self.times_cmr_sync==timestamp)
		boxmask=(self.number_cmr_sync==mrbox)		
		mr=arange(0,21,1.)
		allcounts_mr=self.matrix_rates.T[timemask*boxmask][0]
		allcounts_mr[allcounts_mr<0]=0.
		
		
		#reconstruct counts-per-ESA-step from matrix rates counts (after Hefti PhD thesis [1995])
		counts_full=array([allcounts_mr[8],allcounts_mr[9],allcounts_mr[10],allcounts_mr[11],allcounts_mr[12]])
		counts_half=array([zeros((2))+allcounts_mr[5]/2.,zeros((2))+allcounts_mr[6]/2.,zeros((2))+allcounts_mr[7]/2.,zeros((2))+allcounts_mr[13]/2.,zeros((2))+allcounts_mr[14]/2.,zeros((2))+allcounts_mr[15]/2])	
		counts_quart=array([zeros((4))+allcounts_mr[3]/4.,zeros((4))+allcounts_mr[4]/4.,zeros((4))+allcounts_mr[16]/4.,zeros((4))+allcounts_mr[17]/4.])
		counts_eight=array([zeros((8))+allcounts_mr[1]/8.,zeros((8))+allcounts_mr[2]/8.,zeros((8))+allcounts_mr[18]/8.,zeros((8))+allcounts_mr[19]/8.])
		if (cstep-32)>0:
			len_lowrest=(cstep-32)-0.
		else:
			len_lowrest=0
		if (116.-(cstep+33))>0:
			len_uprest=116.-(cstep+33)
		else:
			len_uprest=0
		counts_rest=array([zeros((len_lowrest))+allcounts_mr[0]/len_lowrest,zeros((len_uprest))+allcounts_mr[20]/len_uprest])
		
		counts_mr=concatenate([counts_rest[0],counts_eight[0],counts_eight[1],counts_quart[0],counts_quart[1],counts_half[0],counts_half[1],counts_half[2],counts_full,counts_half[3],counts_half[4],counts_half[5],counts_quart[2],counts_quart[3],counts_eight[2],counts_eight[3],counts_rest[1]])		
		
		
		Boxcounts_mr=counts_mr	
		mr_steps=array([0,cstep-32,cstep-24,cstep-16,cstep-12,cstep-8,cstep-6,cstep-4,cstep-2, cstep-1,cstep,cstep+1,cstep+2,cstep+3, cstep+5,cstep+7,cstep+9,cstep+13,cstep+17,cstep+25,cstep+33,116])
		mr_steps_diff=mr_steps[1:]-mr_steps[:-1]
		mr_counts=allcounts_mr
		
		return Boxcounts_mr,cstep,mr_steps[:-1],mr_counts,mr_steps_diff

	
	
	def calc_PRPHA(self,prange,timestamp,steprange=arange(0,116+1,1)):
			  
			timemask=(self.times==timestamp)
			prange_mask=(self.range==prange)      
			stepdata_masked=self.step[timemask*prange_mask]

			h=histogram(stepdata_masked,steprange)
			PR_counts=h[0]
	   
			return PR_counts  
		
		
	def compare_PHAMR_PRcounts(self,prange,timestamp,steprange=arange(0,116,1),MRstep_offset=0,Plot=False,mbounds="upplow_exc"):
		
			PR_counts_MR=self.calc_PRbaserates(prange=prange,timestamp=timestamp,mbounds=mbounds)
			
			#return PR_counts_MR
			
			PR_counts_PHA=self.calc_PRPHA(prange=prange,timestamp=timestamp,steprange=append(steprange,steprange[-1]+1))
			#PR_counts_PHA_before=self.calc_PRPHA(prange=prange,timestamp=timestamp_before,steprange=append(steprange,steprange[-1]+1))
			#PR_counts_PHA_after=self.calc_PRPHA(prange=prange,timestamp=timestamp_after,steprange=append(steprange,steprange[-1]+1))
			
			PR_counts_PHA_R2=self.calc_PRPHA(prange=2,timestamp=timestamp,steprange=append(steprange,steprange[-1]+1))
			PR_counts_MR_R2=self.calc_PRbaserates(prange=2,timestamp=timestamp,mbounds=mbounds)
			
			counts_PHA_allranges=zeros((len(steprange)))
			i=0
			while i<=5:
				counts_PHA_prange=self.calc_PRPHA(prange=i,timestamp=timestamp,steprange=append(steprange,steprange[-1]+1)) 
				counts_PHA_allranges=counts_PHA_allranges+counts_PHA_prange
				i+=1
			maxcounts_PHA_allranges=max(counts_PHA_allranges)
			
			
			if Plot==True:
				plt.figure()
				print(timestamp)
				plt.title("priority range: %i, time: %.5f DOY 1996"%(prange,timestamp))
				plt.plot(steprange,PR_counts_PHA,color="b",linewidth=2,label="PHA")
				plt.plot(steprange+MRstep_offset,PR_counts_MR,color="r",linewidth=2,label="BR, %s"%(mbounds))
				plt.plot(steprange+MRstep_offset,1.*PR_counts_MR/PR_counts_PHA,color="k",linewidth=2,label="BR factors")
				plt.plot(steprange,counts_PHA_allranges,color="c",linewidth=2,label="PHA allranges")
				#plt.plot(steprange,PR_counts_PHA_R2,color="g",linewidth=2,label="PHA range 2")
				#plt.plot(steprange,PR_counts_MR_R2,color="m",linewidth=2,label="BR range 2")
				
				#plt.plot(steprange,PR_counts_PHA_before,color="g",linewidth=2,label="PHA rates_before")
				#plt.plot(steprange,PR_counts_PHA_after,color="c",linewidth=2,label="PHA rates_after")
				
				
				plt.xlabel("ESA step")
				plt.ylabel("counts")
				plt.legend()
				plt.show()
		
			return PR_counts_MR,PR_counts_PHA,maxcounts_PHA_allranges


	def return_PHAMR_multiPR(self,pranges,timestamp,steprange=arange(0,116,1),MRstep_offset=0,ESA_stopstep=None,Nbins_redist=2,Plot=False,mbounds="upplow_exc"):
		
			
			counts_MR_allranges=zeros((4,len(steprange)))
			counts_PHA_allranges=zeros((4,len(steprange)))
			i=0
			for PR in pranges:
				PR_counts_MR=self.calc_PRbaserates(prange=PR,timestamp=timestamp,mbounds=mbounds,ESA_stopstep=ESA_stopstep)
				PR_counts_PHA=self.calc_PRPHA(prange=PR,timestamp=timestamp,steprange=append(steprange,steprange[-1]+1))
				
				#only to test, has to be solved properly before publication(!):
				stopmask=[steprange>ESA_stopstep]
				#invstopmask=invert(stopmask)
				MRcounts_redist=float(sum(PR_counts_MR[stopmask]))
				PR_counts_MR[stopmask]=0
				PR_counts_MR[ESA_stopstep+1-Nbins_redist:ESA_stopstep+1]=PR_counts_MR[ESA_stopstep+1-Nbins_redist:ESA_stopstep+1]+round(MRcounts_redist/Nbins_redist)
				
				counts_MR_allranges[i]=PR_counts_MR
				counts_PHA_allranges[i]=PR_counts_PHA
				i=i+1				
			
			return steprange,counts_MR_allranges,counts_PHA_allranges





	def compare_PHAMR_boxcounts(self,mrbox,timestamp,steprange=arange(0,116,1),MRstep_offset=0,Plot=False):
		
		Boxcounts_mr,cstep,mr_steps,mr_counts,mr_steps_diff=self.calc_boxbaserates(mrbox=mrbox,timestamp=timestamp)

		boxcounts_PHA=zeros((len(steprange)))
		step=steprange[0]
		j=0
		while step <=steprange[-1]:
			print(step)
			tof_bounds,E_bounds=self.find_tofEbox(step=step,mrbox_number=mrbox)
			boxcounts_PHA[j]=self.countfilter_PHA(timerange=[timestamp,timestamp],step=step,tofrange=tof_bounds,Erange=E_bounds,Plot=False)
			step+=1    
			j+=1
				  

		if Plot==True:
			plt.figure()
			print(timestamp)
			plt.title("mrbox: %i, time: %.5f DOY 1996"%(mrbox,timestamp))
			plt.bar(steprange-0.5,boxcounts_PHA,width=1.0,color="b",linewidth=2,alpha=0.5,label="PHA")
			#plt.bar(steprange-0.5+MRstep_offset,boxcounts_MR,color="r",alpha=0.5,linewidth=2,label="BR")
			
			plt.bar(mr_steps-0.5+MRstep_offset,mr_counts/mr_steps_diff, width=mr_steps_diff, color="r", alpha=0.5, linewidth=2,label="BR")
			
			
			plt.plot([cstep+MRstep_offset,cstep+MRstep_offset],[0,max(Boxcounts_mr)+0.1*max(Boxcounts_mr)],linewidth=3,color="r",label="cstep")
			#plt.plot(steprange+MRstep_offset,1.*PR_counts_MR/PR_counts_PHA,color="k",linewidth=2,label="BR factors")
			#plt.plot(steprange,counts_PHA_allranges,color="c",linewidth=2,label="PHA allranges")
			#plt.plot(steprange,PR_counts_PHA_R2,color="g",linewidth=2,label="PHA range 2")
			#plt.plot(steprange,PR_counts_MR_R2,color="m",linewidth=2,label="BR range 2")
	
			#plt.plot(steprange,PR_counts_PHA_before,color="g",linewidth=2,label="PHA rates_before")
			#plt.plot(steprange,PR_counts_PHA_after,color="c",linewidth=2,label="PHA rates_after")
	
	
			plt.xlabel("ESA step")
			plt.ylabel("counts")
			plt.legend()
			plt.axis([0,117,0,max(Boxcounts_mr)+0.1*max(Boxcounts_mr)])
			plt.show()

		return steprange,boxcounts_PHA,Boxcounts_mr,mr_steps,mr_steps_diff,mr_counts



	def reconstruct_Epq(self,cstep,MR):
		"""
		MR:= matrix rates for a given matrix rate box, for the same time stamp as the given centerstp 
		len(MR)=21
		len(Epq-steps)=116#check whether it should be 117
		"""
		step_shifts=[range(-116,-32),[-32,-31,-30,-29,-28,-27,-26,-25],[-24,-23,-22,-21,-20,-19,-18,-17],[-16,-15,-14,-13],[-12,-11,-10,-9],[-8,-7],[-6,-5],[-4,-3],[-2],[-1],[0],[1],[2],[3,4],[5,6],[7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20,21,22,23,24],[25,26,27,28,29,30,31,32],range(33,116)]
		

		#self.epq_rates = zeros((self.centersteps.shape[0],116, self.centersteps.shape[1]))
		#self.iB = array(range(508)*self.uT.shape[0])
		#self.iCEpq = self.centersteps[self.riT,self.iB]
		steps_epq=[]
		counts_epq=[]
		for M in range(len(MR)):
			steps=cstep+array(step_shifts[M])
			mask = ((steps>=0)*(steps<116))
			steps_valid=steps[mask]
			norm=float((len(steps_valid)))
			
			counts=MR[M]
			counts_valid=zeros((len(steps_valid)))+counts/norm
			
			steps_epq.append(steps_valid)
			counts_epq.append(counts_valid)
			
		steps_epq=array(concatenate(steps_epq))	
		counts_epq=array(concatenate(counts_epq))
		return steps_epq,counts_epq
			
			
		"""	
			iM = ones(self.iB.size,dtype=int)*M
			norm = zeros(self.iB.size,dtype=float64)
			for s in epq_shifts[M]:
				iEpq = cstep + s
				mask = ((iEpq>=0)*(iEpq<116))
				norm[mask]+=1
			for s in self.mrshifts[M]:
				iEpq = self.iCEpq + s
				mask = ((iEpq>=0)*(iEpq<116))
				self.epq_rates[self.riT[mask],iEpq[mask],self.iB[mask]]+=self.MR[self.riT[mask],iM[mask],self.iB[mask]]/norm[mas
		"""








	def calc_mr_velmoments(self,mrbox_numbers,timestamp,m,q,Plot=False,compression="high_res"):
		
		steprange=arange(0,116,1)
		#mr_steps=array([])
		#mr_counts=array([])		
		mrboxes=mrbox_numbers
		MR_counts=zeros((len(mrboxes),len(steprange)))
		MR_steps=zeros((len(mrboxes),len(steprange)))
		MR_vels=zeros((len(mrboxes),len(steprange)))

		i=0
		for mrbox in mrboxes: 
		
				print("timestamp",timestamp)
		
				#calculate matrix rates center step
				cstep=around(self.calc_mr_centerstep(mrbox,timestamp))
				print("cstep check: %f"%(cstep))
		
				if (cstep>=0) and (cstep<116):
				
					#select desired matrix box rate counts
					timemask=(self.times_cmr_sync==timestamp)
					boxmask=(self.number_cmr_sync==mrbox)		
					mr=arange(0,21,1.)
					if len(where(timemask==True)[0])==508:#cut out not fully filled timestamps
						allcounts_mr=self.matrix_rates.T[timemask*boxmask][0]
					else:
						allcounts_mr=array([-999]*21)
					allcounts_mr[allcounts_mr<0]=0.

					if compression=="low_res":
						counts_mr=zeros((11))
						j=0
						while j<21:
							counts_mr=concatenate([counts_mr,zeros((5))+allcounts_mr[j]/5.])
							print(counts_mr)
							j+=1

					else:
						counts_mr=self.reconstruct_Epq(cstep=cstep,MR=allcounts_mr)[1]	
						MR_counts[i]=counts_mr
			 			#return "test",counts_mr
			 			
						
				else:
					counts_mr=zeros((116))-1.
				
				steps_mr=arange(0,116,1)
				MR_steps[i]=steps_mr
				vels_mr=step_to_vel(steps_mr,q_e=q,m_amu=m)
				MR_vels[i]=vels_mr
				i+=1

		MR_counts_sum=sum(MR_counts,axis=0)
		indmax=where(MR_counts_sum==max(MR_counts_sum))[0]
		if len(shape(indmax))>0:
			indmax=indmax[0]
		velmax=MR_vels[0][indmax]
		
		#phase_space and real space correction
		MR_counts_sum_cor=ps_correction(N=MR_counts_sum,v=MR_vels[0],v_ref=velmax)#phase space correction
		MR_counts_sum_cor2=ps2_correction(N=MR_counts_sum,v=MR_vels[0],v_ref=velmax)#combined phase and real space correction
		
		#calculate mean speeds
		if max(MR_counts_sum)>0:
			MR_vmean=average(MR_vels[0],weights=MR_counts_sum)	
			MR_vmean_pscor=average(MR_vels[0],weights=MR_counts_sum_cor)	
			MR_vmean_ps2cor=average(MR_vels[0],weights=MR_counts_sum_cor2)	
			
		else:
			MR_vmean=-999
			MR_vmean_pscor=-999
			MR_vmean_ps2cor=-999
		
		if Plot==True:
			plt.figure()
			plt.plot(MR_vels[0],MR_counts_sum,color="b")
			plt.plot([MR_vmean,MR_vmean],[0,max(MR_counts_sum)],color="b",label="MR_vmean")
			plt.legend(loc="upper right")
			plt.show()
		
		return MR_steps,MR_counts,MR_vels[0],MR_counts_sum,MR_vmean,MR_vmean_pscor,MR_vmean_ps2cor
				

	def calc_mr_velmoments_timeseries(self,timerange,mrbox_numbers,m=56.,q=9,savedata=False,path="/home/asterix/janitzek/ctof/MR_results/",filename="test_pscor",compression="high_res",pscor=False):
		
		#synchronize CTOF_MR and PM data products
		utimes=unique(self.times_cmr_sync)
		utimes_pm=unique(self.times)
		utimes_sync=intersect1d(utimes,utimes_pm)
		timemask_range=(utimes_sync>=timerange[0])*(utimes_sync<timerange[-1])
		utimes_valid=utimes_sync[timemask_range]
		
		ss=searchsorted(self.times_cmr_sync,utimes_valid)
		uvFE_valid=self.vsw_cmr_sync[ss]
		sspm=searchsorted(self.times,utimes_valid)
		uvsw_pm_valid=self.vsw[sspm]#vsw speed from pm measurements (that were synchronized with CTOF pha data) 
		
		#adjust mass and charge automatically for Hefti boxes		
		if mrbox_numbers==[235]:
			m=16.
			q=6.
		elif mrbox_numbers==[201]:
			m=28.
			q=7.
		elif mrbox_numbers==[41,92,93] or mrbox_numbers==[41] or mrbox_numbers==[92] or mrbox_numbers==[93]:
			m=56.
			q=9.
		else:
			m=0.	
			q=0.

		#calculate mean speeds 
		mr_vmeans_valid=[]
		mr_vmeanscor_valid=[]
		mr_vmeanscor2_valid=[]
		mr_vels=[]
		mr_counts_sum=[]
		utimes_v=[]
		for day in arange(int(timerange[0]),int(timerange[-1])+1,1):
			timemask_day=(utimes_valid>=day)*(utimes_valid>=timerange[0])*(utimes_valid<day+1)*(utimes_valid<timerange[-1])
			utimes_valid_day=utimes_valid[timemask_day]
			mr_vmeans_valid_day=zeros((len(utimes_valid_day)))
			mr_vmeanscor_valid_day=zeros((len(utimes_valid_day)))
			mr_vmeanscor2_valid_day=zeros((len(utimes_valid_day)))
			mr_vels_day=zeros((len(utimes_valid_day),116))
			mr_counts_sum_day=zeros((len(utimes_valid_day),116))

			i=0
			while i<len(utimes_valid_day):
				mr_vels_day[i],mr_counts_sum_day[i],mr_vmeans_valid_day[i] ,mr_vmeanscor_valid_day[i],mr_vmeanscor2_valid_day[i]=self.calc_mr_velmoments(mrbox_numbers=mrbox_numbers,timestamp=utimes_valid_day[i],m=m,q=q,compression=compression)[2:]
				i=i+1
	
			mr_vmeans_valid.append(mr_vmeans_valid_day)
			mr_vmeanscor_valid.append(mr_vmeanscor_valid_day)
			mr_vmeanscor2_valid.append(mr_vmeanscor2_valid_day)
			mr_vels.append(mr_vels_day)	
			mr_counts_sum.append(mr_counts_sum_day)	
			utimes_v.append(utimes_valid_day)
				 	
		mr_vmeans_valid=concatenate(mr_vmeans_valid)
		mr_vmeanscor_valid=concatenate(mr_vmeanscor_valid)
		mr_vmeanscor2_valid=concatenate(mr_vmeanscor2_valid)
		mr_vels=concatenate(mr_vels)
		mr_counts_sum=concatenate(mr_counts_sum)
		utimes_v=concatenate(utimes_v)
 
 		#save data, if desired
		if savedata==True:
			outfile=path+filename
			outdata=array([utimes_valid,uvsw_pm_valid,uvFE_valid,mr_vmeans_valid,mr_vmeanscor_valid,mr_vmeanscor2_valid])
			#return outdata
			with open(outfile, 'wb') as f:
				f.write("timestamp [DOY 1996]	vsw_pm [km/s]	vFE_onboard_est [km/s] vion_uncor [km/s]	vion_pscor [km/s]	vion_ps2cor [km/s]\n")
				savetxt(f, outdata.T, fmt='%.5f', delimiter=' ', newline='\n')
			f.close()	
		
		return utimes_valid,uvsw_pm_valid,uvFE_valid,mr_vmeans_valid,mr_vmeanscor_valid, mr_vmeanscor2_valid,mr_vels,mr_counts_sum
		
		
		

	def check_PHAMR(self,prange,timestamp,step,steprange=arange(0,116+1,1)):	

		timemask=(self.times==timestamp)
		prange_mask=(self.range==prange)      
		stepmask=(self.step==step)
		
		steps_masked=self.step[timemask*prange_mask*stepmask]
		tofs_masked=self.tof[timemask*prange_mask*stepmask]
		energies_masked=self.energy[timemask*prange_mask*stepmask]

		return steps_masked,tofs_masked,energies_masked

	def calc_brfactors(self,timerange,prange=1,steprange=arange(0,116,1),Plot=False,outfile_name="baserates",mbounds="upplow_exc",Plot_factors=False):
		"""
		minimum timerange: 1 day
		"""
		#timestamps_all=intersect1d(self.times,self.times_cmr_sync)
		#timestamps=timestamps_all[(timestamps_all>=timerange[0])*(timestamps_all<timerange[-1])]
		
		
		
		
		
		days=arange(int(timerange[0]),int(timerange[-1])+1,1)
		for day in days:
		
			t0=clock()	
			
			d=ctoflv1([[day,day+1]],[0,1440])
			e=d.prepare_CTOFdata()
			
			timestamps=intersect1d(d.times,d.times_cmr_sync)
			#timestamps=timestamps_all[(timestamps_all>=timerange[0])*(timestamps_all<timerange[-1])]
			print(timestamps)
			
			t1=clock()
			
			outfile=outfile_name+"_day%i"%(day)	
			
			timestamps_day=array([])
			prange_day=array([])
			steprange_day=array([])
			br_factors_day=array([])
		
			timestamps=timestamps[timestamps<d.times_cmr_sync[-1]]
			for timestamp in timestamps:
				print(timestamp)
		
				vsw=unique(d.vsw[d.times==timestamp])
		
				def vstd_Fe(vsw):#estimation for vstd for Fe10+ 
					return 0.16*vsw-34
				vstd_Fe=vstd_Fe(vsw)	
		
				vsw_max=vsw+2*vstd_Fe
				vsw_min=vsw-2*vstd_Fe

	

				#step_min=vel_to_step(m=m,q=q,v=vsw_max)
				#step_mean=vel_to_step(m=m,q=q,v=vsw)
				#step_max=vel_to_step(m=m,q=q,v=vsw_min)
				
				step_min=0
				step_max=116
		
				xrangedata=array([vsw_min,vsw,vsw_max,step_min,step_max])
				#return vsw,vsw_min,vsw_max,step_min,step_max  		
				print(xrangedata)
		
				PR_counts_MR,PR_counts_PHA,maxcounts_PHA_allranges=d.compare_PHAMR_PRcounts(prange=prange,timestamp=timestamp,steprange=steprange,Plot=Plot,mbounds=mbounds)
				PR_counts_PHA=1*PR_counts_PHA 	
			
				"""
				try:
					steprange_valid=steprange[(steprange>=step_min)*(steprange<step_max)]
				except ValueError:
					return vsw,step_min,step_mean,step_max, steprange_valid,steprange 
				"""
			
				#if maxcounts_PHA_allranges>100:
				if maxcounts_PHA_allranges>0:
                
				
					PR_counts_MR_valid=PR_counts_MR[(steprange>=step_min)*(steprange<step_max)]
					PR_counts_PHA_valid=PR_counts_PHA[(steprange>=step_min)*(steprange<step_max)]
		
					PR_counts_PHA_valid[PR_counts_PHA_valid==0]=-1
					brfactors=PR_counts_MR_valid/PR_counts_PHA_valid
					brfactors[brfactors<0]=-1
			
					timestamps_t=array([timestamp]*len(steprange))
					prange_t=array([prange]*len(steprange))
					steprange_t=steprange
					br_factors_t=brfactors
			
				else:
					timestamps_t=array([timestamp]*len(steprange))
					prange_t=array([prange]*len(steprange))
					steprange_t=steprange
					br_factors_t=zeros((len(steprange)))-1.
			
				timestamps_day=append(timestamps_day,timestamps_t)
				prange_day=append(prange_day,prange_t)
				steprange_day=append(steprange_day,steprange_t)
				br_factors_day=append(br_factors_day,br_factors_t)
			
				if Plot_factors==True:
					print("test_plot")
					plt.figure()
					plt.title("timestamp: %.5f"%(timestamp))
					plt.plot(arange(0,116),PR_counts_MR,color="r",label="BR")
					plt.plot(arange(0,116),PR_counts_PHA,color="b",label="PHA")
					plt.plot(arange(0,116),br_factors_t,color="k",label="BR factors")
					plt.legend()
					plt.show()
					return True
		
			t2=clock()

			outdata=array([timestamps_day,prange_day,steprange_day,br_factors_day])
			savetxt(outfile, outdata.T, fmt='%.5f', delimiter=' ', newline='\n')
	
            
			print("(calculation time)/day, day:", t2-t1, t1-t0, t2-t0, day)
		
		print("test now")
		
		#return br_factors_day
		return True
	
	
	def calc_brfactors_allPR(self,timerange,savedata=True,output_path="BRfactors/",outfile_name="test",steprange=arange(0,116,1),mbounds="upplow_exc",Plot=False,Plot_factors=False,pranges=[0,1,2,3,4,5],timestamps_test=None):
		"""
		minimum timerange: 1 day
		"""
		#timestamps_all=intersect1d(self.times,self.times_cmr_sync)
		#timestamps=timestamps_all[(timestamps_all>=timerange[0])*(timestamps_all<timerange[-1])]



		days=arange(int(timerange[0]),int(timerange[-1])+1,1)
		for day in days:

				t0=clock()  

				d=ctoflv1([[day,day+1]],[0,1440])#for long-term data processing
				e=d.prepare_CTOFdata()#for long-term data processing 

				timestamps=intersect1d(d.times,d.times_cmr_sync)
				#timestamps=timestamps_all[(timestamps_all>=timerange[0])*(timestamps_all<timerange[-1])]
				print(timestamps)

				t1=clock()

				outfile=outfile_name+"_day%i"%(day) 

				timestamps_day=array([])
				#prange_day=array([])
				steprange_day=array([])
				br_factors_day=zeros((6,0))
				allPR_PHA_day=array([])
				
				timestamps=timestamps[timestamps<d.times_cmr_sync[-1]]
				for timestamp in timestamps:
				#for timestamp in timestamps_test:#test
						#print timepr_stepsstamp

						step_min=0
						step_max=116

						"""
						vsw=unique(d.vsw[d.times==timestamp])

						def vstd_Fe(vsw):#estimation for vstd for Fe10+ 
								return 0.16*vsw-34
						vstd_Fe=vstd_Fe(vsw)    

						vsw_max=vsw+2*vstd_Fe
						vsw_min=vsw-2*vstd_Fe



						#step_min=vel_to_step(m=m,q=q,v=vsw_max)
						#step_mean=vel_to_step(m=m,q=q,v=vsw)
						#step_max=vel_to_step(m=m,q=q,v=vsw_min)


						xrangedata=array([vsw_min,vsw,vsw_max,step_min,step_max])
						#return vsw,vsw_min,vsw_max,step_min,step_max       
						print xrangedata
						"""

						timestamps_t=array([timestamp]*len(steprange))
						steprange_t=steprange

						#iterate over priority ranges
						prange=0
						br_factors_t=zeros((6,len(steprange)))-1
						allPR_counts_PHA=zeros((6))
						
						for prange in pranges:
								PR_counts_MR,PR_counts_PHA,maxcounts_PHA_allranges=d.compare_PHAMR_PRcounts(prange=prange,timestamp=timestamp,steprange=steprange,Plot=Plot,mbounds=mbounds)
								PR_counts_PHA=1*PR_counts_PHA   
								allPR_counts_PHA[prange]=sum(PR_counts_PHA)
								"""    
								try:
										steprange_valid=steprange[(steprange>=step_min)*(steprange<step_max)]
								except ValueError:
										return vsw,step_min,step_mean,step_max, steprange_valid,steprange 
								"""

								#if maxcounts_PHA_allranges>100:# include only cases where the iron baserates behave as supposed
								if maxcounts_PHA_allranges>0:

										PR_counts_MR_valid=PR_counts_MR[(steprange>=step_min)*(steprange<step_max)]#should not be necessary!
										PR_counts_PHA_valid=PR_counts_PHA[(steprange>=step_min)*(steprange<step_max)]

										PR_counts_PHA_valid[PR_counts_PHA_valid==0]=-1#make unvalid brfacotors equal -1
										brfactors_prange=PR_counts_MR_valid/PR_counts_PHA_valid
										brfactors_prange[brfactors_prange<0]=-1

										br_factors_t[prange]=brfactors_prange

								else:
										#timestamps_t=array([timestamp]*len(steprange))
										#prange_t=array([prange]*len(steprange))
										#steprange_t=steprange
										br_factors_t[prange]=zeros((len(steprange)))-1.



								if Plot_factors==True:
										print("test_plot")
										plt.figure()
										plt.title("timestamp: %.5f, prange: %i"%(timestamp,prange))
										plt.plot(arange(0,116),PR_counts_MR,color="r",label="BR")
										plt.plot(arange(0,116),PR_counts_PHA,color="b",label="PHA")
										plt.plot(arange(0,116),br_factors_t[prange],color="k",label="BR factors")
										plt.legend()
										plt.show()
										#return True
								                
						allPR_counts_PHA_sum=zeros((len(steprange)))+sum(allPR_counts_PHA)		
						allPR_PHA_day=append(allPR_PHA_day,allPR_counts_PHA_sum)
						timestamps_day=append(timestamps_day,timestamps_t)
						#prange_day=append(prange_day,prange_t)
						steprange_day=append(steprange_day,steprange_t)
						
						#return br_factors_day,br_factors_t
						#return shape(br_factors_day),shape(br_factors_t)
						
						br_factors_day=append(br_factors_day,br_factors_t,axis=1)

				#return timestamps_day,steprange_day,br_factors_day

				t2=clock()
				#return shape(timestamps_day),shape(steprange_day),shape(br_factors_day),shape(allPR_PHA_day),allPR_PHA_day
				outdata=vstack([timestamps_day,steprange_day,br_factors_day,allPR_PHA_day])
				
				if savedata==True:
					savetxt(output_path+outfile, outdata.T, fmt='%.5f', delimiter=' ', newline='\n')
					print("brfactors saved")

				print("(calculation time)/day, day:", t2-t1, t1-t0, t2-t0, day)

		return outdata

	
	
	
	
	
	
	
	"""
	def calc_mpq(self):
		self.mpq=tof_to_mq(self.tof,self.step)
		return True
	"""






	
	def get_vsw(self):
		pmdat=pmdata(self.timeframe)
		self.vsw=zeros(self.times.shape)    
		self.dsw=zeros(self.times.shape)
		self.tsw=zeros(self.times.shape)
		for t in self.time:
			tmpptime=pmdat.time-t
			if amin(abs(tmpptime))<0.00348:
				mask=(abs(tmpptime)==amin(abs(tmpptime)))
				self.vsw[self.times==secst]=pmdat.vel[mask][0]
				self.dsw[self.times==t]=pmdat.dens[mask][0]
				self.tsw[self.times==t]=pmdat.vth[mask][0]
		print("test")       
		#self.w=self.vel/self.vsw
		#self.w[isnan(self.w)]=0.
		return True


	def get_vsw_safe(self,tol = 0.00360): #tolerance = 9 s => (302+9)s/(24*3600) < 0.00360d  

		pmdat = pmdata(self.timeframe)
		pmdat.time = pmdat.time - 0.00175	
		self.vsw_safe=ones((len(self.times)))
		print(self.vsw_safe)
		for i in arange(len(self.times)):			
			for j in arange(len(pmdat.time)):
				if (self.times[i] - pmdat.time[j]) >= 0. and abs(self.times[i] - pmdat.time[j]) < tol:
					self.vsw_safe[i] = pmdat.vel[j]
				else:  
					self.vsw_safe[i] = -999	
		return True




	def get_vsw_quick(self, tol = 0.00360):
		#Method get_vsw_quick is docusecsmentated in ctof/Verena.
		pmdat = pmdata(self.timeframe)
		pmdat.time_shifted = pmdat.time - 0.00175	
		pmdat.time_shifted = pmdat.time_shifted.tolist()
		pmdat.vel = pmdat.vel.tolist()		
		pmdat.vth = pmdat.vth.tolist()
		pmdat.dens = pmdat.dens.tolist()
		pmdat.time = pmdat.time.tolist()

		i = 1 
		while i < len(pmdat.time):
			if (pmdat.time_shifted[i] - pmdat.time_shifted[i-1]) > tol:
				print(i)
				pmdat.time_shifted.insert(i,pmdat.time_shifted[i-1]+0.00360) 
				pmdat.vel.insert(i,-999)
				pmdat.vth.insert(i,-999)		
				pmdat.dens.insert(i,-999)
				pmdat.time.insert(i,pmdat.time[i-1]+0.00360)	
			i = i+1
		
		pmdat.time_shifted = array(pmdat.time_shifted)
		pmdat.vel = array(pmdat.vel)
		pmdat.vth = array(pmdat.vth)
		pmdat.dens = array(pmdat.dens)
		pmdat.time = array(pmdat.time)
		
		self.time_indices = (searchsorted(pmdat.time_shifted,self.times, side = "right")-1)		
		
		self.vsw = pmdat.vel[self.time_indices]	
		self.vth = pmdat.vth[self.time_indices]	
		self.dsw = pmdat.dens[self.time_indices]		
		
		self.pmtime_shifted = pmdat.time_shifted[self.time_indices]
		self.pmtime = pmdat.time_shifted[self.time_indices] + 0.00175 #Creates the corresponding timestamp of pm(inaccuracy = 1 second). 
		print("data synchronized!")


	def load_PM(self):
		pmdat = pmdata(self.timeframe)
		return pmdat

	def conv_time(self):
		self.times_dhms = []   	
		for t in self.times:
			T = days_in_dhms(t)
			self.times_dhms.append(T)	
		self.times_dhms = array(self.times_dhms)

		self.times_dhmsu = []
		for t in self.time:	
			T = days_in_dhms(t)
			self.times_dhmsu.append(T)	
		self.times_dhmsu = array(self.times_dhmsu)

		self.pmtime_dhms = []   	
		for t in self.pmtime:
			T = days_in_dhms(t)
			self.pmtime_dhms.append(T)	
		self.pmtime_dhms = array(self.pmtime_dhms)		

		self.pmtime_shifted_dhms = []   	
		for t in self.pmtime_shifted:
			T = days_in_dhms(t)
			self.pmtime_shifted_dhms.append(T)	
		self.pmtime_shifted_dhms = array(self.pmtime_shifted_dhms) #Ungenauigkeit 1s



	def load_rates(self):
		self.ratestep=zeros((0))
		self.ratetcr=zeros((0))
		self.ratedcr=zeros((0))
		self.ratessr=zeros((0))
		self.ratefsr=zeros((0))
		self.rateher=zeros((0))
		self.ratehpr=zeros((0))
		self.ratesecs=zeros((0))
		for tf in self.timeframe:
			for day in range(int(tf[0])-1,int(tf[1])+1):
				tmpdat=loadtxt(self.path+"c"+str(self.year%100)+"%.3i.cra"%(day),skiprows=13)
				mask=(tmpdat[:,3]<3)
				self.ratesecs=append(self.ratesecs,tmpdat[mask][:,0]+15)
				self.ratestep=append(self.ratestep,tmpdat[mask][:,1])
				self.ratefsr=append(self.ratefsr,tmpdat[mask][:,5])
				self.ratedcr=append(self.ratedcr,tmpdat[mask][:,6])
				self.ratetcr=append(self.ratetcr,tmpdat[mask][:,7])
				self.ratessr=append(self.ratessr,tmpdat[mask][:,8])
				self.ratehpr=append(self.ratehpr,tmpdat[mask][:,9])
				self.rateher=append(self.rateher,tmpdat[mask][:,10])
		self.ratesec=unique(self.ratesecs)
		self.sec=unique(self.secs)

		print("rate data loaded")
		return True


	def calc_baserates(self):
		"""
		Load rate files (.cra) and calculate base rates for tcr and dcr.
		"""
		self.load_rates()
		bint=(append(self.ratesec,self.ratesec[-1]+self.ratesec[-1]-self.ratesec[-2]),arange(120))  # one time step after last cycle is appended for histogram reasons
		ctstcr=array(histogram2d(self.secs[(self.energy>0.)],self.step[(self.energy>0.)],bins=bint))    # number of tcr pha counts are histogrammed (per step and time) 
		ctsdcr=array(histogram2d(self.secs[self.energy>=0.],self.step[self.energy>=0.],bins=bint))  # number of dcr pha counts are histogrammed (per step and time)
		rtcr=array(histogram2d(self.ratesecs,self.ratestep,bins=bint,weights=(self.ratetcr-self.ratehpr-0.9*self.rateher)))*2.5   # tcr rates are converted into counts (*2.5) and histogrammed
		#rdcr=array(histogram2d(self.ratesecs,self.ratestep,bins=bint,weights=(self.ratedcr-self.ratetcr)))*2.5 # tcr rates are converted into counts (*2.5) and histogrammed 
		rdcr=array(histogram2d(self.ratesecs,self.ratestep,bins=bint,weights=(self.ratedcr-self.ratehpr-self.rateher)))*2.5 # tcr rates are converted into counts (*2.5) and histogrammed 
		# zero rates and zero pha counts are set to 1. (Needed for base-rate calculation to avoid NAN and INF)
		ctstcr[0]=clip(ctstcr[0],1.,max(1.,amax(ctstcr[0])))
		ctsdcr[0]=clip(ctsdcr[0],1.,max(1.,amax(ctsdcr[0])))
		rtcr[0]=clip(rtcr[0],1.,max(1.,amax(rtcr[0])))
		rdcr[0]=clip(rdcr[0],1.,max(1.,amax(rdcr[0])))
		brbinx=ctstcr[2]
		brbiny=ctstcr[1]
		brwtcrar=rtcr[0]/ctstcr[0]
		brwdcrar=rdcr[0]/ctsdcr[0]
		time=searchsorted(brbiny,self.secs)
		steps=zeros(self.step.shape,int)
		steps[:]=self.step[:]
		brtcr=brwtcrar[time,steps]
		brdcr=brwdcrar[time,steps]
		self.br=zeros(self.times.shape)
		self.br[self.energy==0]=brdcr[self.energy==0]
		#self.br[self.energy>0]=brtcr[self.energy>0]
		self.br=brdcr
		
	def write_br_files(self):
		self.calc_baserates()
		for tf in self.timeframe:
			for day in range(int(tf[0]),int(tf[1])+1):
				outf=open(self.path+"cphbr396%.3i.dat"%(day),"w")
				outf.write("CTOF PHA data + solar wind properties from PM data + base rates from rate data (.cra) day %.3i year 1996\n"%(day))
				outf.write("*****\n")
				outf.write("DoY\t\tsecs1970\tRange\tEnergy\tToF\tEqstep\tvsw\tdsw\ttsw\tmpq\tw\tbr\n")
				mask=(self.times>=day)*(self.times<day+1)
				tmpdat=zeros((self.times[mask].shape[0],12))
				tmpdat[:,0]=self.times[mask]
				tmpdat[:,1]=self.secs[mask]
				tmpdat[:,2]=self.range[mask]
				tmpdat[:,3]=self.energy[mask]
				tmpdat[:,4]=self.tof[mask]
				tmpdat[:,5]=self.step[mask]
				tmpdat[:,6]=self.vsw[mask]
				tmpdat[:,7]=self.dsw[mask]
				tmpdat[:,8]=self.tsw[mask]
				tmpdat[:,9]=self.mpq[mask]
				tmpdat[:,10]=self.w[mask]
				tmpdat[:,11]=self.br[mask]
				savetxt(outf,tmpdat,fmt=["%f","%i","%.2i","%i","%i","%i","%.2f","%.2f","%.2f","%.2f","%.2f","%.2f"],delimiter="\t")
				#for i in range(self.times[mask].shape[0]):
				#    print i,float(i)/float(self.times[mask].shape[0])
				#    outf.write("%f\t%i\t%.2i\t%i\t%i\t%i\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n"%(self.times[mask][i],self.secs[mask][i],self.range[mask][i],self.energy[mask][i],self.tof[mask][i],self.step[mask][i],self.vsw[mask][i],self.dsw[mask][i],self.tsw[mask][i],self.mpq[mask][i],self.w[mask][i],self.br[mask][i]))
				#    outf.flush()
				outf.close()
		
	def calc_baserates_old(self):
		self.load_rates()
		self.brtcr=[]
		self.brdcr=[]
		for j,sec in enumerate(self.ratesec):
			print(float(j)/float(self.ratesec.shape[0]))
			tmpratesteps=self.ratestep[self.ratesecs==sec]
			tmpratedcr=self.ratedcr[self.ratesecs==sec]
			tmpratetcr=self.ratetcr[self.ratesecs==sec]
			tmpratehpr=self.ratehpr[self.ratesecs==sec]
			for i,step in enumerate(tmpratesteps):
				masksecs=(self.secs==sec)
				maskdcr=(self.step[masksecs]==step)*(self.energy[masksecs]==0.)
				masktcr=(self.step[masksecs]==step)*(self.energy[masksecs]>0.)
				if sum(masktcr)>0.:
					self.brtcr.append(((tmpratetcr[i]-tmpratehpr[i])*2.5)/sum(masktcr))
				else:
					self.brtcr.append(1.)
				if sum(maskdcr)>0.:
				#if tmpratedcr[i]-tmpratetcr[i]>0. and sum(maskdcr)>0.:
					self.brdcr.append(((tmpratedcr[i]-tmpratetcr[i])*2.5)/sum(maskdcr))
				else:
					self.brdcr.append(1.)
		self.brtcr=array(self.brtcr)
		self.brdcr=array(self.brdcr)


	def get_rateepqstep(self):
		self.ratestepepq=[]
		for i in range(self.times.shape[0]):
			tmpval=self.rateepq[(self.ratetimes==self.times[i])*(self.ratestep==self.step[i])]
			if tmpval.shape[0]>0:
				self.ratestepepq.append(tmpval[0])
			else:
				self.ratestepepq.append(0)
			#print i,"/",self.times.shape[0]
		self.ratestepepq=array(self.ratestepepq)
		"""
		for t in self.ratetime:
		    print t
		    tmpstep=self.step[self.times==t]
		    tmpratestepepq=zeros(tmpstep.shape)
		    tmpratestep=self.ratestep[self.ratetimes==t]
		    tmprateepq=self.rateepq[self.ratetimes==t]
		    print tmpstep.shape
		    print tmpratestepepq.shape
		    print tmpratestep.shape
		    print tmprateepq.shape
		    for i in range(tmpratestep.shape[0]):
		        tmpratestepepq[tmpstep==tmpratestep[i]]=tmprateepq[i]
		        self.ratestepepq=append(self.ratestepepq,tmpratestepepq)
		"""

class ctofpui(ctoflv1):
	def __init__(self,timeframe,year=1996,path="/data/etph/soho/celias/ctof/"):
		"""
		This class is ment to deal with CTOF PHA data. PUI data only mpq>10 (and 3.5<mpq<4.5) 
		year -> year of data (mind that CTOF only functioned properly from DoY 80 to 230 in 1996) 
		timeframe -> list of periods [[t1,t2],...,[tn-1,tn]]
		path -> should give the path to the pha data

		times,secs,tof,energy,range,step -> contains the information for each individual PHA word.
		time -> contains a list of the starting times of the instrumental cycles.Each cycle (a complete ESA sweep) is about 300s. 
		"""
		self.year=year
		self.timeframe=timeframe
		self.path=path
		self.times=zeros((0))
		self.secs=zeros((0))
		self.tof=zeros((0))
		self.energy=zeros((0))
		self.range=zeros((0))
		self.step=zeros((0))
		self.mpq=zeros((0))
		self.vel=zeros((0))
		self.vsw=zeros((0))
		self.dsw=zeros((0))
		self.w=zeros((0))
		self.loadlv1()
		self.calc_ionvel()
		self.get_vsw()
		self._add_mmpq()
		
	def loadlv1(self):
		for tf in self.timeframe:
			for day in range(int(tf[0]),int(tf[1]+1.)):
				tmpdat=loadtxt(self.path+"cph"+str(self.year%100)+"%.3i.dat"%(day),skiprows=3)
				self.times=append(self.times,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,0])
				self.secs=append(self.secs,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,1])
				self.range=append(self.range,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,2])
				self.energy=append(self.energy,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,3])
				self.tof=append(self.tof,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,4])
				self.step=append(self.step,tmpdat[(tmpdat[:,0]>tf[0])*(tmpdat[:,0]<tf[1])][:,5])
				self.calc_mpq()
				self.cut_mpq()
		self.time=unique(self.times)
		print("pha data loaded")
		return True

	def cut_mpq(self):
		mask=(self.mpq>10.)+((self.mpq>3.5)*(self.mpq<4.5))
		self.times=self.times[mask]
		self.secs=self.secs[mask]
		self.tof=self.tof[mask]
		self.energy=self.energy[mask]
		self.range=self.range[mask]
		self.step=self.step[mask]
		self.mpq=self.mpq[mask]
        


