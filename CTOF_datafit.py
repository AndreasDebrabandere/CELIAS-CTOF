"""
Class to load the CTOF PHA data (both, base-rate-corrcected and not base-rate-corrected data), visualize it as ET-matrices (residual energy vs time-of-flight for each energy-per-charge step) and to fit these ET-matrices with the CTOF response model. The CTOF response model is imported from CTOF_iondist.py  
Author: Nils Janitzek (2021)
"""

from shutil import which
from CTOF_ion import Ion

#pylab imports
import pylab

#numpy imports
import numpy as np
from numpy import *

#scipy imports
from scipy import optimize, interpolate, integrate, stats, constants
from scipy.special import gamma,gammainc, erf, binom
from scipy.optimize import leastsq
from scipy.optimize import fmin_bfgs as minimizer 
from scipy.optimize import minimize

#matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, cbook
import matplotlib.colors as colors
from matplotlib.colors import Normalize, LogNorm
from matplotlib.path import Path

#pandas imports
import pandas as pd

#import selfmade python modules
from pylib.dbData._dbData import dbData
from CTOF_cal import *
from peakshape_functions import *
from CTOF_ion import Ion, iondict, iondict_minimium
from CTOF_iondist import IonDist

#from libsoho.libctof import getionvel
from Libsoho._ctoflv1 import ctoflv1

#import time modules
from time import perf_counter as clock
import datetime as dt

#import peakshape functions
from peakshape_functions import *

#outdated imports
#import fileinput
#import re
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.offsetbox import AnchoredText
#from collections import OrderedDict

#Figure sizes
figx_full=13.8
figx_half=6.8
figy_full=7
figy_half=5

def phi(u):
    return integrate.quad(lambda x: 1/sqrt(2*pi) *  e**(-(x**2)/2), -inf, u)

def plot_PMdata(utimes,uvsw,uvth,udsw,save_figure=False,figpath="",filename="PMdata_timeseries"):
    
    uvsw_mask=uvsw>0
    uvsw_valid_mask=(uvsw>=320)*(uvsw<=550)
    
    
    ut=utimes[uvsw_mask]
    uv=uvsw[uvsw_mask]
    uvt=uvth[uvsw_mask]
    ud=udsw[uvsw_mask]
    
    ut_valid=utimes[uvsw_valid_mask]
    uv_valid=uvsw[uvsw_valid_mask]
    uvt_valid=uvth[uvsw_valid_mask]
    ud_valid=udsw[uvsw_valid_mask]
    
    fig, axs = plt.subplots(3,1,figsize=(figx_full, figy_half),gridspec_kw={'height_ratios': [1, 1, 1]}, sharex=True)
    fig.subplots_adjust(hspace=0)
    
    fontsize=16
    ticklabelsize=14
    legsize=14
    markersize=5
    elinewidth=1.5, 
    capsize=2
    capthick=2        


    axs[0].plot(ut,uv,linestyle="None",linewidth=2,marker="o",markersize=1.,color="r")
    axs[0].plot(ut_valid,uv_valid,linestyle="None",linewidth=2,marker="o",markersize=1.,color="k")
    axs[0].set_ylabel(r"$\rm{v_p \ [km/s]}$",fontsize=fontsize)
    axs[0].tick_params(axis="x", labelsize=ticklabelsize)
    axs[0].tick_params(axis="y", labelsize=ticklabelsize)
    axs[0].set_ylim(300,580)

    axs[1].plot(ut,uvt,linestyle="None",linewidth=2,marker="o",markersize=1.,color="r")
    axs[1].plot(ut_valid,uvt_valid,linestyle="None",linewidth=2,marker="o",markersize=1.,color="k")
    axs[1].set_ylabel(r"$\rm{v_{th,p} \ [km/s]}$",fontsize=fontsize)
    axs[1].tick_params(axis="x", labelsize=ticklabelsize)
    axs[1].tick_params(axis="y", labelsize=ticklabelsize)
    axs[1].set_ylim(10,90)

    axs[2].plot(ut,ud,linestyle="None",linewidth=2,marker="o",markersize=1.,color="r")
    axs[2].plot(ut_valid,ud_valid,linestyle="None",linewidth=2,marker="o",markersize=1.,color="k")
    axs[2].set_ylabel(r"$\rm{n_{p} \ [cm^{-3}] } $",fontsize=fontsize)
    axs[2].tick_params(axis="x", labelsize=ticklabelsize)
    axs[2].tick_params(axis="y", labelsize=ticklabelsize)
    axs[2].set_ylim(0,45)

    axs[2].set_xlim(174,220)
    axs[2].set_xlabel(r"$\rm{ time \ [DOY 1996] }$",fontsize=fontsize)
    

    if save_figure==True:
        plt.savefig(figpath+filename+".png",bbox_inches='tight')


def vsw_RCfilter(step):
	if step<=60:
		vsw_range=[0,1000]
	elif (step>60) and (step<=80): 	
		vsw_range=[0,round(getionvel(mpc=16/7.,step=step))]
	elif (step>80):
		vsw_range=[0,round(getionvel(mpc=16/7.,step=step)+2.5*(step-80))]
	return vsw_range

def distheight_multiplication(heights,hgs,sum_dists=True):
	distmult=(abs(heights)*hgs.T).T
	if sum_dists==True:
		res=sum(distmult,axis=0)
	else:
		res=distmult	
	return res

def distheight_multiplication_reshape(heights,hgs_reshaped,sum_dists=True):
	hgs=hgs_reshaped.reshape(len(heights),len(hgs_reshaped)/len(heights))
	distmult=(abs(heights)*hgs.T).T
	if sum_dists==True:
		res=sum(distmult,axis=0)
	else:
		res=distmult	
	return res


def multres(fitres,model,data):
	"""
	works on one (timestamp,step)	
	shape fitres: (Nions)
	shape model: (Nions,Ntof*Nenergy)
	shape data: (Ntof*Nenergy)
	"""
	m=model#array of response models, sum for each model is normalized to 1	
	mr=m*fitres[:,newaxis]#response models scaled with fit 
	R=sum(mr,axis=0)#total counts at each point in ET calculated by the fit 
	R[R==0]=1.
	mrn=mr*1./R#normalized scaled response models, is equal of the relative count rate contribution from each ion at each ET position 
	CI=sum(mrn*data,axis=1)#number of counts that are assigned to each ion species (-> goes directly into VDF) 
	return CI

L_Arnaud_fullstable_noHe=['C4+','C5+','C6+',"N4+","N5+","N6+","N7+","O5+","O6+","O7+","O8+","Ne5+","Ne6+","Ne7+","Ne8+","Ne9+",'Mg4+','Mg5+','Mg6+','Mg7+','Mg8+','Mg9+','Mg10+',"Si5+","Si6+","Si7+","Si8+","Si9+","Si10+","Si11+","Si12+","S6+","S7+","S8+","S9+","S10+","S11+","S12+","S13+","Ca10+","Ca11+","Fe5+","Fe6+","Fe7+","Fe8+","Fe9+","Fe10+","Fe11+","Fe12+","Fe13+","Fe14+","Fe15+","Fe16+","Ni8+","Ni9+","Ni10+"]


def gauss1d(p,x):

		return p[0]*exp(-(x-p[1])**2/(2*p[2]**2))


def ugauss1d(p,x):#Gaussian of unity height

		return exp(-(x-p[0])**2/(2*p[1]**2))



class MidPointNorm(Normalize):    

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):

        Normalize.__init__(self,vmin, vmax, clip)

        self.midpoint = midpoint



    def __call__(self, value, clip=None):

        if clip is None:

            clip = self.clip



        result, is_scalar = self.process_value(value)



        self.autoscale_None(result)

        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint



        if not (vmin < midpoint < vmax):

            raise ValueError("midpoint must be between maxvalue and minvalue.")       

        elif vmin == vmax:

            result.fill(0) # Or should it be all masked? Or 0.5?

        elif vmin > vmax:

            raise ValueError("maxvalue must be bigger than minvalue")

        else:

            vmin = float(vmin)

            vmax = float(vmax)

            if clip:

                mask = ma.getmask(result)

                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),

                                  mask=mask)



            # ma division is very slow; we can take a shortcut

            resdat = result.data



            #First scale to -1 to 1 range, than to from 0 to 1.

            resdat -= midpoint            

            resdat[resdat>0] /= abs(vmax - midpoint)            

            resdat[resdat<0] /= abs(vmin - midpoint)



            resdat /= 2.

            resdat += 0.5

            result = ma.array(resdat, mask=result.mask, copy=False)                



        if is_scalar:

            result = result[0]            

        return result



    def inverse(self, value):

        if not self.scaled():

            raise ValueError("Not invertible until scaled")

        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint



        if cbook.iterable(value):

            val = ma.asarray(value)

            val = 2 * (val-0.5)  

            val[val>0]  *= abs(vmax - midpoint)

            val[val<0] *= abs(vmin - midpoint)

            val += midpoint

            return val

        else:

            val = 2 * (val - 0.5)

            if val < 0: 

                return  val*abs(vmin-midpoint) + midpoint

            else:

                return  val*abs(vmax-midpoint) + midpoint




class ctof_paramfit(dbData):						
    """
    General class to fit the CTOF PHA data (both, base-rate-corrcected and not base-rate-corrected PHA data). 	
    """
    def load_data(self,timeframe,minute_frame,path="/data/mission_data/SOHO/CELIAS/CTOF/lv1/",load_processed_PHAdata=False,prepare_mrdata=True):
			
        #old path:
        #path="/home/auto/hatschi/janitzek/mission_data/SOHO/CELIAS/CTOF/lv1/"
        
        self.data = {}
        #dat = ctoflv1_etph(timeframe,minute_frame)
        print("timeframe:",timeframe,minute_frame)
        days=arange(timeframe[0][0],timeframe[0][-1],1)
        print("days",days)
        
        if load_processed_PHAdata==True:
            #PHAdata=self.load_ETdata()#old: loads all data DOY174-220 from one file
            PHAdata=self.load_ETdata_daywise(days=days)

            #CTOF PHA data
            self.data["seconds"] = PHAdata[0]
            self.data["time"] = PHAdata[1]
            self.data["steps"] = PHAdata[2]
            self.data["tof"] = PHAdata[3]
            self.data["energy"] = PHAdata[4]
            self.data["range"] = PHAdata[5]		
            self.data["mass-per-charge"] = PHAdata[6]
            self.data["mass"] = PHAdata[7]
            self.data["stepmax"]= PHAdata[8]
            self.data["baserate_raw"] = PHAdata[9]
            self.data["baserate"] = PHAdata[10]
            

            #PM data, synchronized with CTOF PHA data		      
            self.data["dsw"] = PHAdata[11]
            self.data["vsw"] = PHAdata[12]
            self.data["vth"] = PHAdata[13]


        else:
            
            t0=clock()
            dat = ctoflv1(timeframe,minute_frame)
            if prepare_mrdata==True:
                dat.prepare_CTOFdata(days=arange(timeframe[0][0],timeframe[0][-1],1))

            #load CTOF PHA-data (already synchronized with MR-data and PM-data) 
            self.data["seconds"] = dat.secs
            self.data["time"] = dat.times
            self.data["steps"] = dat.step
            self.data["tof"] = dat.tof
            self.data["energy"] = dat.energy
            self.data["range"] = dat.range		
            self.data["mass-per-charge"] = dat.mpq
            self.data["mass"] = dat.m
            self.data["baserate_raw"] = None
            self.data["baserate"] = None
            self.data["ustepmax"],self.data["stepmax"]=self.get_ESAstop()
            #self.data["times_phavalid"]=self.cut_PHAmin()

            #load PM data (already synchronized with MR-data and PHA-data)        
            self.data["dsw"] = dat.dsw
            self.data["vsw"] = dat.vsw
            self.data["vth"] = dat.vth
            self.data["pmtime"] = dat.pmtime 
            self.missmatches = zeros((0))#probably not needed
            self.EThist="none"#probably not needed
            self.plot="none"#current_plot""", probably not needed
            t1=clock()
            print("CTOF PHA data loaded, loading time = %.2f seconds"%(t1-t0))
    
            if prepare_mrdata==True:
                #load MR box and shift definition
                self.data["mrbox_number"]=dat.mrbox_number
                self.data["mrbox_shift"]=dat.mrbox_shift

                #load CTOF MR data (already synchronized with PM-data and PHA-data)
                self.data["seconds_mr"]=dat.secs_mr
                self.data["time_mr"]=dat.times_mr
                self.data["mr_number"]=dat.mr_number#probably not needed
                self.data["vFe_mr"]=dat.vFe_mr#onboard estimated iron speed
                self.data["MR"]=dat.MR#raw matrix rate count data

                self.data["MRC"]=dat.MRC#
                self.data["MRA"]=dat.MRA#
                self.data["centersteps"]=dat.centersteps
                self.data["epq_rates"]=dat.epq_rates
                self.data["cphas"]=dat.cphas
                self.data["cepqs"]=dat.cepqs
                self.data["brfs"]=dat.brfs
                self.data["baserate_raw"]=dat.baserate

                baserate=dat.baserate*1.
                baserate[baserate<1]=1.
                self.data["baserate"]=baserate

    def save_ETdata(self,filepath="/home/hatschi/janitzek/CELIAS_Data_Processed/CTOF_PM_Sync/",filename="PHA_Final_DOY174-220"):
        data=array([self.data["seconds"],self.data["time"],self.data["steps"], self.data["tof"],self.data["energy"],self.data["range"],self.data["mass-per-charge"], self.data["mass"],self.data["stepmax"],self.data["baserate_raw"],self.data["baserate"],self.data["dsw"],self.data["vsw"],self.data["vth"]])
        save(file=filepath+filename,arr=data)		
        print("data saved")
        return True
            
    def load_ETdata(self,filepath="/home/hatschi/janitzek/CELIAS_Data_Processed/CTOF_PM_Sync/",filename="PHA_Final_DOY174-220.npy"):
        t0=clock()
        ETdata=load(file=filepath+filename)
        t1=clock()
        print("processed PHA data loaded, loading time %i seconds"%(t1-t0))
        return ETdata	
		
    def save_ETdata_daywise(self,days,filepath="/home/hatschi/janitzek/CELIAS_Data_Processed/CTOF_PM_Sync/",filename="PHA_Final_day"):
    	for day in days:
            dmask=(self.data["time"]>=day)*(self.data["time"]<(day+1))
            data=array([self.data["seconds"][dmask],self.data["time"][dmask],self.data["steps"][dmask], self.data["tof"][dmask],self.data["energy"][dmask],self.data["range"][dmask],self.data["mass-per-charge"][dmask], self.data["mass"][dmask],self.data["stepmax"][dmask],self.data["baserate_raw"][dmask],self.data["baserate"][dmask],self.data["dsw"][dmask],self.data["vsw"][dmask],self.data["vth"][dmask]])
            save(file=filepath+filename+"%i"%(day),arr=data)		
            print("data saved for day %i"%(day))


    def load_ETdata_daywise(self,days,filepath='C:/Users/andre/CTOF_CELIAS/CELIAS-CTOF/Data'):
        #filepath="/lhome/njanitzek/Projects/SOHO/data/mission_data/SOHO/CELIAS_Data_Processed/CTOF_PM_Sync/" # For Nils
        t0=clock()
        ETdata=empty([14,0])
        for day in days:
            filename="PHA_Final_day%i.npy"%(day)
            print("loading day:", day)
            ETdata_day=load(file='Data/'+filename)
            ETdata=append(ETdata,ETdata_day,axis=1)
        t1=clock()
        print("processed PHA data loaded, loading time %i seconds"%(t1-t0))
        return ETdata	
        


    def prepare_MRdata(self):
        
        dat = ctoflv1([[150,151]])
        usteps=arange(0,116,1)
        boxes_O6=zeros((len(usteps),4))
        boxes_Si7=zeros((len(usteps),4))
        boxes_Fe9=zeros((len(usteps),4))
        boxes_Fe13=zeros((len(usteps),4))
        dat.prepare_CTOFdata()
        i=0
        for  step in usteps: 
            
            #O6
            tof_bounds_O6,energy_bounds_O6=dat.find_tofEbox(step=step,mrbox_number=235)
            tof_low_O6,tof_high_O6=tof_bounds_O6[0],tof_bounds_O6[1]
            energy_low_O6,energy_high_O6=energy_bounds_O6[0],energy_bounds_O6[1]
            boxes_O6[i]=array([tof_low_O6,tof_high_O6,energy_low_O6,energy_high_O6])
            
            #Si7
            tof_bounds_Si7,energy_bounds_Si7=dat.find_tofEbox(step=step,mrbox_number=201)
            tof_low_Si7,tof_high_Si7=tof_bounds_Si7[0],tof_bounds_Si7[1]
            energy_low_Si7,energy_high_Si7=energy_bounds_Si7[0],energy_bounds_Si7[1]
            boxes_Si7[i]=array([tof_low_Si7,tof_high_Si7,energy_low_Si7,energy_high_Si7])
            
            #Fe9
            tof_bounds_Fe9,energy_bounds_Fe9=dat.find_tofEbox(step=step,mrbox_number=41)
            tof_low_Fe9,tof_high_Fe9=tof_bounds_Fe9[0],tof_bounds_Fe9[1]
            energy_high_Fe9=energy_bounds_Fe9[1]
            energy_low_Fe9=dat.find_tofEbox(step=step,mrbox_number=93)[1][0]
            boxes_Fe9[i]=array([tof_low_Fe9,tof_high_Fe9,energy_low_Fe9,energy_high_Fe9])
            
            #Si7
            tof_bounds_Fe13,energy_bounds_Fe13=dat.find_tofEbox(step=step,mrbox_number=45)
            tof_low_Fe13,tof_high_Fe13=tof_bounds_Fe13[0],tof_bounds_Fe13[1]
            energy_low_Fe13,energy_high_Fe13=energy_bounds_Fe13[0],energy_bounds_Fe13[1]
            boxes_Fe13[i]=array([tof_low_Fe13,tof_high_Fe13,energy_low_Fe13,energy_high_Fe13])
            
            i+=1
        
        self.tofE_boxes_O6=boxes_O6   
        self.tofE_boxes_Si7=boxes_Si7    
        self.tofE_boxes_Fe9=boxes_Fe9    
        self.tofE_boxes_Fe13=boxes_Fe13    
    
    
    def brfcor(self,days,br_decimals_round=5,Plot=False,CBlog=True,brfactors_pathfile="/home/asterix/janitzek/ctof/libsoho/BRfactors_improved/test"):
        #make faster, priority range 6 appears occasionally in PHA data (e.g. on DOY 201,202 1996 therefore method crashes!)
        #brfactors_pathfile="/home/asterix/janitzek/ctof/libsoho/BRfactors/BRFincl_PHA/test"
        #brfactors_pathfile="/home/asterix/janitzek/ctof/libsoho/BRfactors_improved/test"

        Brdata=[]
        for day in days: 
            infile_br=brfactors_pathfile+"_day%i"%(day)             
            brdata=loadtxt(infile_br,delimiter=None,skiprows=0,unpack=True)
            Brdata.append(brdata)
        brdata=concatenate(Brdata,axis=1)
        #return brdata 

        wgts=zeros(self.data["time"].shape[0],dtype=float)
        times=around(self.data["time"],decimals=br_decimals_round)
        pr=self.data["range"]
        epq=self.data["steps"]
        tm = in1d(times,unique(brdata[0]))
        times=times[tm]
        pr = pr[tm]
        epq = epq[tm]
        tm2 = in1d(brdata[0],times)
        brdata=brdata[...,tm2]

        #return brdata
        #prates = brdata[2:8].reshape(6,unique(brdata[0])/116,116)
        prates = brdata[2:8].reshape(6,unique(brdata[0]).size,116)
        tmpwgts=[]
        for i,t in enumerate(unique(brdata[0])):
            m = times == t
            print(i,t)
            prt = pr[m]
            epqt = epq[m]
            timest = times[m]
            for j,p in enumerate(timest):
                tmpwgts.append(prates[int(prt[j]),i,int(epqt[j])])
        #tmpwgts = prates[pr.astype(int)][times.astype(int)][epq.astype(int)]
        wgts[tm]=array(tmpwgts)
        wgts[wgts<0]=0.
        self.data["nbr"]=wgts


    def brfcor_daywise(self,days,br_decimals_round=5,Plot=False,CBlog=True,brfactors_pathfile="/home/asterix/janitzek/ctof/libsoho/BRfactors_improved/test"):
        """
        make faster, priority range 6 appears occasionally in PHA data (e.g. on DOY 201,202 1996 therefore method crashes!)
        """	
        #brfactors_pathfile="/home/asterix/janitzek/ctof/libsoho/BRfactors/BRFincl_PHA/test"
        #brfactors_pathfile="/home/asterix/janitzek/ctof/libsoho/BRfactors_improved/test"

        times_alldays=[]
        steps_alldays=[]
        tof_alldays=[]
        energy_alldays=[]
        wgts_alldays=[]
        for day in arange(days[0],days[-1],1): 
            infile_br=brfactors_pathfile+"_day%i"%(day)             
            brdata=loadtxt(infile_br,delimiter=None,skiprows=0,unpack=True)
        
            dpha=ctof_paramfit([[day,day+1]],[0,1440])
            wgts=zeros(dpha.data["time"].shape[0],dtype=float)
            times=around(dpha.data["time"],decimals=br_decimals_round)
            pr=dpha.data["range"]
            pr[pr==6]=0#dirty work-around, but only a handful of counts are in PR6 and PR0 is not interesting for our study
            epq=dpha.data["steps"]
            tm = in1d(times,unique(brdata[0]))
            times=times[tm]
            pr = pr[tm]
            epq = epq[tm]
            tm2 = in1d(brdata[0],times)
            brdata=brdata[...,tm2]

            #return brdata
            #prates = brdata[2:8].reshape(6,unique(brdata[0])/116,116)
            prates = brdata[2:8].reshape(6,unique(brdata[0]).size,116)
            tmpwgts=[]
            for i,t in enumerate(unique(brdata[0])):
                m = times == t
                print(i,t)
                prt = pr[m]
                epqt = epq[m]
                timest = times[m]
                for j,p in enumerate(timest):
                    tmpwgts.append(prates[int(prt[j]),i,int(epqt[j])])
            #tmpwgts = prates[pr.astype(int)][times.astype(int)][epq.astype(int)]
            wgts[tm]=array(tmpwgts)
            wgts[wgts<0]=0.
            wgts_alldays.append(wgts)
            times_alldays.append(dpha.data["time"])
            steps_alldays.append(dpha.data["steps"])
            tof_alldays.append(dpha.data["tof"])
            energy_alldays.append(dpha.data["energy"])
            
        wgts_alldays=concatenate(wgts_alldays)	
        times_alldays=concatenate(times_alldays)
        steps_alldays=concatenate(steps_alldays)
        tof_alldays=concatenate(tof_alldays)
        energy_alldays=concatenate(energy_alldays)
        self.data["nbr"]=wgts_alldays
        return times_alldays,steps_alldays,tof_alldays,energy_alldays,wgts_alldays
			
			
    def Plot_BRcorrection_EThist(self,step,steps=None,tof=None,energy=None,wgts=None,tofrange=arange(200,600,2),Erange=(100,100,2),CBlog=True): 

        if (tof is None) or (energy is None) or (wgts is None):
            steps=self.data["steps"]
            tof=self.data["tof"]
            energy=self.data["energy"]
            wgts=self.data["nbr"]
            
        m=(steps==step)
        h=histogram2d(tof[m],energy[m],[tofrange,Erange])
        hcor=histogram2d(tof[m],energy[m],[tofrange,Erange],weights=wgts[m])
        
        #plot pure PHA ET-matrix
        fig, ax = plt.subplots(1,1)
        #my_cmap = cm.get_cmap("Spectral_r",1024*16)
        my_cmap = cm.get_cmap("jet",1024*16)
        my_cmap.set_under('w')
        if CBlog==True:
            #return h[0],h[1],h[2],
            Cont1=ax.pcolormesh(h[1],h[2],h[0].T,cmap=my_cmap, norm=colors.LogNorm(vmin=0.1,vmax=max(ravel(h[0]))))
        else:
            Cont1=ax.pcolormesh(hcor[1],hcor[2],hcor[0].T,cmap=my_cmap, vmin=0.1,vmax=max(ravel(h[0])))
        cb1 = fig.colorbar(Cont1)
        cb1.set_label("PHA counts per bin")	
        ax.set_title("E/q-step: %i, pure PHA"%(step))

        #plot base-rate corrected PHA ET-matrix
        fig, ax = plt.subplots(1,1)
        #my_cmap = cm.get_cmap("Spectral_r",1024*16)
        my_cmap = cm.get_cmap("jet",1024*16)
        my_cmap.set_under('w')
        if CBlog==True:
            #return h[0],h[1],h[2],
            Cont1=ax.pcolormesh(hcor[1],hcor[2],hcor[0].T,cmap=my_cmap, norm=colors.LogNorm(vmin=0.1,vmax=max(ravel(hcor[0]))))
        else:
            Cont1=ax.pcolormesh(hcor[1],hcor[2],hcor[0].T,cmap=my_cmap, vmin=0.1,vmax=max(ravel(hcor[0])))
        cb1 = fig.colorbar(Cont1)
        cb1.set_label("base-rate corrected PHA counts per bin")	
        ax.set_title("E/q-step: %i, corrected PHA"%(step))

    
    
    
    
    
    def multiply_brfactors_ET(self, days, steps, tofs=arange(150,600,2),energies=arange(1,100,2), pranges=[1,2,3,4,5], vsw_range=[200,1000],PHA_min=600, br_decimals_round=5, Plot_ETmatrix_uncor=True, Plot_ETmatrix_cor=True, Plot_ETmatrix_corratio=True,CBlog=False,ETuncor_range=None,ETcor_range=None, Plot_energycut=True,tofch_Ecut=275,timestamp_index=0,br2_cor=1.,br3_cor=1.,br4_cor=1.,br5_cor=1.,brfactors_pathfile="/home/asterix/janitzek/ctof/libsoho/BRfactors_improved/test"):
    
        """
        #Method multiplies ET PHA matrix (triple coincidences) with timestamp- and step-dependent reverse engineered priority range base rates. The heaviest ions (iron) are in priority range (PR) 5, the lightest (helium) are in PR 1, double coincidences are also included as prange 0 (which reaches up to about ~7 channels in residual energy), but the base rate factors for DC data are orders of magnitudes higher!)
        comment completely!
        """
        #brfactors_pathfile="/home/asterix/janitzek/ctof/libsoho/BRfactors/BRFincl_PHA/test"#old and wrong
        EThists_brcor=zeros((len(days),len(tofs)-1,len(energies)-1))#(-1 channel for histogram reasons)
        EThists_uncor=zeros((len(days),len(tofs)-1,len(energies)-1))#(-1 channel for histogram reasons)
        allPHAs_sync=[]
        i=0
        for day in days:

            t0=clock()
            
            #load brfactors
            infile_br=brfactors_pathfile+"_day%i"%(day)             
            brdata=loadtxt(infile_br,delimiter=None,skiprows=0,unpack=True)

            times_br=around(brdata[0],decimals=br_decimals_round)
            steps_br=brdata[1]
            #factors_PR0=brdata[2]
            #factors_PR1=brdata[3]
            #factors_PR2=brdata[4]
            #factors_PR3=brdata[5]
            #factors_PR4=brdata[6]
            #factors_PR5=brdata[7]
            allPHA=brdata[8]
            
            #set PHA count mask 
            PHA_mask=(allPHA>=PHA_min)
            utimes_br=unique(times_br[PHA_mask])
            ss=searchsorted(times_br,utimes_br)
            uallPHA=allPHA[ss]
            
            #get PHA data and set time masks
            times=around(self.data["time"],decimals=br_decimals_round)
            daymask=(times>=day)*(times<(day+1))
            
            timedata=times[daymask]
            time_mask=in1d(timedata,utimes_br)
            utimedata=unique(timedata)	
            utimedata_mask=in1d(utimedata,utimes_br)
            
            allPHA_sync=zeros((len(utimedata)))
            allPHA_sync[utimedata_mask]=uallPHA
            allPHAs_sync.append(allPHA_sync)
            
            tofdata=self.data["tof"][daymask]
            energydata=self.data["energy"][daymask]
            stepdata=self.data["steps"][daymask]
            rangedata=self.data["range"][daymask]
            vswdata=(self.data["vsw"][daymask])
            vsw_mask=(vswdata>=vsw_range[0])*(vswdata<vsw_range[-1])

            t1=clock()
            #multiply PHA data and base rate factors
            for step in steps:
                step_mask=(stepdata==step)

                #multiply EThist with PR base rates range-wise
                EThists_brcor_step=[]
                EThists_uncor_step=[]
                for prange in pranges:
                    t11=clock()
                    prange_mask=(rangedata==prange)
                    timedata_stepprange=timedata[step_mask*prange_mask*vsw_mask*time_mask]	
                    tofdata_stepprange=tofdata[step_mask*prange_mask*vsw_mask*time_mask]
                    energydata_stepprange=energydata[step_mask*prange_mask*vsw_mask*time_mask]
                    
                    t12=clock()
                    #create uncorrected EThist for comparison
                    ETdata_stepprange=array([timedata_stepprange,tofdata_stepprange,energydata_stepprange])
                    bins = array([utimedata, tofs, energies])
                    EThist_steprange, edges = histogramdd(ETdata_stepprange.T, bins)
                    EThist_uncor_steprange=(EThist_steprange.transpose(2,1,0)*utimedata_mask[:-1]).transpose(2,1,0)#last timestamp is left out for histogram reasons
                    EThists_uncor_step.append(EThist_uncor_steprange)
                    t13=clock()
                    
                    #get base rate factor for each range and synchronize it with the PHA data 	
                    brf_stepprange=brdata[2+prange][steps_br==step]
                    brf_stepprange[brf_stepprange<0]=0
                    brf_stepprange_sync=zeros((len(utimedata)))
                    brf_stepprange_sync[utimedata_mask]=brf_stepprange
                    
                    #multiply EThist with synchronized base rate factors
                    if prange==2:
                        EThist_brcor_steprange=(EThist_steprange.transpose(2,1,0)*br2_cor*brf_stepprange_sync[:-1]).transpose(2,1,0)#test
                    
                    elif prange==3:
                        EThist_brcor_steprange=(EThist_steprange.transpose(2,1,0)*br3_cor*brf_stepprange_sync[:-1]).transpose(2,1,0)#test
                    
                    elif prange==4:
                        EThist_brcor_steprange=(EThist_steprange.transpose(2,1,0)*br4_cor*brf_stepprange_sync[:-1]).transpose(2,1,0)#test
                    elif prange==5:
                        EThist_brcor_steprange=(EThist_steprange.transpose(2,1,0)*br5_cor*brf_stepprange_sync[:-1]).transpose(2,1,0)#test	
                        
                    else:			
                        EThist_brcor_steprange=(EThist_steprange.transpose(2,1,0)*brf_stepprange_sync[:-1]).transpose(2,1,0)#last timestamp is left out for histogram reasons
                    EThists_brcor_step.append(EThist_brcor_steprange)
                    t14=clock()
                    
                    
                    print("histogramming and multiplication run times:",t12-t11,t13-t12,t14-t13,t14-t11)

                EThist_uncor_step=sum(array(EThists_uncor_step),axis=0)
                EThist_brcor_step=sum(array(EThists_brcor_step),axis=0)	
            
            EThist_uncor_day=sum(EThist_uncor_step,axis=0)
            EThist_brcor_day=sum(EThist_brcor_step,axis=0)
            
            EThists_uncor[i]=EThist_uncor_day
            EThists_brcor[i]=EThist_brcor_day
            i=i+1
            t2=clock()

        EThist_uncor=sum(EThists_uncor,axis=0)	
        EThist_brcor=sum(EThists_brcor,axis=0)	
        
        #concatenate total PHA count number data
        allPHAs_sync=concatenate(allPHAs_sync)
        
        #return EThist_uncor,EThist_brcor,utimedata,allPHAs_sync

        
        
        #plot corrected ETmatrix if desired:
        if Plot_ETmatrix_cor==True:
            fig, ax = plt.subplots(1,1)
            #my_cmap = cm.get_cmap("Spectral_r",1024*16)
            my_cmap = cm.get_cmap("jet",1024*16)
            my_cmap.set_under('w')
            if CBlog==True:
                Cont1=ax.pcolor(tofs,energies,EThist_brcor.T,cmap=my_cmap, norm=colors.LogNorm(vmin=0.1,vmax=max(ravel(EThist_brcor))))
            else:
                if ETcor_range!=None:
                    print("range test")
                    Cont1=ax.pcolor(tofs,energies,EThist_brcor.T,cmap=my_cmap, vmin=ETcor_range[0],vmax=ETcor_range[-1])
                    #return brf_range
                else:
                    Cont1=ax.pcolor(tofs,energies,EThist_brcor.T,cmap=my_cmap, vmin=0.1,vmax=max(ravel(EThist_brcor)))
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("base-rate corrected PHA counts per bin")	
            ax.set_title("corrected PHA")

        #plot uncorrected ETmatrix as reference if desired:
        if Plot_ETmatrix_uncor==True:
            fig, ax = plt.subplots(1,1)
            #my_cmap = cm.get_cmap("Spectral_r",1024*16)
            my_cmap = cm.get_cmap("jet",1024*16)
            my_cmap.set_under('w')
            if CBlog==True:
                Cont1=ax.pcolor(tofs,energies,EThist_uncor.T,cmap=my_cmap, norm=colors.LogNorm(vmin=0.1,vmax=max(ravel(EThist_uncor))))
            else:
                if ETuncor_range!=None:
                    print("range test")
                    Cont1=ax.pcolor(tofs,energies,EThist_uncor.T,cmap=my_cmap, vmin=ETuncor_range[0],vmax=ETuncor_range[-1])
                    #return brf_range
                else:
                    Cont1=ax.pcolor(tofs,energies,EThist_uncor.T,cmap=my_cmap, vmin=0.1,vmax=max(ravel(EThist_uncor)))
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("uncorrected PHA counts per bin")	
            ax.set_title("pure PHA")				
        
        EThist_uncor_denom=EThist_uncor*1.
        EThist_uncor_denom[EThist_uncor_denom==0]=1.
        EThist_ratio=EThist_brcor/EThist_uncor_denom
        #for prange in pranges:
            
        

        #plot ("daily averaged") base rates if desired:
        if Plot_ETmatrix_corratio==True:
            fig, ax = plt.subplots(1,1)
            #my_cmap = cm.get_cmap("Spectral_r",1024*16)
            my_cmap = cm.get_cmap("jet",1024*16)
            my_cmap.set_under('w')
            if CBlog==True:
                Cont1=ax.pcolor(tofs,energies,EThist_ratio.T,cmap=my_cmap, norm=colors.LogNorm(vmin=0.1,vmax=max(ravel(EThist_ratio))))
            else:
                if brf_stepprange!=None:
                    print("range test")
                    Cont1=ax.pcolor(tofs,energies,EThist_ratio.T,cmap=my_cmap, vmin=brf_stepprange[0],vmax=brf_stepprange[-1])
                    #return brf_range
                else:
                    Cont1=ax.pcolor(tofs,energies,EThist_ratio.T,cmap=my_cmap, vmin=0.1,vmax=max(ravel(EThist_ratio)))
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("br correction factors")	
            ax.set_title("br correction factors")

        #plot cut in energy range for uncorrected, corrected count rates and base rates 	
        if Plot_energycut==True:
            tof_exist=find_nearest(tofs,tofch_Ecut)
            tofch_ind=where(tofs==tof_exist)[0][0]
            
            plt.figure()
            counts_Ecut=EThist_uncor[tofch_ind]
            countscor_Ecut=EThist_brcor[tofch_ind]
            brf_Ecut=EThist_ratio[tofch_ind]
            #return energies,counts_Ecut
            plt.plot(energies[:-1],counts_Ecut,color="b",label="uncorrected counts")
            plt.plot(energies[:-1],countscor_Ecut,color="r",label="corrected counts")
            plt.plot(energies[:-1],brf_Ecut,color="c",label="calc. base rate factors")
            plt.legend()
            plt.xlabel("energy [ch]")
            plt.ylabel("counts per bin")
            plt.title("energy cut tofch: %i"%(tof_exist))
            plt.show()

        #return EThist_uncor,EThist_brcor,ucounts_allPHA,brdata_all
        print(t1-t0,t2-t1,t2-t0)
        #return ravel(EThist_uncor),ravel(EThist_brcor),ravel(EThist_ratio),utimes_day,brf_prange_sync,ucounts_allPHA,ucounts_allPHA_sync
        #return utimes_mask
        return ravel(EThist_brcor),ravel(EThist_uncor)


    
    
    
    
    #routine to plot 2d-count histogram (counts vs tof and energy) 
    def autoplot(self,mask,tofrange,Erange,bintof,binE,step,vsw=[1,600],remove_mask=True,show_mask=False,plottof=False,plotE=False,closeplot=False):
        
        self.set_mask("Master", "energy",1,100)
        self.set_mask("Master", "vsw", 1, 600)
        self.add_mask(mask)
        self.set_mask(mask,"tof",tofrange[0],tofrange[1])
        self.set_mask(mask,"energy",Erange[0],Erange[1])
        self.set_mask(mask,"steps",step,step)
        self.set_mask(mask,"vsw",vsw[0],vsw[1])
        
        if show_mask==True:
            self.show_mask()		
        
        #global plot			
        try:
            plot = self.hist2d("tof","energy",binx = arange(tofrange[0],tofrange[1],bintof), biny = arange(Erange[0],Erange[1],binE),cb="Counts per bin",smask=[mask],xlabel="ToF[ch]",ylabel="Energy[ch]")
        #plot = self.hist2d("tof","energy",binx = 512, biny = 512,smask =[mask],cb="Normalized Counts",xlabel="ToF/ch",ylabel="Energy/ch")		
        except ValueError:
            plt.close()

        if remove_mask==True:	
            self.remove_mask(mask)
        return None
            
        if plottof==True:		
            plot_tof = self.hist1d("tof", binx = arange(tofrange[0],tofrange[1],bintof), smask =[mask],xlabel ="tof/ch", ylabel = "Counts/Bin")		
        if plotE==True:
            plot_E = self.hist1d("energy", binx = arange(Erange[0],Erange[1],binE), smask =[mask],xlabel ="Energy/ch", ylabel = "Counts/Bin")		

        if remove_mask==True:	
            self.remove_mask(mask)
            
        if closeplot==True:
            plt.close()
        
        if plottof==True:
            output=plot,plot_tof
        else:
            output=plot    
        
        #return output
        #print "return test"
        if plottof==True and plotE==True:
            return plot_tof,plot_E 
    
    
    def model_vs_data(self,ionlist,peakshapes,step,tofrange,Erange,tofbin,Ebin,vsw=[1,600],plottof=True,plotE=True,intensities=None,plot_single_ions=None):
    	
        #get model
        I=IonDist()
        I.add_ionlist(ionlist,peakshapes,intensities)
        x=arange(tofrange[0],tofrange[-1],tofbin)
        y=arange(Erange[0],Erange[-1],Ebin)
        ygrid,xgrid=meshgrid(y,x)
        xg,yg=ravel(xgrid).astype(float),ravel(ygrid).astype(float)
        hg_all=I.create_peakfunc_allions(step=step,xg=xg,yg=yg)	
        H_all=hg_all.reshape(len(x),len(y))
			
			
    #plot data    	
        if plottof==True:
            mask="step: %s"%(step)
            plot_tof,plot_E = self.autoplot(mask, tofrange,Erange,tofbin,Ebin,step, vsw=[1,600],remove_mask=True,show_mask=False,plottof=plottof,plotE=plotE,closeplot=False)	
            #return plot_tof     	  
     	  
    #plot model
        maxdata_tof=max(plot_tof.data.data[mask][0])
        tofhist_model=sum(H_all,axis=1)
        maxmodel_tof=float(max(tofhist_model))
        scaling_tof=maxdata_tof/maxmodel_tof	    	
        plot_tof.set_line(x,scaling_tof*tofhist_model,label="model")	    	
        #return maxdata_tof, maxmodel_tof
        
        maxdata_E=max(plot_E.data.data[mask][0])
        Ehist_model=sum(H_all,axis=0)
        maxmodel_E=float(max(Ehist_model))
        scaling_E=maxdata_E/maxmodel_E	    	
        plot_E.set_line(y,scaling_E*Ehist_model,label="model")	    	
     	
     	
     	#plot_E.set_line(y,scaling*sum(H_all,axis=0),label="model")
      
        if plot_single_ions!=None:
            i=0
            for ion in I.Ions:
                    hg_ion=ion.create_peakfunc(step,xg,yg,normalize_integral=False,cutoff=False,coverage_rel=0.99)
                    H_ion=hg_ion.reshape(len(x),len(y))
                    #return hg_ion
                    print(len(I.Ions), ion.name,max(ravel(H_ion))) 
                    plot_tof.set_line(x,scaling_tof*sum(H_ion,axis=1))	    	
                    plot_E.set_line(y,scaling_E*sum(H_ion,axis=0))
                    i=i+1
        return True	
    		
    
    def save_CTOF2Dhists(self,path="CTOF_2Dhists/",xbin_file="tofbins_ch.dat",ybin_file="Ebins_ch.dat",hist_file="counts_DOY150-220_step",steps=arange(0,117,1),xbins=arange(150,551,2),ybins=arange(1,102,2)):

        savetxt(path+xbin_file,xbins.T,fmt='%i')
        savetxt(path+ybin_file,ybins.T,fmt='%i')

        for step in steps:
                h=self.autoplot(mask="mask",tofrange=[xbins[0],xbins[-1]],Erange=[ybins[0],ybins[-1]],bintof=xbins[1]-xbins[0],binE=ybins[1]-ybins[0],step=step,vsw=[1,1000],remove_mask=True,show_mask=False,plottof=False,plotE=False,closeplot=True)
                if h!=None:
                    histdata=h.data.get_data()["mask"][0]
                    savetxt(path+hist_file+"%i.dat"%(step),histdata.T,fmt='%i')
                else: 
                    print("Warning: No counts measured for E/q step %i. No data could be saved for this step!"%(step))  
				
        return True

    
    
    
    
    def plot_tofE_boxes_hefti(self,steps):
        for step in steps:
            Plot=self.autoplot("Epq-step=%i"%(step),[140,601],[1,151],2,2,step)
            
            #ions=["O6+","Si7+","Fe9+"]
            mrboxes=[self.tofE_boxes_O6,self.tofE_boxes_Si7,self.tofE_boxes_Fe9,self.tofE_boxes_Fe13]
            masses=[16,28,56,56]
            charges=[6,7,9,13]
            atomnumber=[8,14,26,26]
            
            i=0
            
            #loop over ions
            while i<len(mrboxes): 
            
                #hefti
                tof_low,tof_high,energy_low,energy_high=mrboxes[i][step]
                Plot.set_line([tof_low,tof_high],[energy_low,energy_low],linewidth=2,color="r")
                Plot.set_line([tof_low,tof_high],[energy_high,energy_high],linewidth=2,color="r")
                Plot.set_line([tof_low,tof_low],[energy_low,energy_high],linewidth=2,color="r")
                Plot.set_line([tof_high,tof_high],[energy_low,energy_high],linewidth=2,color="r")
                
                #inflight
                m=masses[i]
                q=charges[i]
                tofpos=tof(step=step,m=m,q=q)
                Epos=ESSD(tofch=tofpos,m=m,Z=atomnumber)
                tofsigma=tofsig(step=step,m=m,q=q)
                Esigma=Esig(ESSD=Epos)
                
                tof_low_inf=tofpos-tofsigma
                tof_high_inf=tofpos+tofsigma
                energy_low_inf=Epos-Esigma
                energy_high_inf=Epos+Esigma
                
                Plot.set_line([tof_low_inf,tof_high_inf],[energy_low_inf,energy_low_inf],linewidth=2,color="b")
                Plot.set_line([tof_low_inf,tof_high_inf],[energy_high_inf,energy_high_inf],linewidth=2,color="b")
                Plot.set_line([tof_low_inf,tof_low_inf],[energy_low_inf,energy_high_inf],linewidth=2,color="b")
                Plot.set_line([tof_high_inf,tof_high_inf],[energy_low_inf,energy_high_inf],linewidth=2,color="b")
                
                i+=1
            

    def set_datamask(self,mask,timerange,step,tofrange,Erange,vsw_range=[0,1000],PHAmin=500,stopstep_min=0,PR_range=[0,5],Binx=2,Biny=2,brweights=False):		
        """
        method to select data, not as intuitive as with dbData, but should be faster (check!), 
        """
        #get data
        times=self.data["time"]						
        tof=self.data["tof"]		
        energy=self.data["energy"]		
        steps=self.data["steps"]		
        vsw=self.data["vsw"]
        vth=self.data["vth"]
        dsw=self.data["dsw"]
        prange=self.data["range"]
        stepmax=self.data["stepmax"]	
        
        #set mask
        times_phavalid=self.cut_PHAmin(hmin=PHAmin)
        phamin_mask=(times==times_phavalid)#dangerous to check, replace!
        #return phamin_mask
        #return times,times_phavalid
        
        tofrange_ext=[tofrange[0],tofrange[-1]+Binx]
        Erange_ext=[Erange[0],Erange[-1]+Biny]
        mask=(times>=timerange[0])*(times<=timerange[-1])*(steps==step)*(tof>=tofrange_ext[0])*(tof<=tofrange_ext[1])*(energy>=Erange_ext[0])*(energy<=Erange_ext[-1])*(vsw>=vsw_range[0])*(vsw<=vsw_range[-1])*(stepmax>=stopstep_min)*phamin_mask*(prange>=PR_range[0])*(prange<=PR_range[-1])
        
        times_masked=times[mask]	    
        tof_masked=tof[mask]
        energy_masked=energy[mask]
        steps_masked=steps[mask]
        vsw_masked=vsw[mask]
        vth_masked=vth[mask]
        dsw_masked=dsw[mask]
        prange_masked=prange[mask]
        
        if brweights==True:
            baserate=self.data["baserate"]
            baserate_masked=baserate[mask]
            maskdata=array([times_masked,tof_masked,energy_masked, steps_masked,vsw_masked,vth_masked,dsw_masked,prange_masked,baserate_masked])
        else:
            maskdata=array([times_masked,tof_masked,energy_masked, steps_masked,vsw_masked,vth_masked,dsw_masked,prange_masked])		
        return tofrange_ext,Erange_ext,maskdata
        
    def get_maskdata(self,timerange,tofrange_ext,Erange_ext,maskdata,step,Binx=2,Biny=2,Plot=True,CBlog=True,brweights=False):
            tof_masked=maskdata[1]
            energy_masked=maskdata[2]
            #return tof_masked,energy_masked,tofrange_ext,Erange_ext,Binx,Biny
            if brweights==True:
                brweights_masked=maskdata[-1]
                #return brweights_masked
                h,Bx,By=histogram2d(tof_masked,energy_masked,[arange(tofrange_ext[0],tofrange_ext[-1],Binx),arange(Erange_ext[0],Erange_ext[-1],Biny)], weights=brweights_masked)
            else:
                h,Bx,By=histogram2d(tof_masked,energy_masked,[arange(tofrange_ext[0],tofrange_ext[-1],Binx),arange(Erange_ext[0],Erange_ext[-1],Biny)])
            bx,by=Bx[:-1],By[:-1]
            ygrid,xgrid = meshgrid(by,bx)
            xg,yg= ravel(xgrid), ravel(ygrid)
            hdata=ravel(h)
            #print xg,yg
            #self.EThist=bx,by,xg,yg,h
            
            #plot histogram for checking
            if Plot==True:
                fig, ax = plt.subplots(1,1)
                my_cmap = cm.get_cmap("jet",1024*16)
                my_cmap.set_under('w')
                if brweights==True:
                    h1,bx1,by1=histogram2d(tof_masked,energy_masked,[arange(tofrange_ext[0],tofrange_ext[-1],Binx),arange(Erange_ext[0],Erange_ext[-1],Biny)],weights=brweights_masked)		
                    if CBlog==True:
                        Cont1=ax.pcolor(bx1,by1,h1.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(ravel(h1))))
                    else:
                        Cont1=ax.pcolor(bx1,by1,h1.T,cmap=my_cmap,vmin=0.99)

                else:
                    h1,bx1,by1=histogram2d(tof_masked,energy_masked,[arange(tofrange_ext[0],tofrange_ext[-1],Binx),arange(Erange_ext[0],Erange_ext[-1],Biny)])		
                    if CBlog==True:
                        Cont1=ax.pcolor(bx1,by1,h1.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(ravel(h1))))
                    else:
                        Cont1=ax.pcolor(bx1,by1,h1.T,cmap=my_cmap,vmin=0.99)

                ax.set_xlabel("tof [ch]")		
                ax.set_ylabel("energy [ch]")
                ax.set_title("timerange [DOY 1996]: %.5f - %.5f, step: %i"%(timerange[0],timerange[-1],step))
                self.plot=fig,ax
                cb1 = fig.colorbar(Cont1)
                cb1.set_label("Counts per Bin")				
                self.plot=fig, ax

            return bx,by,xg,yg,hdata



    def show_maskdata(self,timerange,tofrange_ext,Erange_ext, maskdata,step=55,Binx=2,Biny=2,Plot=True,CBlog=True,brweights=False,Xlims=[212,551],Ylims=[1,100],figx_full=13.9,figy=7.0,fontsize=32, ticklabelsize=20, legsize=25,textsize=25,textHe1_x=330,textHe1_y=8,textHe2_x=230,textHe2_y=18,textC4_x=285,textC4_y=30, textC5_x=245,textC5_y=35,textO6_x=265, textO6_y=40, textSi7_x=325,textSi7_y=38,textSi8_x=300, textSi8_y=45,textFe9_x=395, textFe9_y=32, textFe10_x=370,textFe10_y=38,save_figure=False,figpath="achapter2_finalplots/",figname="test",plot_ionnames=False, step_in_title=False, cblabel_brcor=False,plot_box=False,ETgrid=False,title_fix=False,cycle=0,plot_PR4=False,Nocolorbar=False):
        """
        used example:
        d.show_maskdata(timerange=[utimes[1],utimes[1]],tofrange_ext=a,Erange_ext=b,maskdata=c,brweights=False,step=69,Xlims=[255,315],Ylims=[25,62.1],step_in_title=True,cblabel_brcor=True,save_figure=True,figpath="achapter2_finalplots/brcor/",figname="PHAdata_ETbox_step69_vFe3954_PR43",CBlog=False,plot_box=True,Binx=1,Biny=1,ETgrid=True,title_fix=True,cycle=1,plot_PR4=True)
        """
        
        tof_masked=maskdata[1]
        energy_masked=maskdata[2]
        #return tof_masked,energy_masked,tofrange_ext,Erange_ext,Binx,Biny
        if brweights==True:
            brweights_masked=maskdata[-1]
            #return brweights_masked
            h,Bx,By=histogram2d(tof_masked,energy_masked,[arange(tofrange_ext[0],tofrange_ext[-1],Binx),arange(Erange_ext[0],Erange_ext[-1],Biny)], weights=brweights_masked)
        else:
            h,Bx,By=histogram2d(tof_masked,energy_masked,[arange(tofrange_ext[0],tofrange_ext[-1],Binx),arange(Erange_ext[0],Erange_ext[-1],Biny)])
        bx,by=Bx[:-1],By[:-1]
        ygrid,xgrid = meshgrid(by,bx)
        xg,yg= ravel(xgrid), ravel(ygrid)
        hdata=ravel(h)
        #print xg,yg
        #self.EThist=bx,by,xg,yg,h
        
        #plot histogram for checking
        if Plot==True:
            fig, ax = plt.subplots(1,1,figsize=(figx_full,figy))
            my_cmap = cm.get_cmap("jet",1024*16)
            my_cmap.set_under('w')
            if brweights==True:
                h1,bx1,by1=histogram2d(tof_masked,energy_masked,[arange(tofrange_ext[0],tofrange_ext[-1],Binx),arange(Erange_ext[0],Erange_ext[-1],Biny)],weights=brweights_masked)		
                if CBlog==True:
                    Cont1=ax.pcolor(bx1,by1,h1.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(ravel(h1))))
                else:
                    Cont1=ax.pcolor(bx1,by1,h1.T,cmap=my_cmap,vmin=0.99)

            else:
                h1,bx1,by1=histogram2d(tof_masked,energy_masked,[arange(tofrange_ext[0],tofrange_ext[-1],Binx),arange(Erange_ext[0],Erange_ext[-1],Biny)])		
                if CBlog==True:
                    Cont1=ax.pcolor(bx1,by1,h1.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(ravel(h1))))
                else:
                    Cont1=ax.pcolor(bx1,by1,h1.T,cmap=my_cmap,vmin=0.01,vmax=amax(h1))

            ax.set_xlabel(r"$\rm{TOF [ch]}$",fontsize=fontsize)		
            ax.set_ylabel(r"$\rm{ESSD [ch]}$",fontsize=fontsize)
            
            if step_in_title==True:
                ax.set_title(r"$ \rm{DOY: \ %i - %i \ 1996, \ E/q-step: \ } %i$"%(timerange[0],timerange[-1],step),y=1.01,fontsize=fontsize)
            if title_fix:
                ax.set_title(r"$ \rm{DOY: \ %i \ 1996, \ CTOF-Cycle: \ %i, \ E/q-step:\ %i}$"%(timerange[0],cycle,step),fontsize=fontsize,y=1.01)
            else:
                ax.text(.8,.9,r"$\rm{E/q-step: \ } %i$"%(step),horizontalalignment='center',transform=ax.transAxes,fontsize=fontsize)
            
            if plot_ionnames==True:
                #ax.text(textHe1_x,textHe1_y,r"$\rm{He}^{ \rm{\plus}}$",fontsize=textsize)
                ax.text(textHe2_x,textHe2_y,r"$\rm{He}^{2 \rm{\plus}}$",fontsize=textsize)
                ax.text(textC4_x,textC4_y,r"$\rm{C}^{4\rm{\plus}}$",fontsize=textsize)
                ax.text(textC5_x,textC5_y,r"$\rm{C}^{5\rm{\plus}}$",fontsize=textsize)
                ax.text(textO6_x,textO6_y,r"$\rm{O}^{6\rm{\plus}}$",fontsize=textsize)
                ax.text(textSi7_x,textSi7_y,r"$\rm{Si}^{7\rm{\plus}}$",fontsize=textsize)
                ax.text(textSi8_x,textSi8_y,r"$\rm{Si}^{8\rm{\plus}}$",fontsize=textsize)
                ax.text(textFe9_x,textFe9_y,r"$\rm{Fe}^{9\rm{\plus}}$",fontsize=textsize)
                ax.text(textFe10_x,textFe10_y,r"$\rm{Fe}^{10\rm{\plus}}$",fontsize=textsize)
            
            if plot_box==True:
                linewidth_box=5
                color_box="g"	
                #ax.plot([275,285],[37,35],linewidth=linewidth_box,color=color_box)
                #ax.plot([275,285],[41,38],linewidth=linewidth_box,color=color_box)		
                #ax.plot([275,275],[37,41],linewidth=linewidth_box,color=color_box)
                #ax.plot([285,285],[35,38],linewidth=linewidth_box,color=color_box)		
            
                polx=[275,285,285,275]
                #poly=[38,35,38,41]
                poly=[38,35,38.6,42]
                ax.fill(polx,poly,color="m",alpha=0.5)
            
            if plot_PR4==True:
                x_P4up=array([250,260,270,280,290,300,305,315])
                y_P4up=array([53,48,44,40.3,37,34.5,33.5,31.5])
                ax.plot(x_P4up,y_P4up,linewidth=2,color="g")
            
                x_P4low=array([250,260,270,280,290])
                y_P4low=array([33,30,28,26,24])
                ax.plot(x_P4low,y_P4low,linewidth=2,color="orange")
            
                x_P3up=array([270,280,290,300,310,315])
                y_P3up=array([63,58,53,49,45,43])
                ax.plot(x_P3up,y_P3up,linewidth=2,color="royalblue")
            
                x_P2up=array([303,305,310,315,320])
                y_P2up=array([61.99,61,59,57,55])
                ax.plot(x_P2up,y_P2up,linewidth=2,color="darkblue")
            
            
            self.plot=fig,ax
            if Nocolorbar!=True:
                cb1 = fig.colorbar(Cont1)
                if cblabel_brcor==True and brweights==True:
                    cb1.set_label(r"$\rm{corr. \ counts \ per \ bin}$",fontsize=fontsize)				
                else:
                    cb1.set_label(r"$\rm{counts \ per \ bin}$",fontsize=fontsize)				
                for ctick in cb1.ax.get_yticklabels():
                    ctick.set_fontsize(ticklabelsize)
                #self.plot=fig, ax
            
            #xticks=arange(200,551,50)
            xticks=arange(100,600,10)
            xticks_minor=arange(100,600,5)
            yticks=arange(0,200,10)
            yticks_minor=arange(0,200,5)
            
            ax.set_xticks(xticks_minor, minor=True)
            ax.set_xticks(xticks, minor=False)
            ax.set_yticks(yticks_minor, minor=True)
            ax.set_yticks(yticks, minor=False)
            
            ax.tick_params(axis="x", labelsize=ticklabelsize)
            ax.tick_params(axis="y", labelsize=ticklabelsize)
            if ETgrid==True:
                ax.grid(which="both",linewidth=1,linestyle="-")
                
            ax.set_xlim([Xlims[0],Xlims[-1]])
            ax.set_ylim([Ylims[0],Ylims[-1]])

            if save_figure==True:
                plt.savefig(figpath+figname,bbox_inches='tight')

        return bx,by,xg,yg,hdata
			

    def cut_PHAmin(self,hmin=500):
            utimes=unique(self.data["time"])
            timestamp_extend=utimes[-1]+5./(24*3600)
            utimes_extend=concatenate([utimes,array([timestamp_extend])])
            h,bt=histogram(self.data["time"],bins=utimes_extend)
            mt=(h>=hmin)
            utimes_phavalid=utimes[mt]	
            mask_phavalid=in1d(self.data["time"],utimes_phavalid)
            times_phavalid=self.data["time"]*mask_phavalid
            return times_phavalid	

    def get_ESAstop(self,plot_stopstep=False,CBlog=True):
        
        utimes=unique(self.data["time"])
        timestamp_extend=utimes[-1]+5./(24*3600)
        utimes_extend=concatenate([utimes,array([timestamp_extend])])
        
        ss=searchsorted(utimes,self.data["time"])
        
        steps=arange(0,116,1)
        steps_extend=arange(0,117,1)
        s=array([steps]*len(utimes))
        
        h,bt,bs=histogram2d(self.data["time"],self.data["steps"],bins=[utimes_extend,steps_extend])
        ms=h>0
        ustep_max=amax(s*ms,axis=1) 
        step_max=ustep_max[ss]
        
        if plot_stopstep==True:
            fig, ax = plt.subplots(1,1)
            my_cmap = cm.get_cmap("jet",1024*16)
            my_cmap.set_under('w')
            if CBlog==True:
                Cont1=ax.pcolor(utimes,steps,h.T,cmap=my_cmap, norm=colors.LogNorm(vmin=1,vmax=max(ravel(h))))
            else:
                Cont1=ax.pcolor(utimes,steps,h.T,cmap=my_cmap,vmin=1,vmax=max(ravel(h)))
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("PHA counts")				
            ax.set_xlabel("time [DOY 1996]")		
            ax.set_ylabel("E/q-step")
        return ustep_max,step_max	
			


    def plot_cyclehist(self,vsw_range=[530,550],hmin=500,figx=6.8, figy=7,fontsize=32,ticklabelsize=20,adjust_top=0.9,title_lift=1.01, xlim=[0,117],ylim=[0,100],save_figure=False,path="/home/asterix/janitzek/ctof/achapter6_finalplots/",figname="testhist",stepstop=0,text_x=49.5,text_yrel=.9):
        ut,i=unique(self.data["time"],return_index=True)
        uvsw=self.data["vsw"][i]
        ustepmax=self.data["stepmax"][i]

        tv=self.cut_PHAmin(hmin=hmin)
        utv=unique(tv)

        uvswmask=(uvsw>=vsw_range[0])*(uvsw<=vsw_range[-1])
        utsw=ut[uvswmask]
        utvalid=intersect1d(utv,utsw)
        umask=in1d(ut,utvalid)

        fig, ax = plt.subplots(1,1,figsize=(figx, figy))
        fig.subplots_adjust(top=adjust_top)
        h,bx,by=ax.hist(ustepmax[umask],arange(0,117))
        #if minimum_stopstep!=None:
        #print "cycles with stop-step of at least Epq-step = %i"%(minimum_stopstep), sum(h[(bx[:-1])>=minimum_stopstep])
        ax.set_xlabel(r"$\rm{E/q-step \ maximum}$",fontsize=fontsize)
        ax.set_ylabel(r"$\rm{number \ of \ cycles}$",fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=ticklabelsize)
        ax.tick_params(axis="y", labelsize=ticklabelsize)
        #ax.set_title(r"$\rm{v}_P \in \ [%i\, \rm{km/s} \ , \ %i\, \rm{km/s]}$"%(vsw_range[0],vsw_range[-1]),fontsize=fontsize,y=title_lift)
        ax.text(text_x,text_yrel*ylim[-1],r"$\rm{v}_P \in \ [%i\, \rm{km/s} \ , \ %i\, \rm{km/s]}$"%(vsw_range[0],vsw_range[-1]),fontsize=fontsize)
                        

        ax.set_xlim([xlim[0],xlim[-1]])
        ax.set_ylim([ylim[0],ylim[-1]])

        if save_figure==True:
            plt.savefig(path+figname,bbox_inches='tight')
        else:
            plt.show()	
                
        N=sum(sum(h[bx[:-1]>=stepstop]))
                
        return N

							



    def estimate_gk_peakpars_2d_sequence_fast(self,ionlist,steps, timerange=[174,220],tofrange=[180,611],Erange=[1,121], PR_range=[1,4],vsw_range=[0,1000], vproton_RCfilter=True, peakshapes="kappa_moyalpar_Easym",tailranges=None,Minimization="Normal",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=1.e-04,include_countweights=True,fitmincounts=10,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=0,check_fitarea=False,Plot_start_model=False,plot_chi2_label=True,Plot_2dcontour=True,figy_size=7, plot_2dcontour_xlabel=True,plot_model_title=True,plot_modelcontour=True, model_contourcolor="m", plot_datacontour=True, plot_elementhypes=True, elementhyp_markersize=5,plot_vswfilter=False, ionpos_markercolor="k", ionpos_markeredgecolor="k",CBlog=True, Plot_residuals=False, absmax=10., Plot_tofproj=False, plot_tof_errbars=False, plot_toflegend=False, Plot_tofproj_log=False, plot_Eproj=False,figuresize="fullwidth",save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_rectangle=None,TOFPOS_shift_O6=False,TOFPOS_shift_Si7=False,TOFPOS_shift_Fe10=False,xrange_plot=None,yrange_plot=None):
      
        """
        Status: The correctness of the overall optimization is checked.
        TODO: -describe in detail how to use the method: explain all input, options/flags!
        #default parameters for best fits
        ionlist=L_Arnaud_fullstable_noHe 
        tailranges=[[0.00325,0.]]
        FTOL=1.e-08

        
        - Plot residuals argument is "absolute", "relative" or False.
        -Note that the termination tolerance parameter "Termtol" has a different meaning depending on the used minimization algorithm (see the detailed documentation of the "scipy.optimize" modul) and therefore should be chosen in dependence of the method! Empirically, the default value has the necessary accuracy for all methods for the CTOF counting statistics. 
        TODO: implement minor "todo" points below in this method!
        TODO: Check and outsource plot methods (when the Figures for the papaer are plotted!
        TODO: Include BFGS and Nelder minimization algorithms in this method for completeness! 
        TODO: get rid of pres=abs(res), by defining the fit function properly in the first place (in auxilliary functions), This is not a physical issue because the chi2 is calculated correctly, but not transparent to the user!
        TODO: Try to make it work stable with full ionlist and analytical Poisson expression! (e.g. by taking as start values the counts at the peak position
        
        """
        t0=clock()
        
        
        #initialize general quantities for this method
        #TODO: outsource plot related definitions!
        if xrange_plot==None:
            xrange_plot=tofrange
        if yrange_plot==None:
            yrange_plot=Erange
        
        tofrange=array(tofrange)
        Erange=array(Erange)
        tofbin=2
        Ebin=2
        binshift_contour_x=tofbin/2.
        binshift_contour_y=Ebin/2.
        ion_peakbounds="countlevel"
        countlevel_rel=0.001
        vswfilter=[]
        if TOFPOS_shift_O6==True:
            path0="fit_figures/tofshift/O6/"
        elif TOFPOS_shift_Si7==True:
            path0="fit_figures/tofshift/Si7/"
        elif TOFPOS_shift_Fe10==True:
            path0="fit_figures/tofshift/Fe10/"
        else:
            path0="fit_figures/test/"
        print("Model ion list, (length):",ionlist, len(ionlist))
        if peakshapes=="gauss2d":
            N_tailpar=1
        elif peakshapes=="kappa_moyalpar_Easym":
            N_tailpar=len(tailranges)
        print("Peakshape model:",peakshapes)   
        
        plot_tofslices=None
        plot_Efits=False 
        Ecountmin=1
        He2_cut=None
        plot_tofresiduals=False
        contourcolor=model_contourcolor
        #xrange_plot=tofrange
        #yrange_plot=Erange
        
        adjust_top=0.9
        title_lift=1.01
        Plot_residuals_rel=False
        plot_chi2=plot_chi2_label
        #plot_datacontour=True
        plot_modelcontour=plot_modelcontour
        plottof_element_colors=True
        plottof_single_ions=True
        contour_elementcolors=False
        plot_contour_legend=False
        legtof=plot_toflegend
        #plot_rectangle=None
        figpath=""
        
        rcolor="k"
        rwidth=2.5
        rstyle="-"
        
        #figx_full=13.9
        figx_full=10.5
        
        figx_half=6.8
        figy=figy_size#7or 9
        #initialize plot format quantities for this method
        if figuresize=="fullwidth":
            fontsize=20
            ticklabelsize=16
            legsize=20
        elif figuresize=="halfwidth":
            fontsize=28
            ticklabelsize=20
            legsize=25
        
        #TOF_proj_legend position (out of plot)
        bbox_x=1.22 
        bbox_y=1.0
            
            
        #initialize start intensities
        #TODO: use data counts at the peak position (optional)
        start_intensities=ones((len(ionlist)))
        
        #Initialize fitmodel
        I=IonDist()
        I.add_ionlist(ionlist,peakshapes,intensities=start_intensities)
        
        #remove ion species from the model that are out of TOF or ESSD range in at least one of the selected Epq-steps
        #TODO put this in if-condition: "out-of-range-ions" should be removed"
        Ionlist_removed=[]
        """
        for step in steps:
            
            ilr=I.posclean_ionlist(step,tofrange,Erange)
            Ionlist_removed.append(ilr)
            
        ionlist=I.Ion_names
        print "ionlist range-reduced:", ionlist, len(ionlist)
        """
        
        #iterate over Epq steps
        Ioncounts=zeros((len(steps),N_tailpar,len(ionlist)))
        Ioncounts_mult=zeros((len(steps),N_tailpar,len(ionlist)))
        P=zeros((len(steps),N_tailpar,len(ionlist)))
        P_errors=zeros((len(steps),N_tailpar,len(ionlist)))
        P_errors_rel=zeros((len(steps),N_tailpar,len(ionlist)))
        Chi2=zeros((len(steps),N_tailpar))
        Chi2_P=zeros((len(steps),N_tailpar))
        Chi2_tof=zeros((len(steps),N_tailpar))
        Absres_rel=zeros((len(steps),N_tailpar))
        j=0
        for step in steps:
            print("Processing E/q step:", step)
            
            #filter out random counts via vsw-range (optional)
            if vproton_RCfilter==True:
                vsw_range=vsw_RCfilter(step=step)
                #return vsw_range
            
            #get data
            if baserate_correction==True:
                mask="step: %s"%(step)
                tofrange_ext,Erange_ext,maskdata=self.set_datamask(mask,timerange,step,tofrange,Erange,vsw_range=vsw_range,Binx=tofbin,Biny=Ebin, PR_range=PR_range,brweights=True,stopstep_min=Epq_stopstep_min,PHAmin=PHAmin)
                bx,by,xg,yg,hdata=self.get_maskdata(timerange,tofrange_ext,Erange_ext,maskdata,step,Binx=tofbin,Biny=Ebin,Plot=False,brweights=True)
                x=arange(tofrange[0],tofrange[-1],tofbin)
                y=arange(Erange[0],Erange[-1],Ebin)
                bx,by,xg,yg,hdata_truecounts=self.get_maskdata(timerange,tofrange_ext,Erange_ext,maskdata,step,Binx=tofbin,Biny=Ebin,Plot=False,brweights=False)
                mincounts_mask=(hdata_truecounts>=fitmincounts)
            else:
                mask="step: %s"%(step)
                tofrange_ext,Erange_ext,maskdata=self.set_datamask(mask,timerange,step,tofrange, Erange,vsw_range=vsw_range,Binx=tofbin, Biny=Ebin,PR_range=PR_range, stopstep_min=Epq_stopstep_min,PHAmin=PHAmin)
                bx,by,xg,yg,hdata=self.get_maskdata(timerange,tofrange_ext,Erange_ext,maskdata,step,Binx=tofbin,Biny=Ebin,Plot=False)
                x=arange(tofrange[0],tofrange[-1],tofbin)
                y=arange(Erange[0],Erange[-1],Ebin)
                mincounts_mask=(hdata>=fitmincounts)
            #return hdata
            hdata=hdata*mincounts_mask
            
            
            #include counting errors (sqrt(n)) as weights in the fit, these weights are only used if regular chi2 minimization is used! 
            if baserate_correction==True and br_scaling==True:
                hdata_truecounts[hdata_truecounts==0]=1
                h_err=sqrt(hdata_truecounts)
                hscale=hdata/(hdata_truecounts.astype(float))
                hdata_err=h_err*hscale
                hdata_err[hdata_err==0]=1.#no better approximation possible within normal error assumption via normal chi2
            else:
                hdata_err=sqrt(hdata)
                hdata_err[hdata_err==0]=1.#no better approximation possible within normal error assumption via normal chi2                        
            weights=1./hdata_err
                    
            
            #iterate over tailrange parameters
            Ioncounts_step=zeros((N_tailpar,len(ionlist)))
            Ioncounts_mult_step=zeros((N_tailpar,len(ionlist)))
            P_step=zeros((N_tailpar,len(ionlist)))	
            P_step_errors=zeros((N_tailpar,len(ionlist)))	
            P_step_errors_rel=zeros((N_tailpar,len(ionlist)))	
            Chi2_step=zeros((N_tailpar))
            Chi2_P_step=zeros((N_tailpar))
            Chi2_tof_step=zeros((N_tailpar))
            Absres_step=zeros((N_tailpar))
            k=0
            while k<N_tailpar:
                if peakshapes=="kappa_moyalpar_Easym":
                    print("Processing tail parameter:", tailranges[k])
                
                #only include ions in fit that have at least one count within their (99%-contribution) peak environment (the counts of the other ion stay at their initial value of zero) 
                ionlist_stepfit=[]
                for ion in I.Ions:
                    #print ion.name,ion.posTOF[step],ion.posE[step]
                    ionmask=ion.create_peakmincounts_mask(step,xg,yg,countlevel_rel=countlevel_rel,Plot=False,CBlog=False,vmin=1e-10)
                    if sum(hdata*ionmask)>0:
                        ionlist_stepfit.append(ion.name)
                #return True
                ionlist_stepfit_mask=in1d(array(ionlist),array(ionlist_stepfit))
                ionlist_stepfit_inds=where(ionlist_stepfit_mask==True)[0]
                I_stepfit=IonDist()
                I_stepfit.add_ionlist(ionlist_stepfit,peakshapes,intensities=ones((len(ionlist_stepfit))))
                #return I, I_stepfit    
                
                #insert tailrange parameter into the fitmodel for each ion
                Ion_ind=0
                for ion in I_stepfit.Ions:
                    if peakshapes=="kappa_moyalpar_Easym": 
                        tailrange=tailranges[k]
                        ion.tailscale_TOF=tailscale_linear(tofch=ion.posTOF,tail_grad=tailrange[0],tail_offset=tailrange[1])
                        if ion.tailscale_TOF[step]<0:
                            ion.tailscale_TOF=ones((len(ion.steps)))#out of range value, just a work-around for now#TODO:check that when doing the final check of the calibration
                        ion.peakpars_values[step][4]=ion.tailscale_TOF[step]	
                    Ion_ind=Ion_ind+1
                
                    #TOFPOS_shift: Only use to illustrate model sensitivity!
                    if TOFPOS_shift_O6==True:
                        #shift O6+ by +2 TOF channels
                        if ion.mass==16 and ion.charge==6:
                            ion.posTOF[step]=ion.posTOF[step]+2 
                            ion.peakpars_values[step][1]=ion.posTOF[step]
                    if TOFPOS_shift_Si7==True:
                        #shift Si7+ by +2 TOF channels
                        if ion.mass==28 and ion.charge==7:
                            ion.posTOF[step]=ion.posTOF[step]+2 
                            ion.peakpars_values[step][1]=ion.posTOF[step]
                    if TOFPOS_shift_Fe10==True:
                        #shift Fe9+ by +2 TOF channels
                        if ion.mass==56 and ion.charge==10:
                            ion.posTOF[step]=ion.posTOF[step]+2 
                            ion.peakpars_values[step][1]=ion.posTOF[step]
                            
                
                
                #create response model function with normed peak heights as (only) free parameters
                hgs=I_stepfit.create_peakfuncs_norm(step=step,xg=xg,yg=yg,ion_peakbounds=ion_peakbounds,coverage_rel=0.99,countlevel_rel=countlevel_rel)
                #return hgs
                
                #create fit function and start parameters for the heights and start model
                #TODO: If intensities==None,then the counts at the ion peak position should be taken as approximation!
                fitfunc = lambda p,hgs: distheight_multiplication(heights=p,hgs=hgs,sum_dists=True)*mincounts_mask
                if list(start_intensities)!=None:
                    p0=array(start_intensities)[ionlist_stepfit_mask].astype(float)
                else:
                    p0 = array([10]*len(I_stepfit.Ions))					
                
                if Minimization=="Normal":
                    print("Minimization: Normal")
                    if Minimization_algorithm=="Levenberg-Marquardt":
                        print("Minimizer algorithm: Levenberg-Marquardt")
                        if include_countweights==True:
                            print("Weights are included in minimization")
                            #errfunc = lambda p, hgs, h: (fitfunc(p,hgs)-h)*weights
                            errfunc = lambda p, hgs, h, w: (fitfunc(p,hgs)-h)*w
                        else:
                            print("Weights are NOT included in minimization")
                            errfunc = lambda p, hgs, h: (fitfunc(p,hgs)-h)
                        if Termtol==None:
                            #args = leastsq(errfunc, p0[:], args=(hgs,hdata),full_output=1)
                            args = leastsq(errfunc, p0[:], args=(hgs,hdata,weights),full_output=1)
                        else:
                            print("Termination tolerance parameter ftol:",Termtol)
                            #args = leastsq(errfunc, p0[:], args=(hgs,hdata),full_output=1,ftol=Termtol)
                            args = leastsq(errfunc, p0[:], args=(hgs,hdata,weights),full_output=1,ftol=Termtol)
                        pcov=args[1]
                        pres=abs(args[0])
                        Nfev=args[2]["nfev"]
                        if len(shape(pcov))>0:
                            fit_unvalid=False#flag for storage of chi2-values			
                            print("Optimization terminated successfully!")    
                            print("Fitresults (ion peak heights):", pres)
                            print("Covariance matrix pcov:",pcov) 
                        else:
                            print("Optimization did NOT terminate successfully!")    
                            fit_unvalid=True
                            pres=zeros((len(args[0])))-1
                            Nfev=-1
            
                        #obtain optimal model from fitted parameters
                        hfit = fitfunc(pres,hgs)
                        H_model_fitted=hfit.reshape(len(x),len(y))
                        hmax=max(ravel(H_model_fitted))
                        
                        #calculate chi-squared
                        if fit_unvalid==True:
                            Chi_squared=-1
                            Chi2_red=-1
                            p_errors=zeros((len(pres)))-1
                            p_errors_rel=zeros((len(pres)))-1
                        else:	
                            if include_countweights==True:
                                Chi_squared=sum(((hdata-hfit)*weights)**2)
                            else:
                                Chi_squared=sum((hdata-hfit)**2)
                            Nbins_valid=float(len(hdata[mincounts_mask]))
                            Npar=len(pres)
                            Chi2_red=Chi_squared/(Nbins_valid-Npar)
                        
                            #Calculation of fit parameter errors (i.e. of fitted peak height errors). This is only possible straightforward from the covariance matric "pcov" for regular chi2 minimization! 
                            if baserate_correction==True and br_scaling==True:
                                residvar=sum(((hdata-hfit)/hscale)**2)/(float(len(hdata[mincounts_mask]))-len(pres))
                            else:
                                residvar=sum(((hdata-hfit))**2)/(float(len(hdata[mincounts_mask]))-len(pres))
                            p_errors=sqrt(diag(pcov)*residvar)
                            p_errors_rel=p_errors/(pres.astype("float"))
                    
                    else: 
                        if include_countweights==True:
                            print("Weights are included in minimization")
                            #errfunc = lambda p, hgs, h: sum(((fitfunc(p,hgs)-h)*weights)**2)
                            errfunc = lambda p, hgs, h, w: sum(((fitfunc(p,hgs)-h)*w)**2)
                            if Minimization_algorithm=="Powell":
                                print("Minimization algorithm: Powell")
                                print("Termination tolerance parameters xtol,ftol:",Termtol)    
                                #optimized = minimize(fun=errfunc, x0=p0, args=((hgs,hdata)),method="Powell",options={'xtol':Termtol,'ftol':Termtol})
                                optimized = minimize(fun=errfunc, x0=p0, args=((hgs,hdata,weights)),method="Powell",options={'xtol':Termtol,'ftol':Termtol})
                            elif Minimization_algorithm=="BFGS":
                                print("Minimization algorithm: BFGS")
                                print("Termination tolerance parameter gtol:",Termtol)
                                #optimized = minimize(fun=errfunc, x0=p0, args=((hgs,hdata)),method="BFGS",options={'gtol':Termtol})
                                optimized = minimize(fun=errfunc, x0=p0, args=((hgs,hdata,weights)),method="BFGS",options={'gtol':Termtol})
                        else:
                            print("Weights are NOT included in minimization")
                            errfunc = lambda p, hgs, h: sum((fitfunc(p,hgs)-h)**2)
                            if Minimization_algorithm=="Powell":
                                print("Minimization algorithm: Powell")
                                print("Termination tolerance parameters xtol,ftol:",Termtol)    
                                optimized = minimize(fun=errfunc, x0=p0, args=((hgs,hdata)),method="Powell",options={'xtol':Termtol,'ftol':Termtol})
                            elif Minimization_algorithm=="BFGS":
                                print("Minimization algorithm: BFGS")
                                print("Termination tolerance parameter gtol:",Termtol)
                                optimized = minimize(fun=errfunc, x0=p0, args=((hgs,hdata)),method="BFGS",options={'gtol':Termtol})
                        fitsucces=optimized.success
                        if fitsucces==True:  
                            fit_unvalid=False
                        else:
                            fit_unvalid=True
                            print("Optimization did NOT terminate successfully!")
                        print(optimized.message)
                        pres=abs(optimized.x)
                        Nfev=optimized.nfev
                        print("fitresults (ion peakheights):", pres)
                    
                        #obtain optimal model from fitted parameters
                        hfit = fitfunc(pres,hgs)
                        H_model_fitted=hfit.reshape(len(x),len(y))
                        hmax=max(ravel(H_model_fitted))
                        
                        #calculate chi-squared
                        if fit_unvalid==True:
                            Chi_squared=-1
                            Chi2_red=-1
                        else:	
                            if include_countweights==True:
                                Chi_squared=sum(((hdata-hfit)*weights)**2)
                            else:
                                Chi_squared=sum((hdata-hfit)**2)
                            Nbins_valid=float(len(hdata[mincounts_mask]))
                            Npar=len(pres)
                            Chi2_red=Chi_squared/(Nbins_valid-Npar)
                        
                        ##fit parameter error calculation for other methods than Levenberg-Marquardt not straight-forward, because no pcov is given automatically, but in principle the error derivation should be possible.TODO: check that!
                        p_errors=zeros((len(pres)))-1
                        p_errors_rel=zeros((len(pres)))-1
                        
                    
                    
                elif Minimization=="Poisson":
                    print("Minimization: Poisson")
                    if Poisson_Minapprox==True:
                        errfunc = lambda p, hgs, h: sum(2*(fitfunc(p,hgs)-h)+(2*h+1)*log((2*h+1)/(2*fitfunc(p,hgs)+1)))
                        print("Poisson approximation is used in the minimization.")
                    else:
                        errfunc = lambda p, hgs, h: sum(fitfunc(p,hgs))-sum(h*log(fitfunc(p,hgs)))
                        print("Analytically exact Poisson minimization is used.")
                    
                    if Minimization_algorithm=="Powell":                            
                        print("Minimization algorithm: Powell")
                        print("Termination tolerance parameters xtol,ftol:",Termtol)
                        optimized = minimize(fun=errfunc, x0=p0, args=((hgs,hdata)),method="Powell", options={'xtol':Termtol,'ftol':Termtol})
                    elif Minimization_algorithm=="BFGS":
                        print("Minimization algorithm: BFGS")
                        print("Termination tolerance parameter gtol:",Termtol)
                        optimized = minimize(fun=errfunc, x0=p0, args=((hgs,hdata)),method="BFGS",options={'gtol':Termtol})
                    fitsucces=optimized.success
                    print(optimized.message)
                    pres=abs(optimized.x)
                    Nfev=optimized.nfev
                    print("fitresults (ion peakheights):", pres)
                    Chi_squared=-1
                    Chi2_red=-1
                    p_errors=zeros((len(pres)))-1
                    p_errors_rel=zeros((len(pres)))-1

                    #obtain optimal model from fitted parameters
                    hfit = fitfunc(pres,hgs)
                    H_model_fitted=hfit.reshape(len(x),len(y))
                    hmax=max(ravel(H_model_fitted))
                    

                #TODO:Check this part, when residual calculation is needed!
                Ionmodels=distheight_multiplication(heights=pres,hgs=hgs,sum_dists=False)*mincounts_mask
                ioncounts=sum(Ionmodels,axis=1)                    
                
                
                gridmask_allions=zeros((len(hdata)),dtype="bool")                            
                m=0
                while m<len(pres):
                    ionmodel=Ionmodels[m]
                    ionmask=(ionmodel>0)
                    gridmask_allions=gridmask_allions+ionmask
                    m=m+1
                #return ioncounts,hgs,hdata*(mincounts_mask*gridmask_allions)    
                ioncounts_mult=multres(ioncounts,hgs,hdata*(mincounts_mask*gridmask_allions))
                print("ioncounts_mult:",ioncounts_mult,max(ioncounts_mult),len(ioncounts_mult))
                #return ioncounts_mult,hdata*(mincounts_mask*gridmask_allions)
                
                
                absres=sum(abs(hdata-hfit))/float(sum(hdata))
                H_data=hdata.reshape(len(x),len(y))
                
                
                #Plotting of the fits starts here: TODO: this whole part should be "outsourced" as individual methods or even a small plotting class  
                
                #plot fitted model as contour plot,if desired
                elcolors=["b","g","r","k","orange","gray","m","y","c","olive","royalblue","pink","brown","darkviolet"]
                
                if Plot_2dcontour==True:
                    z=H_data
                    fig, ax = plt.subplots(1,1,figsize=(figx_full, figy))
                    #fig = pylab.figure()
                    #ax = fig.gca()
                    #CBlog=False
                    
                    #my_cmap = cm.get_cmap("jet",1024*16)
                    if fitmincounts==0:
                        colorbar_min=1
                    else: 
                        colorbar_min=fitmincounts
                    if CBlog==True:
                        Cont=ax.pcolor(x,y,z.T,cmap="jet",norm=colors.LogNorm(vmin=colorbar_min, vmax=max(ravel(z))))
                    else:	
                        Cont=ax.pcolor(x,y,z.T,cmap="jet",vmin=colorbar_min, vmax=max(ravel(z)))
                    Cont.cmap.set_under('w')
                    cb = fig.colorbar(Cont)
                    cb.set_label(r"$\rm{counts \ per \ bin \ (C)}$",fontsize=fontsize-2)
                    for ctick in cb.ax.get_yticklabels():
                        ctick.set_fontsize(ticklabelsize)

                    #ax.set_title(r"$\rm{E/q-step: \ } %i, \ \ \ \chi^2_{red} = %.1e$"%(step,Chi2_red),fontsize=fontsize)
                    
                    if plot_chi2==True:
                        ax.text(.7,.9,r"$\rm{E/q-step: \ } %i,\ \ \ \chi^2_{red} = %.1f$"%(step,  Chi2_red),horizontalalignment='center',transform=ax.transAxes,fontsize=fontsize-2)
                    else:
                        ax.text(.8,.9,r"$\rm{E/q-step: \ } %i$"%(step),horizontalalignment='center',transform=ax.transAxes,fontsize=fontsize-2)
                    if plot_2dcontour_xlabel==True:
                        ax.set_xlabel(r"$\rm{TOF \ [ch]}$",fontsize=fontsize)
                    ax.set_ylabel(r"$\rm{ESSD \ [ch]}$",fontsize=fontsize)
                    
                    if plot_vswfilter==True:
                        ax.text(.33,.9,r"$\rm{v}_{\rm{p}} \ \in \ [%i\, \rm{km/s}, \ %i\, \rm{km/s}]$"%(vsw_range[0],vsw_range[-1]),horizontalalignment='center',transform=ax.transAxes,fontsize=fontsize-2)
                    
                    if plot_model_title==True:
                        if peakshapes=="kappa_moyalpar_Easym":
                            ax.set_title(r"$\rm{model: \ Gaussian-Moyal, \ fit: %s}$"%(Minimization),fontsize=fontsize)
                        elif peakshapes:
                            ax.set_title(r"$\rm{model: \ Gaussian, \ fit: %s}$"%(Minimization),fontsize=fontsize)
                            
                    #if check_fitarea==True:
                        #if He2_cut!=None:
                            #ax.plot(xg[mincounts_mask*He2mask],yg[mincounts_mask*He2mask],linestyle="None",marker="o",color="b")#area used to calculate chi2 calc
                        #else:
                            #ax.plot(xg[mincounts_mask*gridmask_allions],yg[mincounts_mask*gridmask_allions],linestyle="None",marker="o",color="b")#area used to calculate chi2 calc
                        #hmask=hfit>0
                        #ax.plot(xg[hmask],yg[hmask],linestyle="None",marker="o",color="k")#fitted area, must lie entirely in chi2-calc area, only then chi2 is not underestimated!
                    
                    zmax=max(ravel(z))
                    #hmax=max(ravel(H_model_fitted))
                    #return H_data,H_model_fitted
                    #ax.contour(x,y,H_model_fitted.T,levels=[1e-3*hmax,5e-3*hmax,1e-2*hmax, 5e-2*hmax, 1e-1*hmax, 5e-1*hmax],cmap=pylab.cm.RdYlBu)
                    if plot_modelcontour==True:
                        #ax.contour(x,y,H_model_fitted.T,levels=[1e-3*hmax,5e-3*hmax,1e-2*hmax, 5e-2*hmax, 1e-1*hmax, 5e-1*hmax],colors=contourcolor,linewidths=1.)
                        ax.contour(x+binshift_contour_x,y+binshift_contour_y,H_model_fitted.T,levels=[10**(-7/3.)*zmax,10**(-6/3.)*zmax,10**(-5/3.)*zmax,10**(-4/3.)*zmax, 10**(-3/3.)*zmax, 10**(-2/3.)*zmax, 10**(-1/3.)*zmax], colors=contourcolor, linewidths=1.75)
                    elif plot_modelcontour=="minimum_countourset":
                        ax.contour(x+binshift_contour_x,y+binshift_contour_y,H_model_fitted.T,levels=[10**(-3/3.)*zmax, 10**(-2/3.)*zmax, 10**(-1/3.)*zmax],colors=contourcolor,linewidths=1.75)
                    
                    
                    if plot_datacontour==True:
                        ax.contour(x+binshift_contour_x,y+binshift_contour_y,z.T,levels=[10**(-7/3.)*zmax,10**(-6/3.)*zmax,10**(-5/3.)*zmax,10**(-4/3.)*zmax, 10**(-3/3.)*zmax, 10**(-2/3.)*zmax, 10**(-1/3.)*zmax],colors="k",linewidths=2.)
                        #print " data contour on"
                    elif plot_datacontour=="minimum_countourset":
                        ax.contour(x+binshift_contour_x,y+binshift_contour_y,z.T,levels=[10**(-3/3.)*zmax, 10**(-2/3.)*zmax, 10**(-1/3.)*zmax],colors="k",linewidths=2.)
                        
                    
                    for ion in I_stepfit.Ions:
                        #return I_stepfit.Ions
                        #if ion.name=="Si6+":
                        #	return ion 
                        
                        hyp_pars=[1.,0]
                        if peakshapes=="gauss2d":
                            if ion.mass==4:	
                                ax.plot(arange(tofrange[0],tofrange[-1],2.),hyp_pars[0]*ESSD(tofch=arange(tofrange[0],tofrange[-1],2.),m=ion.mass,Z=ion.atomnumber)+hyp_pars[1],color="k",linestyle="-")
                                ax.plot(tof(step,ion.mass,ion.charge),ESSD(tof(step,ion.mass,ion.charge), ion.mass,Z=ion.atomnumber), linestyle="None", marker="o", markersize=5., color="k")
                            else:	
                                TOFpos_step=ion.posTOF[step]
                                Epos_step=ion.posE[step]
                                TOFposerr_step=toferr_interpol(step=step,q=ion.charge,m=ion.mass)
                                #ax.errorbar(TOFpos_step,Epos_step,xerr=TOFposerr_step,linestyle="None", marker="o", markersize=5., color="k")
                                ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=5., color="k")
                                tofs=arange(tofrange[0],tofrange[-1],2.)
                                a0=2.00723e-1#time conversion in ns/ch 
                                b0=-1.46909#time offset in ns		
                                epos=hyp_pars[0]*ion.ESSD_0[step]*(a0*tofs+b0)**-2+hyp_pars[1]
                                """
                                #Epos=c1_tilt*ESSD_0*(a0*x+b0)**-2+c2_tilt
                                tofpos_step=ion.posTOF[step]
                                epos_step=hyp_pars[0]*ion.ESSD_0[step]*(a0*tofpos_step+b0)**-2+hyp_pars[1]
                                #ax.plot(tofs,epos,linestyle="None",marker="o", markersize=5., color="k")
                                ax.plot(tofpos_step,epos_step,linestyle="None",marker="o", markersize=elementhyp_markersize, color="k")
                                #print ion.name, tofpos_step, TOFpos_step, epos_step, Epos_step
                                #ax.plot(TOFpos_step,Epos_step,linestyle="None",marker="o", markersize=elementhyp_markersize, color="k")
                                """
                                if plot_elementhypes==True:
                                    ax.plot(tofs,epos,linestyle="-", markersize=elementhyp_markersize, color="k")
                        
                        else:	
                            TOFpos_step=ion.posTOF[step]
                            TOFposerr_step=toferr_interpol(step=step,q=ion.charge,m=ion.mass)
                            a0=2.00723e-1#time conversion in ns/ch 
                            b0=-1.46909#time offset in ns		
                            Epos_step=hyp_pars[0]*ion.ESSD_0[step]*(a0*ion.posTOF[step]+b0)**-2+hyp_pars[1]
                            tofs=arange(tofrange[0],tofrange[-1],2.)
                            epos=hyp_pars[0]*ion.ESSD_0[step]*(a0*tofs+b0)**-2+hyp_pars[1]
                            #ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=5., color="k")
                            #ax.errorbar(TOFpos_step,Epos_step,xerr=TOFposerr_step,linestyle="None", marker="o", markersize=5., color="k")
                            
                            #ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=5., color="k")
                            ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=elementhyp_markersize, color=ionpos_markercolor, markeredgewidth=2.5, markeredgecolor=ionpos_markeredgecolor)
                            if plot_elementhypes==True:
                                ax.plot(tofs,epos,linestyle="-", color="k")
                    
                    
                    
                    
                        if contour_elementcolors==True:
                            
                            alph = ''.join(re.findall(r'[a-zA-Z]', ion.name))
                            if alph=="He":
                                charge_min=""
                                charge_max="2"
                                elcolor="b"
                            elif alph=="C":
                                charge_min="4"
                                charge_max="6"
                                elcolor="g"
                            elif alph=="N":
                                charge_min="4"
                                charge_max="7"
                                elcolor="r"
                            elif alph=="O":
                                charge_min="5"
                                charge_max="8"
                                elcolor="k"
                            elif alph=="Ne":
                                charge_min="5"
                                charge_max="9"
                                elcolor="orange"
                            elif alph=="Na":
                                charge_min="4"
                                charge_max="9"
                                elcolor="pink"
                            elif alph=="Mg":
                                charge_min="4"
                                charge_max="10"
                                elcolor="m"
                            elif alph=="Al":
                                charge_min="5"
                                charge_max="11"
                                elcolor="y"
                            elif alph=="Si":
                                charge_min="5"
                                charge_max="12"
                                elcolor="c"
                            elif alph=="S":
                                charge_min="6"
                                charge_max="13"
                                elcolor="olive"
                            elif alph=="Ar":
                                charge_min="7"
                                charge_max="13"
                                elcolor="royalblue"
                            elif alph=="Ca":
                                charge_min="6"
                                charge_max="14"
                                elcolor="gray"
                            elif alph=="Fe":
                                charge_min="5"
                                charge_max="16"
                                elcolor="brown"
                            elif alph=="Ni":
                                charge_min="6"
                                charge_max="14"
                                elcolor="darkviolet"
                            
                            
                            #elcolors=["b","g","r","k","orange","gray","m","y","c","olive","royalblue","pink","brown","darkviolet"]
                            if alph!=alph_old:
                                #ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=5., color=elcolor, label=r"$\rm{%s}^{1\plus \  - \ 2\plus}$"%alph)
                                ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=5., color=elcolor, label=r"$\rm{%s}^{%s\plus} \ - \ \rm{%s}^{%s\plus} $"%(alph,charge_min,alph,charge_max))
                            else:
                                    ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=5., color=elcolor)
                            ax.plot(tofs,epos,linestyle="-", markersize=5., color=elcolor)
                            if plot_contour_legend==True:
                                ax.legend(loc="upper center", ncol=4,prop={'size': legsize},numpoints=1)
                            alph_old=alph
                    
                            
                    
                    ax.tick_params(axis="x", labelsize=ticklabelsize)
                    ax.tick_params(axis="y", labelsize=ticklabelsize)
                    #ax.grid(which='both')
            
                                        
                    if xrange_plot!=None:
                        print("xrange_plot", xrange_plot)
                        ax.set_xlim(xrange_plot[0],xrange_plot[-1])
                    if yrange_plot!=None:
                        ax.set_ylim(yrange_plot[0],yrange_plot[-1])
            
                    
            
                    #save contourplot, if desired
                    if save_figures==True:
                        plt.savefig(path0+figpath+"contour_d%i-%i_vsw%i-%i_%s_step%i.png"%(timerange[0],timerange[-1],vsw_range[0],vsw_range[-1],peakshapes,step),bbox_inches='tight')
                    else:
                        plt.show()	
            
                
                #plot relative residuals, if desired, relative residuals := res/sqrt(hdata) 
                if Plot_residuals_rel==True:
                    H_data_denom=H_data*1.
                    H_data_denom[H_data_denom==0]=1.
                    #return H_data_denom	
                    z_res=(H_data-H_model_fitted)/H_data_denom.astype(float)
                    if absmax!=None:
                        vmin_resrel=-1
                        z_res[z_res<vmin_resrel]=vmin_resrel
                    #return 1./H_data_denom, (w.reshape(len(x),len(y)))**2
                    cb_label=r"$ \rm{rel. \ residuals \ (\Delta C_{rel})}$"
                    fig, ax = plt.subplots(1,1,figsize=(figx_full, figy))
                    #fig = pylab.figure()
                    #ax = fig.gca()
                    #faster way to get white background: h = plt.hist2d(tof, essd, bins = (np.arange(0,512,1), np.arange(0,512,1)), cmap = 'jet', cmin = 1) #, norm = mpl.colors.LogNorm()
                    
                    #print	min(ravel(z_res)),max(ravel(z_res))
                    minz=min(ravel(z_res))
                    maxz=max(ravel(z_res))
                    if absmax==None:
                        Cont=ax.pcolor(x,y,z_res.T,cmap='seismic',norm = MidPointNorm(midpoint=1e-3), vmin=minz, vmax=maxz)
                        #absmax=max(abs(min(ravel(z))),max(ravel(z)))
                    else:
                        Cont=ax.pcolor(x,y,z_res.T,cmap='seismic',norm = MidPointNorm(midpoint=1e-3), vmin=vmin_resrel, vmax=1)
                    #cmap="Spectral"
                    Cont.cmap.set_under('w')
                    cb = fig.colorbar(Cont)
                    cb.set_label(cb_label,fontsize=fontsize-2)
                    for ctick in cb.ax.get_yticklabels():
                        ctick.set_fontsize(ticklabelsize)

                    #ax.set_title(r"$\rm{E/q-step: \ } %i,  \ \ \  \chi^2_{red} = %.1e $"%(step,Chi2_red),fontsize=fontsize)
                    ax.text(.7,.9,r"$\rm{E/q-step: \ } %i, \ \ \ \chi^2_{red} = %.1f$"%(step,Chi2_red),horizontalalignment='center',transform=ax.transAxes,fontsize=fontsize)
                    
                    ax.set_xlabel(r"$ \rm{TOF \ [ch]}$",fontsize=fontsize)
                    ax.set_ylabel(r"$ \rm{ESSD \ [ch]}$",fontsize=fontsize)
                    ax.tick_params(axis="x", labelsize=ticklabelsize)
                    ax.tick_params(axis="y", labelsize=ticklabelsize)
                    
                    if plot_rectangle!=None:
                        rx=plot_rectangle[0]
                        ry=plot_rectangle[1]
                        #rcolor="k"
                        #rwidth=2.
                        #rstyle="-"
                        ax.plot([rx[0],rx[-1]],[ry[0],ry[0]],linewidth=rwidth,linestyle=rstyle,color=rcolor)
                        ax.plot([rx[0],rx[-1]],[ry[-1],ry[-1]],linewidth=rwidth,linestyle=rstyle,color=rcolor)
                        ax.plot([rx[0],rx[0]],[ry[0],ry[-1]],linewidth=rwidth,linestyle=rstyle,color=rcolor)
                        ax.plot([rx[-1],rx[-1]],[ry[0],ry[-1]],linewidth=rwidth,linestyle=rstyle,color=rcolor)
                    
                    
                    
                    hmax=max(ravel(H_model_fitted))
                    #ax.contour(x,y,H_model_fitted.T,levels=[10**(-7/3.)*zmax,10**(-6/3.)*zmax,10**(-5/3.)*zmax,10**(-4/3.)*zmax, 10**(-3/3.)*zmax, 10**(-2/3.)*zmax, 10**(-1/3.)*zmax],colors="k")
                    #ax.contour(x,y,z.T,levels=[10**(-7/3.)*zmax,10**(-6/3.)*zmax,10**(-5/3.)*zmax,10**(-4/3.)*zmax, 10**(-3/3.)*zmax, 10**(-2/3.)*zmax, 10**(-1/3.)*zmax],colors="k",linewidths=1.)
                    
                    #cmap=pylab.cm.RdYlBu
                    for ion in I_stepfit.Ions:
                        if peakshapes=="gauss2d" and ion.mass==4:
                            ax.plot(arange(tofrange[0],tofrange[-1],2.),hyp_pars[0]*ESSD(tofch=arange(tofrange[0],tofrange[-1],2.),m=ion.mass,Z=ion.atomnumber)+hyp_pars[1],color="k",linestyle="-")
                            ax.plot(tof(step,ion.mass,ion.charge),ESSD(tof(step,ion.mass,ion.charge), ion.mass,Z=ion.atomnumber), linestyle="None", marker="o", markersize=5., color="k")
                        else:	
                            TOFpos_step=ion.posTOF[step]
                            a0=2.00723e-1#time conversion in ns/ch 
                            b0=-1.46909#time offset in ns		
                            Epos_step=hyp_pars[0]*ion.ESSD_0[step]*(a0*ion.posTOF[step]+b0)**-2+hyp_pars[1]
                            tofs=arange(tofrange[0],tofrange[-1],2.)
                            epos=hyp_pars[0]*ion.ESSD_0[step]*(a0*tofs+b0)**-2+hyp_pars[1]
                            ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=5., color="k")
                            ax.plot(tofs,epos,linestyle="-", markersize=5., color="k")
                    
                    if xrange_plot!=None:
                        ax.set_xlim(xrange_plot[0],xrange_plot[-1])
                    if yrange_plot!=None:
                        ax.set_ylim(yrange_plot[0],yrange_plot[-1])
                    if save_figures==True:
                        plt.savefig(path0+figpath+"rel_res2d_d%i-%i_vsw%i-%i_%s_step%i.png"%(timerange[0],timerange[-1],vsw_range[0],vsw_range[-1],peakshapes,step),bbox_inches='tight')
                    else:
                        plt.show()	
            
                
                
                
                
                
                
                
                
                #plot (relative or absolute) residuals, if desired, relative residuals := res/sqrt(hdata) 
                if Plot_residuals!=False:
                    if Plot_residuals=="absolute":
                        z_res=H_data-H_model_fitted
                        cb_label=r"$absolute count diff. per bin$"
                    elif Plot_residuals=="chi_relative":
                        H_data_denom=H_data*1.
                        H_data_denom[H_data_denom==0]=1.
                        #return H_data_denom	
                        #z_res=(H_data-H_model_fitted)/sqrt(H_data_denom)
                        z_res=(H_data-H_model_fitted)*weights.reshape(len(x),len(y))
                        cb_label=r"$ \rm{rel. \ deviation \ (\Delta C_{\sigma})}$"
                        #cb_label="test"
                    elif Plot_residuals=="relative":
                        H_data_denom=H_data*1.
                        H_data_denom[H_data_denom==0]=1.
                        #return H_data_denom	
                        z_res=(H_data-H_model_fitted)/H_data_denom.astype(float)
                        cb_label=r"$ \rm{rel. \ count \ diff. \ per \ bin}$"
                
                    fig, ax = plt.subplots(1,1,figsize=(figx_full, figy))
                    #fig = pylab.figure()
                    #ax = fig.gca()
                    
                
                    #print	min(ravel(z_res)),max(ravel(z_res))
                    minz=min(ravel(z_res))
                    maxz=max(ravel(z_res))
                    if absmax==None:
                        divnorm=colors.TwoSlopeNorm(vmin=minz, vcenter=1e-3, vmax=maxz)
                        Cont=ax.pcolor(x,y,z_res.T,cmap='seismic',norm = divnorm)
                        #absmax=max(abs(min(ravel(z))),max(ravel(z)))
                    else:
                        divnorm=colors.TwoSlopeNorm(vmin=-absmax, vcenter=1e-3, vmax=absmax)
                        Cont=ax.pcolor(x,y,z_res.T,cmap='seismic',norm = divnorm)
                    #cmap="Spectral"
                    Cont.cmap.set_under('w')
                    cb = fig.colorbar(Cont)
                    cb.set_label(cb_label,fontsize=fontsize-2)
                    for ctick in cb.ax.get_yticklabels():
                        ctick.set_fontsize(ticklabelsize)

                    #ax.set_title(r"$\rm{E/q-step: \ } %i,  \ \ \  \chi^2_{red} = %.1e $"%(step,Chi2_red),fontsize=fontsize)
                    ax.text(.7,.9,r"$\rm{E/q-step: \ } %i, \ \ \ \chi^2_{red} = %.1f$"%(step,Chi2_red),horizontalalignment='center',transform=ax.transAxes,fontsize=fontsize)
                    
                    ax.set_xlabel(r"$ \rm{TOF \ [ch]}$",fontsize=fontsize)
                    ax.set_ylabel(r"$ \rm{ESSD \ [ch]}$",fontsize=fontsize)
                    ax.tick_params(axis="x", labelsize=ticklabelsize)
                    ax.tick_params(axis="y", labelsize=ticklabelsize)
                    
                    if plot_rectangle!=None:
                        rx=plot_rectangle[0]
                        ry=plot_rectangle[1]
                        #rcolor="k"
                        #rwidth=2.
                        #rstyle="-"
                        ax.plot([rx[0],rx[-1]],[ry[0],ry[0]],linewidth=rwidth,linestyle=rstyle,color=rcolor)
                        ax.plot([rx[0],rx[-1]],[ry[-1],ry[-1]],linewidth=rwidth,linestyle=rstyle,color=rcolor)
                        ax.plot([rx[0],rx[0]],[ry[0],ry[-1]],linewidth=rwidth,linestyle=rstyle,color=rcolor)
                        ax.plot([rx[-1],rx[-1]],[ry[0],ry[-1]],linewidth=rwidth,linestyle=rstyle,color=rcolor)
                    
                    
                    hmax=max(ravel(H_model_fitted))
                    ax.contour(x,y,H_model_fitted.T,levels=[10**(-7/3.)*zmax,10**(-6/3.)*zmax,10**(-5/3.)*zmax,10**(-4/3.)*zmax, 10**(-3/3.)*zmax, 10**(-2/3.)*zmax, 10**(-1/3.)*zmax],colors="k")
                    #cmap=pylab.cm.RdYlBu
                    for ion in I_stepfit.Ions:
                        if peakshapes=="gauss2d" and ion.mass==4:
                            ax.plot(arange(tofrange[0],tofrange[-1],2.),hyp_pars[0]*ESSD(tofch=arange(tofrange[0],tofrange[-1],2.),m=ion.mass,Z=ion.atomnumber)+hyp_pars[1],color="k",linestyle="-")
                            ax.plot(tof(step,ion.mass,ion.charge),ESSD(tof(step,ion.mass,ion.charge), ion.mass,Z=ion.atomnumber), linestyle="None", marker="o", markersize=5., color="k")
                        else:	
                            TOFpos_step=ion.posTOF[step]
                            a0=2.00723e-1#time conversion in ns/ch 
                            b0=-1.46909#time offset in ns		
                            Epos_step=hyp_pars[0]*ion.ESSD_0[step]*(a0*ion.posTOF[step]+b0)**-2+hyp_pars[1]
                            tofs=arange(tofrange[0],tofrange[-1],2.)
                            epos=hyp_pars[0]*ion.ESSD_0[step]*(a0*tofs+b0)**-2+hyp_pars[1]
                            ax.plot(TOFpos_step,Epos_step,linestyle="None", marker="o", markersize=5., color="k")
                            ax.plot(tofs,epos,linestyle="-", markersize=5., color="k")
                    
                    if xrange_plot!=None:
                        ax.set_xlim(xrange_plot[0],xrange_plot[-1])
                    if yrange_plot!=None:
                        ax.set_ylim(yrange_plot[0],yrange_plot[-1])
                    if save_figures==True:
                        plt.savefig(path0+figpath+"relchi_res2d_d%i-%i_vsw%i-%i_%s_step%i.png"%(timerange[0],timerange[-1],vsw_range[0],vsw_range[-1],peakshapes,step),bbox_inches='tight')
                    else:
                        plt.show()	
            
                    
                    
                    
                    #plot rel. deviation histogram
                    fig, ax = plt.subplots(1,1,figsize=(figx_half, figy))
                    fig.subplots_adjust(top=adjust_top)
                    #fig = pylab.figure()
                    #ax = fig.gca()
                    #minz=min(z_res[H_model_fitted>0])
                    #maxz=max(z_res[H_model_fitted>0])
                    #Fcurr=histogram(ravel(z_res[H_model_fitted>0]),arange(-10,10,0.1))[0]
                    #Fcurr=histogram(ravel(z_res[H_model_fitted>0]),arange(minz,maxz+0.1,0.1))[0]
                    if absmax!=None:
                        Fcurr=histogram(ravel(z_res[H_data>0]),arange(-absmax,absmax+0.1,0.1))[0]
                    else:
                        Fcurr=histogram(ravel(z_res[H_data>0]),arange(minz,maxz+0.1,0.1))[0]							
                    #Ncurr=float(sum(Fcurr))
                    Fcurr_all=histogram(ravel(z_res[H_data>0]),arange(-1000,1000))[0]
                    Ncurr=float(sum(Fcurr_all))
                    
                    #xperf=arange(-10,10.1,0.1)
                    Fperf=gauss1d([1.,0.,1.],arange(-10,10,0.1))
                    Nperf=float(sum(Fperf))
                    
                    #indswhere
                    #return Fcurr
                    
                    if absmax!=None:
                        ax.plot(arange(-absmax,absmax,0.1),Fcurr,linewidth=2,label= r"$\rm{current \ model \ N}_{fbin}=%i$"%(Ncurr))
                    else:
                        ax.plot(arange(minz,maxz,0.1),Fcurr,linewidth=2,label=r"$\rm{current \ model \ (total \ number \ of \ fitted \ bins: \ } %.1f)$"%(Ncurr))
                    
                    ax.plot(arange(-10,10,0.1),Fperf*Ncurr/Nperf,linewidth=2.,label=r"$ \rm{ideal \ model}$")			
                    hist_nmax=amax(array([amax(Fcurr),amax(Fperf*Ncurr/Nperf)]))
                    ax.set_xlabel(r"$\rm{rel. \ deviation}$",fontsize=fontsize)
                    ax.set_ylabel(r"$\rm{number \ of \ bins}$",fontsize=fontsize)
                    ax.legend(loc="upper left",prop={'size': legsize-2})
                    ax.set_title(r"$ \rm{E/q-step: \ } %i,  \ \ \  \chi^2_{red} = %.1f$"%(step,Chi2_red),fontsize=fontsize,y=title_lift)
                    ax.set_xlim(-absmax,absmax+0.01)
                    ax.set_ylim(0,1.5*hist_nmax)
                    ax.tick_params(axis="x", labelsize=ticklabelsize)
                    ax.tick_params(axis="y", labelsize=ticklabelsize)
                    
                    """
                    if xrange_plot!=None:
                        ax.set_xlim(xrange_plot[0],xrange_plot[-1])
                    if yrange_plot!=None:
                        ax.set_ylim(yrange_plot[0],yrange_plot[-1])
                    return True
                    """
                    
                    #save residual figure if desired 
                    if save_figures==True:
                        fig.savefig(path0+figpath+"residuals_d%i-%i_vsw%i-%i_%s_step%i.png"%(timerange[0],timerange[-1],vsw_range[0],vsw_range[-1],peakshapes,step),bbox_inches='tight')
                    else:	
                        plt.show()	

                            
                
                H_fit=H_model_fitted
                
                data_tof=sum(H_data,axis=1)
                model_tof=sum(H_fit,axis=1)
                data_tof_err=sqrt(data_tof).astype(float)
                data_tof_err[data_tof_err==0]=1.
                chi2_tof=sum((data_tof-model_tof)**2/data_tof_err**2)
                Chi2_tof_red=chi2_tof/float(len(data_tof)-len(ionlist))

                data_E=sum(H_data,axis=0)
                model_E=sum(H_fit,axis=0)
                data_E_err=sqrt(data_E).astype(float)
                data_E_err[data_E_err==0]=1.
                chi2_E=sum((data_E-model_E)**2/data_E_err**2)
                Chi2_E_red=chi2_E/float(len(data_E)-len(ionlist))
                    
            
                #plot tof-integrals (=counts vs tof) of data and fitted model, incl. single ion peaks

                """    
                ioncounts=sum(Ionmodels,axis=1)                    
                #return ioncounts,hgs,hdata*(mincounts_mask*gridmask_allions)    
                ioncounts_mult=multres(ioncounts,hgs,hdata*(mincounts_mask*gridmask_allions))
                print "ioncounts_mult:",ioncounts_mult,max(ioncounts_mult),len(ioncounts_mult)
                #return ioncounts,ioncounts_mult,hgs,Ionmodels
                #return ioncounts
                """
                
                
                if Plot_tofproj==True:		
                
                    fig, ax = plt.subplots(1,1,figsize=(figx_full-0.2*figx_full, figy))
                    #fig = pylab.figure()
                    #ax = fig.gca()
                    ax.plot(x,data_tof,linewidth=2,label=r"$\rm{data}$",color="k")
                    ax.plot(x,model_tof,linewidth=2,label=r"$\rm{model}$",color="m")
                    if legtof==True:
                        ax.legend(loc="upper right",prop={'size': legsize-2})#return x,data_tof,x,model_tof	
                
                    #print "len(data_tof)",len(data_tof)
                
                    #elements=["C","N","O",   "Ne",Mg,   "Si",    "S",   "Ca",   "Fe",   "Ni"]
                    elcolors=["r","k","orange","m","c","olive","brown","yellow","pink","gray"]
                    #["b","r","g","c","y","m","k","gray","orange","olive","brown","pink"]
                
                    elnames=[]
                    i=0
                    if plottof_single_ions==True:
                    
                        for ion in I_stepfit.Ions:
                            #if model=="gauss2d_allparfree":
                            #    ionmodel=hfit
                            #else:
                            #    ionmodel=Ionmodels[i]
                            ionmodel=Ionmodels[i]
                    
                    
                            ionmodel_tof=sum(ionmodel.reshape(len(x),len(y)),axis=1)
                            #return ionmodel_tof
                            #print i, max(ionmodel_tof),min(ionmodel_tof)
                        
                            #elcolors=["b","g","r","k","orange","gray","m","y","c","olive","royalblue","pink","brown","darkviolet"]
                        
                            if plottof_element_colors==True:
                                if ion.mass==59: 
                                    elcolor="darkviolet"
                                    elname=r"$\rm{Ni}$"
                                if ion.mass==56: 
                                    elcolor="brown"
                                    elname=r"$\rm{Fe}$"
                                if ion.mass==40: 
                                    elcolor="gray"
                                    elname=r"$\rm{Ca}$"
                                if ion.mass==32: 
                                    elcolor="g"
                                    elname=r"$\rm{S}$"
                                if ion.mass==28: 
                                    elcolor="orange"
                                    elname=r"$\rm{Si}$"
                                if ion.mass==24: 
                                    elcolor="c"
                                    elname=r"$\rm{Mg}$"
                                if ion.mass==20: 
                                    elcolor="lime"
                                    elname=r"$\rm{Ne}$"
                                if ion.mass==16: 
                                    elcolor="r"
                                    elname=r"$\rm{O}$"
                                if ion.mass==14: 
                                    elcolor="olive"
                                    elname=r"$\rm{N}$"
                                if ion.mass==12: 
                                    elcolor="b"
                                    elname=r"$\rm{C}$"
                                if ion.mass==4: 
                                    elcolor="y"
                                    elname=r"$\rm{He}$"
                            
                            
                                if elname not in elnames:  	
                                    ax.plot(x,ionmodel_tof,linewidth=1.5, color=elcolor,label="%s"%(elname))
                                    elnames.append(elname)
                                else:
                                    ax.plot(x,ionmodel_tof,color=elcolor)
                                if legtof==True:
                                    if figy==5.5:	
                                        ax.legend(loc="upper right",prop={'size': legsize-4},ncol=2)
                                    else:
                                        ax.legend(loc="upper right",prop={'size': legsize-4})
                            
                        
                            else:
                                ax.plot(x,ionmodel_tof,label="%s"%(ion.name))
                        
                            #plot ion count rate errorbars
                            hmax_tof=max(ionmodel_tof)
                            indmax=where(ionmodel_tof==hmax_tof)[0]
                            #print "errors:", i,len(p_errors),p_errors
                            hmax_toferr=p_errors_rel[i]*hmax_tof
                            
                            if plot_tof_errbars==True:
                                if (hmax_toferr<1.1)*hmax or (hmax_toferr<100*hmax_tof):
                                    ax.errorbar(x[indmax],ionmodel_tof[indmax],yerr=hmax_toferr,color="k")
                        
                            i=i+1
                    
                    ax.set_xlabel(r"$\rm{TOF \ [ch]}$",fontsize=fontsize)
                    ax.set_ylabel(r"$\rm{counts \ per \ bin}$",fontsize=fontsize)
                    #ax.legend(loc="upper right",bbox_to_anchor=(1.3, 1.1))
                    #ax.set_title(r"$\rm{E/q-step:  \ \ \  } %i$"%(step),fontsize=fontsize)
                    ax.text(.5,.9,r"$\rm{E/q-step: \ } %i$"%(step),horizontalalignment='center',transform=ax.transAxes,fontsize=fontsize)
                    ax.tick_params(axis="x", labelsize=ticklabelsize)
                    ax.tick_params(axis="y", labelsize=ticklabelsize)
                    #if len(shape(pcov))>0:
                    #    if plot_tof_errbars==True:
                    #        ax.set_title("No covariance matrix calculated!\nE/q-step: \  %i ,reduced chi_tof^2 = %.1e"%(step,Chi2_tof_red))
                    hmax=max(max(data_tof),max(model_tof))
                    #save tof-integral figure if desired
                    if xrange_plot!=None:
                        ax.set_xlim(xrange_plot[0],xrange_plot[-1])
                    ax.set_ylim(0,1.1*hmax)
                    #save logtof-integral figure if desired
                    if save_figures==True:
                        plt.savefig(path0+figpath+"tofintegrals_d%i-%i_vsw%i-%i_%s_step%i.png"%(timerange[0],timerange[-1],vsw_range[0],vsw_range[-1],peakshapes,step),bbox_inches='tight')
                    else:	
                        plt.show()
                    
                    
                if Plot_tofproj_log==True:		
                
                    fig, ax = plt.subplots(1,1,figsize=(figx_full-0.2*figx_full, figy))
                    
                    #fig = pylab.figure()
                    #ax = fig.gca()
                    ax.plot(x,data_tof,linewidth=2,color="k",label=r"$\rm{data}$")
                    ax.plot(x,model_tof,linewidth=2,color="m",label=r"$\rm{model}$")
                    ax.legend(loc="upper right",prop={'size': legsize-2})#return x,data_tof,x,model_tof	
                    #return x,data_tof,x,model_tof	
                
                    #print "len(data_tof)",len(data_tof)
                
                    if plottof_single_ions==True:
                    
                        elnames=[]
                        i=0
                        for ion in I_stepfit.Ions:
                            ionmodel=Ionmodels[i]
                    
                            ionmodel_tof=sum(ionmodel.reshape(len(x),len(y)),axis=1)
                            #return ionmodel_tof
                            #print i, max(ionmodel_tof),min(ionmodel_tof)
                        
                            if plottof_element_colors==True:
                                if ion.mass==59: 
                                    elcolor="darkviolet"
                                    elname=r"$\rm{Ni}$"
                                if ion.mass==56: 
                                    elcolor="brown"
                                    elname=r"$\rm{Fe}$"
                                if ion.mass==40: 
                                    elcolor="gray"
                                    elname=r"$\rm{Ca}$"
                                if ion.mass==32: 
                                    elcolor="g"
                                    elname=r"$\rm{S}$"
                                if ion.mass==28: 
                                    elcolor="orange"
                                    elname=r"$\rm{Si}$"
                                if ion.mass==24: 
                                    elcolor="c"
                                    elname=r"$\rm{Mg}$"
                                if ion.mass==20: 
                                    elcolor="lime"
                                    elname=r"$\rm{Ne}$"
                                if ion.mass==16: 
                                    elcolor="r"
                                    elname=r"$\rm{O}$"
                                if ion.mass==14: 
                                    elcolor="olive"
                                    elname=r"$\rm{N}$"
                                if ion.mass==12: 
                                    elcolor="b"
                                    elname=r"$\rm{C}$"
                                if ion.mass==4: 
                                    elcolor="y"
                                    elname=r"$\rm{He}$"
                            
                            
                                if elname not in elnames:  	
                                    ax.plot(x,ionmodel_tof,color=elcolor,label="%s"%(elname))
                                    elnames.append(elname)
                                else:
                                    ax.plot(x,ionmodel_tof,color=elcolor)
                                ax.legend(loc="upper right",prop={'size': legsize-4},bbox_to_anchor=(bbox_x, bbox_y))
                            else:
                                ax.plot(x,ionmodel_tof,label="%s"%(ion.name))
                        
                            #plot ion count rate errorbars
                            hmax_tof=max(ionmodel_tof)
                            indmax=where(ionmodel_tof==hmax_tof)[0]
                            #print "errors:", i,len(p_errors),p_errors
                            hmax_toferr=p_errors_rel[i]*hmax_tof
                            
                            if plot_tof_errbars==True:
                                if (hmax_toferr<1.1)*hmax or (hmax_toferr<100*hmax_tof):
                                    ax.errorbar(x[indmax],ionmodel_tof[indmax],yerr=hmax_toferr,color="k")
                            
                            #ax.errorbar(x[indmax],ionmodel_tof[indmax],yerr=hmax_toferr,color="k")
                        
                            i=i+1
                    
                    ax.set_yscale("log")
                    ax.set_xlabel(r"$\rm{TOF \ [ch]}$",fontsize=fontsize)
                    ax.set_ylabel(r"$\rm{counts \ per \ bin}$",fontsize=fontsize)
                    #ax.legend(loc="upper right",bbox_to_anchor=(1.3, 1.1))
                    #ax.set_title(r"$\rm{E/q-step:  \ \ \  } %i$"%(step),fontsize=fontsize)
                    ax.text(.5,.9,r"$\rm{E/q-step: \ } %i$"%(step),horizontalalignment='center',transform=ax.transAxes,fontsize=fontsize)
                    ax.tick_params(axis="x", labelsize=ticklabelsize)
                    ax.tick_params(axis="y", labelsize=ticklabelsize)
                    #if len(shape(pcov))>0:
                    #    if plot_tof_errbars==True:
                    #        ax.set_title("No covariance matrix calculated!\nE/q-step: \  %i ,reduced chi_tof^2 = %.1e"%(step,Chi2_tof_red))
                    hmax=max(max(data_tof),max(model_tof))
                    if xrange_plot!=None:
                        ax.set_xlim(xrange_plot[0],xrange_plot[-1])
                    ax.set_ylim(10,1e7)
                    #save tof-integral figure if desired
                    if save_figures==True:
                        plt.savefig(path0+figpath+"logtofintegrals_d%i-%i_vsw%i-%i_%s_step%i.png"%(timerange[0],timerange[-1],vsw_range[0],vsw_range[-1],peakshapes,step),bbox_inches='tight')
                    else:	
                        plt.show()
                    
                    
                    
                    
                    #plot tof residuals
                    if plot_tofresiduals==True:
                        plt.figure()
                        plt.title(r"$\rm{E/q-step \ :} %i$"%(step),fontsize=fontsize)
                        plt.plot(x,abs(data_tof-model_tof),color="k")
                        plt.plot(x,abs(data_tof-model_tof),linestyle="None",marker="o",label="abs(model residual)",color="k")
                        plt.plot(x,sqrt(data_tof),color="b")
                        plt.plot(x,sqrt(data_tof),linestyle="None",marker="o",label="poisson count error (=sqrt(data_tof))",color="b")
                        plt.xlabel("tof channel")
                        plt.ylabel("absolute counts per bin")
                        plt.legend()
                        plt.show()	
                    
                    
                    
                        #plt.figure()
                        #plt.hist((data_tof-model_tof)/data_tof_err,arange(-100,100,0.1))
                        #plt.show()					
                    
                    
                if plot_Eproj==True:
                    fig, ax = plt.subplots(1,1,figsize=(figx_half, figy))
                    fig.subplots_adjust(top=adjust_top)
                    #fig = pylab.figure()
                    #ax = fig.gca()
                    ax.plot(y,data_E,linewidth=2,color="royalblue",label=r"$\rm{data}$")
                    ax.plot(y,model_E,linewidth=2,color="darkred",label=r"$\rm{model}$")
                
                    ax.set_xlabel(r"$\rm{ESSD \ [ch]}$",fontsize=fontsize)
                    ax.set_ylabel(r"$\rm{counts \ per \ bin}$",fontsize=fontsize)
                    #ax.legend(loc="upper right",bbox_to_anchor=(1.3, 1.1))
                    ax.set_title(r"$\rm{E/q-step: \ } %i$"%(step),fontsize=fontsize,y=title_lift)
                    #ax.text(.5,.9,r"$\rm{E/q-step: \ } %i$"%(step),horizontalalignment='left',transform=ax.transAxes,fontsize=fontsize)
                    ax.tick_params(axis="x", labelsize=ticklabelsize)
                    ax.tick_params(axis="y", labelsize=ticklabelsize)
                    
                    hmax=max(max(data_E),max(model_E))
                    ax.set_ylim([0,1.3*hmax])
                    ax.legend(prop={'size': legsize})
                    if yrange_plot!=None:
                        ax.set_xlim(yrange_plot[0],yrange_plot[-1]) 
                    
                    
                    if save_figures==True:
                        fig.savefig(path0+figpath+"Eintegrals_d%i-%i_vsw%i-%i_%s_step%i.png"%(timerange[0],timerange[-1],vsw_range[0],vsw_range[-1],peakshapes,step),bbox_inches='tight')
                    else:	
                        plt.show()
                    
                    
                #plot energy slice
                if plot_tofslices!=None:
                    #return plot_tofslices
                    tofslices=arange(plot_tofslices[0],plot_tofslices[-1],2.)
                    Emeans_cal=zeros((len(tofslices)))
                    Emeans_fit=zeros((len(tofslices)))
                    Esigs_left=zeros((len(tofslices)))
                    Esigs_right=zeros((len(tofslices)))
                    Ekappas_left=zeros((len(tofslices)))
                    Ekappas_right=zeros((len(tofslices)))
                    l=0
                    for tofslice in tofslices:
                            
                        #return x,plot_tofslice
                        tofind=where(x==find_nearest(x,tofslice))[0][0]
                        Ecountmin_mask=(H_data[tofind]>=Ecountmin)
                        yE=y[Ecountmin_mask]
                        #return x,plot_tofslice,tofind,H_data
                        #energies=y
                        Edata=H_data[tofind][Ecountmin_mask]
                        #return Edata
                        #Edata_denom=Edata*1.0
                        #Edata_denom=Edata_denom[Edata_denom==0]=1.
                        Eweights=1./sqrt(Edata)
                        Emodel=H_model_fitted[tofind][Ecountmin_mask]
                        #Emodel_func=gauss1d_asym
                            
                    
                        #fitfunc = lambda p,x: I.tofkappa_allions_scalable(x=x,c=sigtofrel_0,heights=p,step=step,kappatof=100)
                        #errfunc = lambda p, hgs, h: (fitfunc(p,hgs)-h)
                    
                        Efitfunc = lambda p,x: Asymkappa(p,x)
                        #Efitfunc = lambda p,x: gauss1d_asym(p,x)
                        Eerrfunc = lambda p, x, h, h_w: (Efitfunc(p,x)-h)*h_w
                    
                        #p0=array([max(Edata),40,5,5])
                        p0=array([max(Edata),40,5,5,1.3,1.3])
                        
                        Eargs = leastsq(Eerrfunc, p0[:], args=(yE,Edata,Eweights),full_output=1)
                        Ep=Eargs[0]
                        #return Ep
                        #return Emodel,Efitfunc(Ep,yE),Edata,Eweights,yE
                    
                        Echi_m=sum(((Emodel-Edata)*Eweights)**2)/float(len(yE)-3)
                        Echi_bm=sum(((Efitfunc(Ep,yE)-Edata)*Eweights)**2)/float(len(yE)-4)
                        Emeans_cal[l]=ESSD(tofch=tofslice,m=56,Z=26)
                        Emeans_fit[l]=Ep[1]
                        Esigs_left[l]=Ep[2]
                        Esigs_right[l]=Ep[3]
                        Ekappas_left[l]=Ep[4]
                        Ekappas_right[l]=Ep[5]
                        
                        if plot_Efits==True:
                            plt.figure()
                            plt.plot(yE,Edata,label="data")
                            plt.plot(yE,Emodel,label="model, chi2=%.2e"%(Echi_m))
                            plt.plot(yE,Efitfunc(Ep,yE),label="best energy shape fit: posE=%.1f ,sigl=%.1f, sigr=%.1f, chi2=%.2e"%(Ep[1],Ep[2],Ep[3],Echi_bm))
                            plt.legend()
                            plt.xlabel("energy [ch]")
                            plt.ylabel("counts")
                            plt.title("E/q-step = %i, tofch_slices=%i ,\nreduced chi^2 = %.1e, reduced chi_Poisson^3 = %.1e"%(step,plot_tofslices[l],Chi2_red,sig2_avg_poisson))
                            plt.show()
                        l=l+1
                    
                
                Ioncounts_model=zeros(len(ionlist))
                Ioncounts_mult_model=zeros(len(ionlist))
                P_model=zeros(len(ionlist))
                P_model_errors=zeros(len(ionlist))
                P_model_errors_rel=zeros(len(ionlist))
                
                
                Ioncounts_model[ionlist_stepfit_inds]=ioncounts
                Ioncounts_mult_model[ionlist_stepfit_inds]=ioncounts_mult
                P_model[ionlist_stepfit_inds]=pres
                P_model_errors[ionlist_stepfit_inds]=p_errors
                P_model_errors_rel[ionlist_stepfit_inds]=p_errors_rel
                Chi2_model=Chi2_red
                #Chi2_P_model=sig2_avg_poisson
                Chi2_tof_model=Chi2_tof_red
                Absres_model=absres
                
                Ioncounts_step[k]=Ioncounts_model	
                Ioncounts_mult_step[k]=Ioncounts_mult_model
                P_step[k]=P_model	
                P_step_errors[k]=P_model_errors	
                P_step_errors_rel[k]=P_model_errors_rel	
                Chi2_step[k]=Chi2_model
                #Chi2_P_step[k]=Chi2_P_model
                Chi2_tof_step[k]=Chi2_tof_model
                Absres_step[k]=Absres_model
                vswfilter.append(vsw_range[-1])
                
                k=k+1	
                
            Ioncounts[j]=Ioncounts_step
            Ioncounts_mult[j]=Ioncounts_mult_step
            P[j]=P_step
            P_errors[j]=P_step_errors
            P_errors_rel[j]=P_step_errors_rel
            Chi2[j]=Chi2_step
            #Chi2_P[j]=Chi2_P_step
            Chi2_tof[j]=Chi2_tof_step
            Absres_rel[j]=Absres_step
            j=j+1
        
            
        
        #save countdata, if desired, has to be adapted to sequence! 
        
        #return Ioncounts_mult
        
        Ioncounts_total_perstep=sum(Ioncounts_mult.transpose(1,0,2)[0],axis=1)
        
        if save_countdata==True:
            #return array([Ioncounts,P,P_errors,Ioncounts_mult]), len(ionlist), len(Ionlist_removed)
            #Countdata=array([Ioncounts,P,P_errors,Ioncounts_mult]).reshape(len(steps),len(ionlist)*3)
            #return Countdata
            #Chidata=array([Chi2,Chi2_P]).T
            Chidata=array([Chi2,Absres_rel]).T
            #return Chidata[0],steps
            
            #return vswfilter
            vswfilter=array(vswfilter)
            
            Chidata_cal=vstack([steps,vswfilter,Ioncounts_total_perstep,Chidata[0].T])
            #return Chidata
            """
            #here is the original fitdata:
            savetxt(path0+figpath+"countdata",Countdata,fmt="%.2e",delimiter=" ",newline="\n")
            savetxt(path0+figpath+"Chidata",Chidata,fmt="%.2e",delimiter=" ",newline="\n")
            """
            savetxt(figpath+chidata_filename,Chidata_cal.T,fmt="%.2e",delimiter=" ",newline="\n")
            
            
        t1=clock()
        print("Total method runtime[s]:",t1-t0)  
        print("fitted points, chi2-evaluated counts:", len(hdata[gridmask_allions]),len(hdata[mincounts_mask]),len(hdata[hfit>0]))
        print("vsw_range:", vsw_range)
        
        #return I,I_stepfit
        #return ionlist_stepfit
        return Ioncounts,P,P_errors,P_errors_rel,Absres_rel,Chi2_tof,Chi2,Chi2_P,Ioncounts_mult,Ionlist_removed,ionlist
        #return optimized


    def analyze_veldist(self,ionlist,Chi,modelnumber,steps,ions_plot, cfracs=[0.61,0.32,0.14],velref=335.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=65, cmult=False,plot_evalrange=False, Xrange=[380,720], Yrange=None,Xrange_log=[210,950],Yrange_log=None, figx=13.9,figy=9,adjust_top=0.57,lgx=-0.028,lgy=1.9,legsize=18, labelsize=20,ticklabelsize=16,vproton=None,figtitle="", savefigure=False,figpath="",figname="test",peakshape="gauss2d",plot_errorbars=False, Nboot=1000,plot_steps=False,Plot=False,PlotEff=True,scale_ions=None,figformat_autom=True,fitgauss=False,vth_fitguess=None,save_meanvels=False,filepath="",filename="Test",save_totalcounts=False,counts_filename="Counts_Test",plot_436=False):	
				
			
            """
            runavg must be an odd number Z>0
            #for the moment it only works for len(cfracs==3)
            #above is standard for 5 ions, 
            for 2 ions: figx=13.7, figy=6.7,  lgy=1.47, adjust_top=0.71
            for 3 ions: figx=13.7, figy=7.8,  lgy=1.63, adjust_top=0.61 
            for 4 ions: figx=13.7, figy=9.0,  lgy=1.85, adjust_top=0.53
            for 5 ions: figx=13.9, figy=9.0,  lgy=2.0,  adjust_top=0.55
            for 7 ions: figx=13.9, figy=11.5, lgy=2.25, adjust_top=0.48

            for element comparison: scale_ions=[["O6+","C5+","Ne8+"],[0.2,0.5,0.5]]
            example:a=d.analyze_veldist(ionlist=chi_530550_035_stop65_Astable[-1],Chi=chi_530550_035_stop65_Astable,modelnumber=0,steps=arange(10,80,1),ions_plot=["O6+","O7+"],Yrange=[1,6101],savefigure=True,vproton=539,figname="O_stop65_LA",stopstep=65,figy=9.,adjust_top=0.57, lgy=1.45)
            """

            if len(shape(scale_ions))>0:
                iext=""
                scext="_scperc"
                for nion,ionname in enumerate(scale_ions[0]): 
                    scext=scext+"_%s-%s"%(ionname[:-1],int(scale_ions[1][nion]*100))#scaling in percent
                fext=iext+scext
            else:
                fext=""

            if figformat_autom==True:
                if len(ions_plot)==2:
                    figx=13.7
                    figy=6.7
                    #lgy=1.47
                    lgy=1.35
                    adjust_top=0.71
                elif len(ions_plot)==3:
                    figx=13.7
                    figy=7.8
                    lgy=1.50
                    adjust_top=0.61 
                elif len(ions_plot)==4:
                    figx=13.7
                    figy=9.   
                    lgy=1.65
                    #adjust_top=0.53
                    adjust_top=0.50
                elif len(ions_plot)==5:
                    figx=13.9
                    figy=9.   
                    lgy=2.0
                    adjust_top=0.55
                elif len(ions_plot)==7:
                    figx=13.9
                    figy=11.5   
                    lgy=2.25
                    adjust_top=0.48

            avgoff=(runavg-1)/2
            #get ion counts and count uncertainties
            if cmult==True:
                    Ioncounts=Chi[-3].transpose(2,1,0)
            else:
                Ioncounts=Chi[0].transpose(2,1,0)
            Ioncounts_model=Ioncounts.transpose(1,0,2)[modelnumber]
            Ioncounts_errors_rel=Chi[3].transpose(2,1,0)
            Ioncounts_errors_rel_model=Ioncounts_errors_rel.transpose(1,0,2)[modelnumber]
            Ioncounts_errors_model=Ioncounts_errors_rel_model*Ioncounts_model

            I=IonDist()
            I.add_ionlist(names=ions_plot,peakshapes=peakshape,intensities=[1.]*len(ions_plot),print_ions=True)

            VelMin=zeros((len(cfracs),len(I.Ions)))
            VelMax=zeros((len(cfracs),len(I.Ions)))
            Velmeans=zeros((len(cfracs),len(I.Ions)))
            Velmean_errors_boot=zeros((len(cfracs),len(I.Ions)))
            Counts_total=zeros((len(cfracs),len(I.Ions)))
            for l,cfrac in enumerate(cfracs):
                #plot velidt analysis
                Vmin_stop=zeros((len(I.Ions)))
                #Vmax_stop=zeros((len(I.Ions)))
                for i,ion in enumerate(I.Ions):
                    m,q=ion.mass,ion.charge
                    
                    vmin_stop=step_to_vel(step=stopstep,q_e=q,m_amu=m)
                    print('vminstop',vmin_stop)
                    Vmin_stop[i]=vmin_stop
                velmin_stop=max(Vmin_stop)#too strict, because the other ions are still not cut by the stepper
                
                #calculate mean speed and standard error of the mean with Gaussian error propagation (only exact for high count rates that follow normal statistics at each eT bin in each fitted step )
                Vels=zeros((len(I.Ions),len(steps)))
                Countscor=zeros((len(I.Ions),len(steps)))
                Countscor_errors=zeros((len(I.Ions),len(steps)))
                Vmin=zeros((len(I.Ions)))
                Vmax=zeros((len(I.Ions)))
                Velmean=zeros((len(I.Ions)))
                Velmean_error=zeros((len(I.Ions)))
                Velmean_error_boot=zeros((len(I.Ions)))
                Count_total=zeros((len(I.Ions)))
                Ions_out=[]
                for i,ion in enumerate(I.Ions):

                    #get ion speeds
                    ion_ind=where(array(ionlist)==ion.name)[0]
                    m,q=ion.mass,ion.charge
                    counts=Ioncounts_model[ion_ind][0]
                    counts_errors=Ioncounts_errors_model[ion_ind][0]
                    vels=step_to_vel(step=steps,q_e=q,m_amu=m)
                    Maxmask=(vels>=MAX_velmin)*(vels<=MAX_velmax)
                
                    #apply phase space correction to the speed distributions 				
                    countscor=counts*(velref/vels)**2	
                    countscor_errors=counts_errors*(velref/vels)**2

                    cmax=max(countscor*Maxmask)
                    indmax=where(countscor==cmax)[0]
                    maxvel=vels[indmax]
                    ml=vels<maxvel
                    mh=vels>maxvel
                    vels_low=vels[ml]
                    countscor_low=countscor[ml]
                    vels_high=vels[mh]
                    countscor_high=countscor[mh]

                    j_id=0
                    j_id+=avgoff
                    while j_id<len(countscor_low):
                        cl=countscor_low[int(j_id-avgoff):int(j_id+avgoff)+1]
                        if average(cl)<cfrac*cmax:
                            vmin=vels_low[int(j_id)]
                            jl=j_id*1
                            j_id=len(countscor_low)
                        else: 
                            j_id+=1
                    j_id=avgoff		
                    while j_id<len(countscor_high):
                        ch=countscor_high[::-1][int(j_id-avgoff):int(j_id+avgoff)+1]
                        if average(ch)<cfrac*cmax:
                            vmax=vels_high[::-1][int(j_id)]
                            jh=j_id*1
                            j_id=len(countscor_high)
                        else: 
                            j_id+=1
                    #return array([[countscor_low,vels_low,cmax,maxvel,jl,cl,average(cl),vmin],[countscor_high[::-1],vels_high[::-1],cmax,maxvel,jh,ch,average(ch),vmax]])
                    
                    #if vmin<velmin_stop:
                    if vmin<Vmin_stop[i]:
                        print("speed range error: vmin<vmin_stop")
                        #Ions_out.append([ion.name,vmin,velmin_stop])
                        Ions_out.append([ion.name,vmin,Vmin_stop[i]])
                    velmask=(vels>=vmin)*(vels<=vmax)
                    velmean=average(vels[velmask],weights=countscor[velmask])	
                    velmean_error=1./sum(countscor[velmask])*sqrt(sum(((vels[velmask]-velmean)*countscor_errors[velmask])**2))
                    Vmin[i]=vmin
                    Vmax[i]=vmax
                    Velmean[i]=velmean
                    Velmean_error[i]=velmean_error
                    Vels[i]=vels
                    Countscor[i]=countscor
                    Countscor_errors[i]=countscor_errors
                    Count_total[i]=sum(countscor[velmask])
                    """
                    #calculate error of the mean speed with bootstrapping
                    cerr_mask=(countscor_errors>0)
                    Cdist=zeros((len(countscor[velmask*cerr_mask]),Nboot))
                    vvels=array([vels[velmask*cerr_mask]]*Nboot).T
                    for k,c in enumerate(countscor[velmask*cerr_mask]):
                        countscor_errors[velmask*cerr_mask]#approx.
                        cdist_fit=random.normal(loc=c,scale=countscor_errors[velmask*cerr_mask][k],size=Nboot)
                        cdist_fit[cdist_fit<0]=0
                        cdist_stats=random.poisson(lam=c,size=Nboot)
                        Cdist[k]=cdist_fit+cdist_stats
                    Cdist[Cdist==   0]=1e-10
                    Cmeans=average(vvels,weights=Cdist,axis=0)
                    velmean_boot,velmean_error_boot=mean(Cmeans),std(Cmeans)
                    #return len(countscor_errors[velmask*cerr_mask]),velmean,velmean_boot,velmean_error_boot
                    Velmean_error_boot[i]=velmean_error_boot
                    """
                
                VelMin[l]=Vmin
                VelMax[l]=Vmax
                Velmeans[l]=Velmean
                Velmean_errors_boot[l]=Velmean_error_boot	
                Counts_total[l]=Count_total


            if Plot==True:
                fitparams=zeros((len(ions_plot),3))
                fig, ax = plt.subplots(1,2,figsize=(figx, figy))
                #ax = fig.add_axes([0.1, 1.0, 0.1, 0.1])
                #ax = fig.add_axes([0.1, 0.1, 1.0, 0.75])
                #fig.subplots_adjust(left=1.1*bbox.width)
                fig.subplots_adjust(top=adjust_top)
                
                Color=["b","r","green","cyan","y","m","k","gray","orange","olive","brown","pink","indigo","teal","palegreen","navy","sandybrown","springgreen","gold","b","r","green","cyan","y","m","k","gray","orange","olive","brown","pink","indigo","teal","palegreen","navy","sandybrown","springgreen","gold"]
                

                ###############################################################################

                #Left Plot:
                element_names=[]	
                for i,ion in enumerate(I.Ions):
                    nt=ion.name[1:2]
                    try: 
                        float(nt)
                        print(float(nt))
                    except ValueError:
                        nt=0
                    if nt==0:
                        element_name=ion.name[0:2]
                    else:
                        element_name=ion.name[0:1]
                    element_names.append(element_name)
                #return element_names
                
                for i,ion in enumerate(I.Ions):
                    q=ion.charge
                    Cmax=amax(Countscor)
                    color=Color[i]
                    vmin=Vmin[i]
                    vmax=Vmax[i]
                    vels=Vels[i]
                    countscor=Countscor[i]
                    countscor_errors=Countscor_errors[i]
                    if len(shape(scale_ions))>0 and ion.name in scale_ions[0]:
                        scale_ions=array(scale_ions)
                        scind=where(scale_ions[0]==ion.name)[0][0]
                        fsc=float(scale_ions[1][scind])
                        #return fsc
                        countscor=fsc*countscor
                        countscor_errors=fsc*countscor_errors
                    #velmean=Velmean[i]
                    #velmean_error=Velmean_error[i]
                    #velmean_error_boot=Velmean_error_boot[i]
                    
                    if plot_errorbars==True:
                        ax[0].errorbar(vels,countscor,yerr=countscor_errors,color=color,marker="o",label=r"$ \rm{%s}^{%i+}: \ \ \langle \rm{v} \rangle_{1\sigma} = \ %i \  \rm{km/s} \pm \ %i \ \rm{km/s}, \ \langle \rm{v} \rangle_{1.5\sigma} = \ %i \  \rm{km/s} \pm \ %i \ \rm{km/s}, \ \langle \rm{v} \rangle_{2\sigma} = \ %i \  \rm{km/s} \pm \ %i \ \rm{km/s}$"%(element_names[i],q,Velmeans[0,i],round(Velmean_errors_boot[0,i]),Velmeans[1,i],round(Velmean_errors_boot[1,i]),Velmeans[2,i],round(Velmean_errors_boot[2,i])))
                    else:
                        ax[0].plot(vels,countscor,color=color,marker="o",label=r"$ \rm{%s}^{%i+}: \ \ \langle \rm{v} \rangle_{1\sigma} = \ %.1f \  \rm{km/s}, \ \langle \rm{v} \rangle_{1.5\sigma} = \ %.1f \  \rm{km/s}, \ \langle \rm{v} \rangle_{2\sigma} = \ %.1f \  \rm{km/s}$"%(element_names[i],q,Velmeans[0,i],Velmeans[1,i],Velmeans[2,i]))
                    if plot_evalrange==True:
                        if i==0:
                            ax[0].plot([VelMin[0,i],VelMin[0,i]],[0.01,2.*Cmax],color=color,linewidth=1.5)
                            ax[0].plot([VelMax[0,i],VelMax[0,i]],[0.01,2.*Cmax],color=color,linewidth=1.5)
                            #print "test1:", VelMin[1,i]
                            ax[0].plot([VelMin[1,i],VelMin[1,i]],[0.01,2.*Cmax],color=color,linewidth=1.5)
                            ax[0].plot([VelMax[1,i],VelMax[1,i]],[0.01,2.*Cmax],color=color,linewidth=1.5)
                            #print "test2:", VelMin[2,i]
                            ax[0].plot([VelMin[2,i],VelMin[2,i]],[0.01,2.*Cmax],color=color,linewidth=1.5)
                            ax[0].plot([VelMax[2,i],VelMax[2,i]],[0.01,2.*Cmax],color=color,linewidth=1.5)
                        if i==1 or i==2:
                        #if i==1:
                            ax[0].plot([VelMin[2,i],VelMin[2,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                            ax[0].plot([VelMax[2,i],VelMax[2,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                        
                        print("fitgauss", fitgauss)
                        if fitgauss==True:
                            xdata=vels[countscor>0]
                            ydata=countscor[countscor>0]				
                            fitfunc = lambda p, x: gauss1d(p,x)
                            #errfunc = lambda p, x, y: (fitfunc(p, x) - y)/sqrt(y) 
                            #p0 = [Cmax,velref,30]
                            errfunc = lambda p, x, y: (fitfunc(p, x) - y) 
                            
                            velmask_evalrange=(xdata>=VelMin[2,i])*(xdata<=VelMax[2,i])
                            xdata_eval=xdata[velmask_evalrange]
                            ydata_eval=ydata[velmask_evalrange]
                            print("xdata_eval, ydata_eval", xdata_eval, ydata_eval, ion.name)
                            indmax=where(ydata_eval==max(ydata_eval))[0][0]
                            p0 = [max(ydata_eval),xdata_eval[indmax],vth_fitguess]
                            print("Gaussian VDF fit start parameters:", p0, ion.name)  
                            args = optimize.leastsq(errfunc, p0[:], args=(xdata, ydata), full_output=1)
                            p1 = args[0]
                            print("Gaussian VDF best fit parameters for %s:"%(ion.name), p1)  
                            text_xpos=0.70
                            if i==0:
                                text_ypos=0.92
                            elif i==1:
                                text_ypos=0.82
                            elif i==2:
                                text_ypos=0.72
                            ax[1].text(text_xpos,text_ypos,r"$\rm{ \langle v}_{fit} \rm{\rangle: \ } %.1f \rm{\ km/s} $"%(p1[1]),horizontalalignment='center',transform=ax[1].transAxes,fontsize=18,color=color)
                            pcov = args[1]
                            #A=p1[0]
                            #B=p1[1]
                            #C=p1[2]
                            ax[0].plot(xdata,gauss1d(p1,xdata),color="k",linewidth=2)			
                            fitparams[i]=p1
                                
                            
                    if vproton!=None:
                        ax[0].plot([vproton,vproton],[0.01,2.*Cmax],linewidth=2.0,color="k")
                    if plot_436==True:
                        ax[0].plot([436,436],[0.01,2.*Cmax],linewidth=1.5,linestyle="--",color="k")
                ax[0].legend(prop={'size': legsize})
                if lgx!=None and lgy!=None:
                    ax[0].legend(loc="upper left",prop={'size': legsize},bbox_to_anchor=(lgx, lgy),ncol=1)
                ax[0].set_xlabel(r"$ \rm{ion \ speed \ [km/s]}$",fontsize=labelsize)
                ax[0].set_ylabel(r"$ \rm{phase \ space \ density}$" "\n" r"$\rm{[arb. \ units]}$",fontsize=labelsize)
                ax[0].set_title(figtitle)
                minor_xticks=arange(Xrange[0],Xrange[-1]+1,10)
                ax[0].set_xticks(minor_xticks, minor=True)
                
                
                ax[0].tick_params(axis="x", labelsize=ticklabelsize)
                ax[0].tick_params(axis="y", labelsize=ticklabelsize)
                ax[0].grid(which='both')

                """
                major_xticks=MXT
            minor_xticks=mxt
            if Alfven==True:	
            major_yticks = arange(Yrange[0], Yrange[-1], 0.2)                                              
            minor_yticks = arange(Yrange[0], Yrange[-1], 0.1)                                               
            else:	 
            #major_yticks = arange(-60, Yrange[-1], 20)                                              
            #minor_yticks = arange(Yrange[0], Yrange[-1], 10)                                               
            major_yticks=MYT
            minor_yticks=myt

            ax.set_xticks(major_xticks)                                                       
            #ax.set_xticks(minor_xticks, minor=True)                                           
            ax.set_yticks(major_yticks)                                                       
            ax.set_yticks(minor_yticks, minor=True) 
            ax.grid(which='both') 
                """
                
                if Xrange!=None:
                    ax[0].set_xlim(Xrange[0],Xrange[-1])
                
                if Yrange==None:
                    ax[0].set_ylim(0.01,1.1*Cmax)
                else:
                    ax[0].set_ylim(Yrange[0],Yrange[-1])
                




                #####################################################################
                #Right Plot:
                    
                for i,ion in enumerate(I.Ions):
                    Cmax=amax(Countscor)
                    color=Color[i]
                
                    velmean=Velmean[i]
                    vmin=Vmin[i]
                    vmax=Vmax[i]
                    velmean_error=Velmean_error[i]
                    velmean_error_boot=Velmean_error_boot[i]
                    vels=Vels[i]
                    countscor=Countscor[i]
                    countscor_errors=Countscor_errors[i]
                    
                    if plot_errorbars==True:
                        ax[1].errorbar(vels,countscor,yerr=countscor_errors,color=color,marker="o")
                    
                    #r"$\rm{v_{Fe^{10+}}[km/s]}$",fontsize=18, color='black'
                    
                    else:
                        ax[1].plot(vels,countscor,color=color,marker="o")
                    
                    if vproton!=None:
                        ax[1].plot([vproton,vproton],[0.01,10.*Cmax],linewidth=2.0,color="k")
                    if plot_436==True:
                        ax[1].plot([436,436],[0.01,10.*Cmax],linewidth=2.0,linestyle="--",color="k")
                    
                    #if vproton!=None:
                    #    ax[1].plot([vproton,vproton],[0.01,10.*Cmax],color="k")
                    if plot_evalrange==True:
                        if i==0:
                            ax[1].plot([VelMin[0,i],VelMin[0,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                            ax[1].plot([VelMax[0,i],VelMax[0,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                            #print "test1:", VelMin[1,i]
                            ax[1].plot([VelMin[1,i],VelMin[1,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                            ax[1].plot([VelMax[1,i],VelMax[1,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                            #print "test2:", VelMin[2,i]
                            ax[1].plot([VelMin[2,i],VelMin[2,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                            ax[1].plot([VelMax[2,i],VelMax[2,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                        
                        if i==1 or i==2:
                            ax[1].plot([VelMin[2,i],VelMin[2,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                            ax[1].plot([VelMax[2,i],VelMax[2,i]],[0.01,10.*Cmax],color=color,linewidth=1.5)
                        
                        
                        if fitgauss==True:
                            xdata=vels[countscor>0]
                            ydata=countscor[countscor>0]
                            if len(shape(scale_ions))>0:
                                if i<len(scale_ions[0]):
                                    ysc=float(scale_ions[1,i])				
                                    ax[1].plot(xdata,gauss1d(fitparams[i],xdata)/ysc,color="k",linewidth=1.5)			
                                else:
                                    ax[1].plot(xdata,gauss1d(fitparams[i],xdata),color="k",linewidth=1.5)			
                            else:
                                ax[1].plot(xdata,gauss1d(fitparams[i],xdata),color="k",linewidth=1.5)			
                            
                ax[1].legend()
                ax[1].set_yscale("log")
                if lgx!=None and lgy!=None:
                    ax[1].legend(loc="upper left",bbox_to_anchor=(lgx, lgy))
                ax[1].set_xlabel(r"$ \rm{ion \ speed \ [km/s]}$",fontsize=labelsize)
                #ax[1].set_ylabel(r"$ \rm{counts \ cor.}$",fontsize=labelsize)
                ax[1].set_title(figtitle)
                minor_xticks=arange(Xrange[0],Xrange[-1]+1,10)
                ax[1].set_xticks(minor_xticks, minor=True)
                
                
                ax[1].tick_params(axis="x", labelsize=ticklabelsize)
                ax[1].tick_params(axis="y", labelsize=ticklabelsize)
                ax[1].grid(which='major')

                if Xrange_log!=None:
                    ax[1].set_xlim(Xrange_log[0],Xrange_log[-1])
                
                if Yrange_log==None:
                    ax[1].set_ylim(0.1,10.*Cmax)
                else:
                    ax[1].set_ylim(Yrange_log[0],Yrange_log[-1])
                
                if savefigure==True:
                    #plt.savefig(figpath+figname+fext+".png",bbox_inches='tight')
                    plt.savefig(filepath+filename+".png",bbox_inches='tight')
                plt.show()
                calc_fracs=array(cfracs)
                if save_meanvels==True:
                    data_out=vstack([calc_fracs,Velmeans.T]).T
                    outfile_pathfile=filepath+filename
                    speed_header=""
                    for ionname in ions_plot:
                        ion_header=", vmean_%s [km/s]"%(ionname)
                        speed_header=speed_header+ion_header
                    header="TODO: detailed description of the data\n speed interval [rel. count threshold of VDF maximum]"+speed_header
                    savetxt(outfile_pathfile, data_out, fmt='%.2f', delimiter=' ', newline='\n',header=header)
                    
                if save_totalcounts==True:
                    #counts_filename="ioncounts_total"
                    counts_data_out=vstack([calc_fracs,Counts_total.T]).T
                    counts_outfile_pathfile=filepath+counts_filename
                    counts_header="cfracs "
                    for ionname in ions_plot:
                        counts_header=counts_header+"counts_%s "%(ionname)
                    savetxt(counts_outfile_pathfile, counts_data_out, fmt='%.2f', delimiter=' ', newline='\n',header=counts_header)


########################################  EFFICIENCY

            if PlotEff==True:
                with open("./Data/DCeffAellig.csv") as file_name:
                    DCeffAellig = loadtxt(file_name, delimiter=",")
                f_eff = interpolate.interp2d([4, 16, 40, 56], arange(1, 21, 1), DCeffAellig, kind='linear')
                SSDeff_file = pd.read_csv("./Data/effKoeten.csv")  
                #fig, ax = plt.subplots(1,2,figsize=(figx, figy))
                fig, ax = plt.subplots()
                element_names=[]
                element_charges=[]

                abundances = zeros(len(I.Ions))
                
                for i,ion in enumerate(I.Ions):
                    name=ion.name
                    element_names.append(name)
                    print(name)
                    m=ion.mass
                    q=ion.charge
                    element_charges.append(q)
                    Cmax=amax(Countscor)
                    color=Color[i]
                    velmean=Velmean[i]
                    vmin=Vmin[i]
                    vmax=Vmax[i]
                    vels=Vels[i]
                    countscor=Countscor[i]
                    countscor_errors=Countscor_errors[i]
                    totcountseff=0
                    for v,vel in enumerate(vels):
                        Eoq=10**(-3)*m*constants.proton_mass/(2*q*constants.elementary_charge)*(vel*10**3)**2 #in KeV/e
                        Etot = (Eoq+25)*q # in KeV
                        Eamu = Etot/m  # in Kev/amu
                        #print('Eoq, Etot, Eamu:',Eoq, Etot, Eamu)
                        FT = interpolate.interp1d(SSDeff_file['eoq'], SSDeff_file[name+'T'])
                        FAS = interpolate.interp1d(SSDeff_file['eoq'], SSDeff_file[name+'AS'])
                        FS = interpolate.interp1d(SSDeff_file['eoq'], SSDeff_file[name+'S'])
                        TOFeff = f_eff(m, Eamu)[0]
                        SSDeff = FT(Eoq)*FAS(Eoq)/FS(Eoq)*10**(-2)
                        countscor[v]/=(TOFeff*SSDeff)
                        if VelMin[2,i] <= vel <= VelMax[2,i]:
                            totcountseff+=countscor[v]
                    abundances[i]=totcountseff
                    '''ax[0].plot(vels,countscor,color=color,marker="o")
                    if fitgauss==True:
                        xdata=vels[countscor>0]
                        ydata=countscor[countscor>0]				
                        fitfunc = lambda p, x: gauss1d(p,x)
                        #errfunc = lambda p, x, y: (fitfunc(p, x) - y)/sqrt(y) 
                        #p0 = [Cmax,velref,30]
                        errfunc = lambda p, x, y: (fitfunc(p, x) - y) 
                        
                        velmask_evalrange=(xdata>=VelMin[2,i])*(xdata<=VelMax[2,i])
                        xdata_eval=xdata[velmask_evalrange]
                        ydata_eval=ydata[velmask_evalrange]
                        #print("xdata_eval, ydata_eval", xdata_eval, ydata_eval, ion.name)
                        indmax=where(ydata_eval==max(ydata_eval))[0][0]
                        p0 = [max(ydata_eval),xdata_eval[indmax],vth_fitguess] 
                        args = optimize.leastsq(errfunc, p0[:], args=(xdata, ydata), full_output=1)
                        p1 = args[0]
                        print("Gaussian VDF best fit parameters for %s:"%(ion.name), p1)  
                        text_xpos=0.70
                        if i==0:
                            text_ypos=0.92
                        elif i==1:
                            text_ypos=0.82
                        elif i==2:
                            text_ypos=0.72
                        pcov = args[1]
                        #A=p1[0]
                        #B=p1[1]
                        #C=p1[2]
                        ax[0].text(text_xpos,text_ypos,r"$\rm{ \langle v}_{fit} \rm{\rangle: \ } %.1f \rm{\ km/s} $"%(p1[1]),horizontalalignment='center',transform=ax[0].transAxes,fontsize=18,color=color)
                        ax[0].plot(xdata,gauss1d(p1,xdata),color="k",linewidth=2)			
                        fitparams[i]=p1
                    #ax[0].legend(prop={'size': legsize})
                    if lgx!=None and lgy!=None:
                        ax[0].legend(loc="upper left",prop={'size': legsize},bbox_to_anchor=(lgx, lgy),ncol=1)
                    ax[0].set_yscale('log')
                    ax[0].set_xlabel(r"$ \rm{ion \ speed \ [km/s]}$",fontsize=labelsize)
                    ax[0].set_ylabel(r"$ \rm{phase \ space \ density}$" "\n" r"$\rm{[arb. \ units]}$",fontsize=labelsize)
                    ax[0].set_title(figtitle)'''
                abundances = [float(i)/sum(abundances) for i in abundances]
                #ax.scatter(arange(1,len(element_names)+1,1), abundances, s=50)
                ax.bar(arange(1,len(element_names)+1,1), abundances)
                #for i, label in enumerate(element_names):
                #    plt.annotate(label, (arange(1,len(element_names)+1,1)[i], abundances[i]))
                #ax.set_xticks(np.add(element_charges,(0.8/2))) # set the position of the x ticks
                ax.set_xticks(arange(1,len(element_names)+1,1))
                labels=(elementn[:-1] for elementn in element_names)
                ax.set_xticklabels(labels)
                #ax.tick_params(axis='y', which='minor')
                #ax[1].legend()
                if lgx!=None and lgy!=None:
                    ax.legend(loc="upper left",bbox_to_anchor=(lgx, lgy))
                #ax.set_yscale('log')
                #ax.grid(which='both', axis='y')
                ax.set_title('Slow wind abundances')
                ax.set_xlabel(r"$ \rm{Ion \ species}$",fontsize=labelsize)
                ax.set_ylabel(r"$ \rm{abundance \ at \ v_p \ 335 \ [km/s]}$" "\n" r"$\rm{[arb. \ units]}$",fontsize=labelsize)
                plt.show()
                elems=[]
                elemabundances=[]
                ssabundance=[]
                FIPS = pd.read_csv("./Data/FIP.csv")
                plotfips=[]
                for i,ion in enumerate(I.Ions):  
                    name = ion.name
                    print(name)
                    elem=''
                    for letter in name:
                        print(letter)
                        if not letter.isdigit() and letter!='+':
                            elem+=letter
                    if not elem in elems and i!=0:
                        print(elem)
                        elems.append(elem)
                        elemabundances.append(elemabundance)
                        plotfips.append(FIPS.iloc[ion.atomnumber-1,1])
                        elemabundance=abundances[i]
                        ssabundance.append(10**(FIPS.iloc[ion.atomnumber-1,2]-12))
                    elif not elem in elems and i==0:
                        print(elem)
                        elems.append(elem)
                        plotfips.append(FIPS.iloc[ion.atomnumber-1,1])
                        elemabundance=abundances[i]
                        ssabundance.append(10**(FIPS.iloc[ion.atomnumber-1,2]-12))
                    else:
                        elemabundance+=abundances[i]
                elemabundances.append(elemabundance)
                print(elemabundances)
                print('ssabundance',ssabundance)
                elemabundances=[(elemabundances[i] / elemabundances[2]) / (ssabundance[i]/ssabundance[2]) for i in range(len(elemabundances))]
                print(elemabundances)
                fig, ax = plt.subplots()
                ax.scatter(plotfips, elemabundances)
                for i, txt in enumerate(elems):
                    ax.annotate(txt, (plotfips[i]+.25, elemabundances[i]), fontsize=12)
                ax.set_yscale('log')
                ax.set_title('Average element abundance ratios dependency on FIP for slow wind')
                ax.set_xlabel(r"$ \rm{First \ ionization \ potential \ (FIP) \ [V]}$",fontsize=labelsize)
                ax.set_ylabel(r"$ \rm{Relative \ abundance \ to \ oxygen}$" "\n" r"$\rm{[arb. \ units]}$",fontsize=labelsize)
                plt.show()
                    
                    
                print('abundances',abundances)
            return ions_plot,Velmeans
            #return ions_plot,Velmean,Velmean_error,Velmean_error_boot, Vmin,Vmax,velmin_stop,fitparams,Ions_out 
    