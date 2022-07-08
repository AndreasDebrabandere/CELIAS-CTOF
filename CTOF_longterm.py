#imports


#numpy
from numpy import *

#scipy
from scipy import optimize, interpolate, integrate, stats
from scipy.special import gamma,gammainc, erf, binom
from scipy.optimize import leastsq
from scipy.optimize import fmin_bfgs as minimizer
from scipy.optimize import minimize 

#matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, cbook
import matplotlib.colors as colors
from matplotlib.colors import Normalize, LogNorm
from matplotlib.path import Path
#from matplotlib.nxutils import points_inside_poly
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import AnchoredText

#pandas
import pandas as pd

#classes, functions from other moduls/scripts
from CTOF_ionlists import L_Arnaud_Min_noHe, L_Arnaud_stable_noHe
#from CTOF_paramfit_reconstruct import ctof_paramfit, iondict
from CTOF_ion import iondict
from CTOF_datafit import ctof_paramfit

import pickle
#from CTOF_dataprocess import ctof_paramfit


#others
from time import perf_counter as clock 
import datetime as dt

Element_labeldict={12:"C",14:"N",16:"O",20:"Ne",24:"Mg",28:"Si",32:"S",40:"Ca",56:"Fe",59:"Ni"}

#Element_colordict={12:"k",14:"lime",16:"r",20:"orange",24:"c",28:"m",32:"b",40:"gray",56:"brown",59:"darkviolet"}#for AGU
Element_colordict={12:"b",14:"olive",16:"r",20:"lime",24:"c",28:"orange",32:"green",40:"gray",56:"brown",59:"darkviolet"}

L_Arnaud_fullstable_noHe=['C4+','C5+','C6+',"N4+","N5+","N6+","N7+","O5+","O6+","O7+","O8+","Ne5+","Ne6+","Ne7+","Ne8+","Ne9+",'Mg4+','Mg5+','Mg6+','Mg7+','Mg8+','Mg9+','Mg10+',"Si5+","Si6+","Si7+","Si8+","Si9+","Si10+","Si11+","Si12+","S6+","S7+","S8+","S9+","S10+","S11+","S12+","S13+","Ca10+","Ca11+","Fe5+","Fe6+","Fe7+","Fe8+","Fe9+","Fe10+","Fe11+","Fe12+","Fe13+","Fe14+","Fe15+","Fe16+","Ni8+","Ni9+","Ni10+"]

#iondict_color={'C4+':"k",'C5+':"k",'C6+':"k","N4+":"gray","N5+":"gray","N6+":"gray","N7+":"gray","O5+":"r","O6+":"r","O7+":"r","O8+":"r","Ne5+":"yellow","Ne6+":"yellow","Ne7+":"yellow","Ne8+":"yellow","Ne9+":"yellow",'Mg4+':"m",'Mg5+':"m",'Mg6+':"m",'Mg7+':"m",'Mg8+':"m",'Mg9+':"m",'Mg10+':"m","Si5+":"c","Si6+":"c","Si7+":"c","Si8+":"c","Si9+":"c","Si10+":"c","Si11+":"c","Si12+":"c","S6+":"orange","S7+":"orange","S8+":"orange","S9+":"orange","S10+":"orange","S11+":"orange","S12+":"orange","S13+":"orange","Ca6+":"pink","Ca7+":"pink","Ca8+":"pink","Ca9+":"pink","Ca10+":"pink","Ca11+":"pink","Ca12+":"pink","Ca13+":"pink","Fe5+":"brown","Fe6+":"brown","Fe7+":"brown","Fe8+":"brown","Fe9+":"brown","Fe10+":"brown","Fe11+":"brown","Fe12+":"brown","Fe13+":"brown","Fe14+":"brown","Fe15+":"brown","Fe16+":"brown","Ni8+":"green","Ni9+":"green","Ni10+":"green"}




#"He+":[4,1,2]

def linfunc(p,x):
    return p[0]*x+p[1]

def constfunc(p,x):
    return p[0]*ones((len(x)))

def plot_chi2_modelevaluation(chi2,tailrange_parameters,steps,CBlog=False,fontsize=18,ticklabelsize=14,save_figure=False,figpath="",filename="test_modelevaluation"):
    """
    chi2 is a 2D landscape of dimensions (N_steps X N_tail_paramaters)
    """
    #pcolor(tailrange_parameters,steps,chi2)
    trp=tailrange_parameters

    figx=13
    figy=6
    fig, ax = plt.subplots(1,1,figsize=(figx, figy))
    #fig = plt.figure()
    #ax = fig.gca()
    chi2_valid=chi2[chi2>0]
    if CBlog==True:
        Cont=ax.pcolor(steps,trp,chi2.T,norm=colors.LogNorm(vmin=min(ravel(chi2_valid)), vmax=max(ravel(chi2_valid))))
    else:	
        Cont=ax.pcolor(steps,trp,chi2.T,vmin=min(ravel(chi2_valid)), vmax=max(ravel(chi2_valid)))
    Cont.cmap.set_under('w')
    cb = fig.colorbar(Cont)
    #cb.set_label(r"Chi2",fontsize=fontsize)
    cb.set_label(r"$ \rm{\chi^2}_{\rm{red}}$",fontsize=fontsize)
    for ctick in cb.ax.get_yticklabels():
        ctick.set_fontsize(ticklabelsize)
    ax.set_ylabel(r"$\rm{A}_C \ \rm{[ch]}^{-1}$",fontsize=fontsize)
    ax.set_xlabel("Epq-step",fontsize=fontsize)
    
    ax.tick_params(axis="x", labelsize=ticklabelsize)
    ax.tick_params(axis="y", labelsize=ticklabelsize)
                

    plt.show()	
    #return max(ravel(chi2_valid))



    ygrid,xgrid=meshgrid(steps,trp)	
    xg,yg=ravel(xgrid),ravel(ygrid)
    i=0
    for chi2_line in chi2:
        step=steps[i]
        try:
            chi2_lmin=min(chi2_line[chi2_line>0])
            chi2_lmin_index=where(chi2_line==chi2_lmin)[0][0]
            #print chi2_lmin, chi2_lmin_index
            trpmin=trp[chi2_lmin_index]
            print(step, trpmin)
            ax.plot([step],[trpmin],marker="o",color="r")
        except ValueError:
            pass
        i+=1
    #ax.legend(loc="upper left")

    
    if save_figure==True:
        plt.savefig(figpath+filename+".png",bbox_inches='tight')
  
  
def estimate_lowerspeedlimit_Epqfilter(vsw_range=[500,510],N_sigma=2):  
  """
  returns minimum speed that has to be reached in the Epq stop-step of the CTOF electrostatic analyzer so that the cycle is valid (and can be included in the long-term analysis ET-matrix to be fitted.)
  The ion speed  v(Epq) that should be compared with is v_O7+ as this is the ion with the lowest mpq of all analyzed ion species (and therefore is measured at the highest speed for each Epq-step). 
  Recommended to take even one Epq-step more than what comes out from the comparison condition: v(epq_step)<vmin_epq (!). Therefore the Epq-stopstep for vsw in [500,510] is 68 (not 67). 
  """
  vsw=mean(vsw_range)
  vth_min=10.#approximate ion thermal speed at ion speeds of about 335 km/s
  vth_max=40.#approximate ion thermal speed  at ion speeds of about 505 km/s
  v_high=525.
  v_low=335.
  
  vsw_low=335.
  vsw_high=505.
  diffspeed_low=0
  diffspeed_high=20
  diffspeed_estim=(diffspeed_high-diffspeed_low)/(vsw_high-vsw_low)*(vsw-vsw_low)
  print(diffspeed_estim)

  vmin_epq=vsw+diffspeed_estim-N_sigma*(vth_min+(vth_max-vth_min)/(v_high-v_low)*(vsw-v_low))
  return vmin_epq  


#Main
class CTOF(object):
    
    def __init__(self,dayrange=[174,202]):
        self.data=self.load_data(dayrange=dayrange)
        
    
    def load_data(self,dayrange=[174,202]):
        data=ctof_paramfit(timeframe=[dayrange],minute_frame=[0,1440],load_processed_PHAdata=True,prepare_mrdata=False)
        #data=ctof_paramfit(timeframe=[dayrange],minute_frame=[0,1440],load_processed_PHAdata=False,prepare_mrdata=False)
        
        return data

    def run_fit_modelevaluation(self,steps,ionlist,peakshapes="kappa_moyalpar_Easym", tailranges=[[0.00325,0.]],Minimization_algorithm="Powell",Termtol=1e-4,Plot_2dcontour=False):
        
        fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps,timerange=[174,202],tofrange=[202,611],Erange=[1,121],PR_range=[1,4],vsw_range=[0,1000],vproton_RCfilter=True,peakshapes=peakshapes,tailranges=tailranges,Minimization="Normal",Poisson_Minapprox=True,Minimization_algorithm=Minimization_algorithm,Termtol=Termtol,include_countweights=True,fitmincounts=10,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=0,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=10., Plot_tofproj=True,plot_tof_errbars=False, Plot_tofproj_log=False, plot_Eproj=False, save_figures=False, save_countdata=False, chidata_filename="chidata_test")
        return fitresult_data

    def run_fit_modelevaluation_showexample(self,ionlist,steps,peakshapes="kappa_moyalpar_Easym", tailranges=[[0.00325,0.]],Minimization_algorithm="Powell",Termtol=1e-4,Plot_2dcontour=False,save_figures=False,plot_rectangle=[[300,350],[20,50]]):
        
        fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps,timerange=[174,202],tofrange=[202,600],Erange=[5,105],PR_range=[1,4],vsw_range=[0,1000],vproton_RCfilter=True,peakshapes=peakshapes,tailranges=tailranges,Minimization="Normal",Poisson_Minapprox=True,Minimization_algorithm=Minimization_algorithm,Termtol=Termtol,include_countweights=True,fitmincounts=10,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=0,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals="chi_relative", absmax=10., Plot_tofproj=True,plot_tof_errbars=False, Plot_tofproj_log=True,plot_toflegend=True, plot_Eproj=False, save_figures=save_figures, save_countdata=False, chidata_filename="chidata_test",figy_size=5.3,plot_rectangle=plot_rectangle)
        return fitresult_data


    def run_fit_modelevaluation_showexample_ironexcerpt(self,ionlist,steps,peakshapes="kappa_moyalpar_Easym", tailranges=[[0.00325,0.]],Minimization_algorithm="Powell",Termtol=1e-4,Plot_2dcontour=False,save_figures=False):
        
        fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps,timerange=[174,202],tofrange=[335,560],Erange=[10,77],PR_range=[1,4],vsw_range=[0,1000],vproton_RCfilter=True,peakshapes=peakshapes,tailranges=tailranges,Minimization="Normal",Poisson_Minapprox=True,Minimization_algorithm=Minimization_algorithm,Termtol=Termtol,include_countweights=True,fitmincounts=10,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=0,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals="chi_relative", absmax=10., Plot_tofproj=True,plot_tof_errbars=False, Plot_tofproj_log=True,plot_toflegend=True, plot_Eproj=False, save_figures=save_figures, save_countdata=False, chidata_filename="chidata_test",figy_size=5.3)
        return fitresult_data




    def run_fit_longtermspectra_showexample(self,ionlist, vswfilter=[500,510],peakshapes="kappa_moyalpar_Easym", tailranges=[[0.00325,0.]], Minimization_algorithm="Powell", Termtol=1e-4, Plot_2dcontour=True, Plot_residuals="relative", absmax=1.,Plot_tofproj=True, steps=arange(35,95,1),plot_elementhypes=False, elementhyp_markersize=10):
        if vswfilter==[330,340]:
            #steps=arange(35,95,1)
            Epq_stopstep_min=90
        
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=Plot_residuals, absmax=1.,Plot_tofproj=Plot_tofproj,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test", plot_datacontour="minimum_countourset",plot_modelcontour=True,model_contourcolor="m",plot_chi2_label=False, plot_elementhypes=plot_elementhypes,elementhyp_markersize=elementhyp_markersize, plot_toflegend=True, ionpos_markercolor="w", ionpos_markeredgecolor="k", plot_vswfilter=True,plot_model_title=False, plot_2dcontour_xlabel=False)
        
        if vswfilter==[500,510]:
            #steps=arange(35,95,1)
            Epq_stopstep_min=68
        
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=Plot_residuals, absmax=1.,Plot_tofproj=Plot_tofproj,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test", plot_datacontour="minimum_countourset",plot_modelcontour=True,model_contourcolor="m",plot_chi2_label=False, plot_elementhypes=plot_elementhypes,elementhyp_markersize=elementhyp_markersize, plot_toflegend=True, ionpos_markercolor="m", ionpos_markeredgecolor="k", plot_vswfilter=True,plot_model_title=False, plot_2dcontour_xlabel=False)
        
        
        return fitresult_data


    def run_fit_longtermspectra(self,ionlist, vswfilter=[500,510],peakshapes="kappa_moyalpar_Easym", tailranges=[[0.00325,0.]], Minimization_algorithm="Powell",Termtol=1e-4,Plot_2dcontour=False,steps=arange(35,95,1)):
        """
        TODO: Change "tailranges!", Check docstring!
        Parameters:\n
        vswfilter must be vswfilter=[330,340] or vswfilter=[500,510]
        """
        
        #slow
        if vswfilter==[320,330]:
            #steps=arange(28,97,1)
            Epq_stopstep_min=90#this is where the "second cycle peak" starts
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[330,340]:
            #steps=arange(35,95,1)
            Epq_stopstep_min=90#this is where the "second cycle peak" starts
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[340,350]:
            #steps=arange(27,95,1)
            Epq_stopstep_min=87
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[350,360]:
            #steps=arange(24,93,1)
            Epq_stopstep_min=83
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[360,370]:
            #steps=arange(19,93,1)
            Epq_stopstep_min=82
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[370,380]:
            #steps=arange(19,91,1)
            Epq_stopstep_min=81
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        
        #intermediate:
        elif vswfilter==[380,390]:
            Epq_stopstep_min=79
            #steps=arange(19,89,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[390,400]:
            Epq_stopstep_min=78
            #steps=arange(25,87,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[400,410]:
            Epq_stopstep_min=77
            #steps=arange(22,86,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[410,420]:
            Epq_stopstep_min=76
            #steps=arange(21,85,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[420,430]:
            Epq_stopstep_min=75
            #steps=arange(19,86,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[430,440]:
            Epq_stopstep_min=74
            #steps=arange(20,84,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[440,450]:
            Epq_stopstep_min=73
            #steps=arange(24,84,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[450,460]:
            Epq_stopstep_min=72
            #steps=arange(21,81,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)

        elif vswfilter==[460,470]:
            Epq_stopstep_min=71
            #steps=arange(20,81,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        #fast
        elif vswfilter==[470,480]:
            Epq_stopstep_min=71
            #steps=arange(20,77,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[480,490]:
            Epq_stopstep_min=70
            #steps=arange(16,76,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[490,500]:
            Epq_stopstep_min=69
            #steps=arange(16,76,1)
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[500,510]:
            #steps=arange(20,72,1)
            steps=arange(18,75,1)
            Epq_stopstep_min=68#regular!
            #Epq_stopstep_min=0#for check of filter influence!
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[510,520]:
            #steps=arange(18,71,1)
            Epq_stopstep_min=67
            #Epq_stopstep_min=0#for check of filter influence!
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)

        elif vswfilter==[520,530]:
            #steps=arange(18,72,1)
            Epq_stopstep_min=66
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[530,540]:
            #steps=arange(17,71,1)
            Epq_stopstep_min=65
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[540,550]:
            #steps=arange(18,71,1)
            Epq_stopstep_min=64
            #Epq_stopstep_min=0#for check of filter influence!
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        elif vswfilter==[550,560]:
            #steps=arange(15,60,1)
            #Epq_stopstep_min=64#yet there is no data left under this filter-condition
            Epq_stopstep_min=0#for check of filter influence!
            fitresult_data=self.data.estimate_gk_peakpars_2d_sequence_fast(ionlist,steps, timerange=[174,202], tofrange=[202,611],Erange=[1,121], PR_range=[1,4],vsw_range=vswfilter,vproton_RCfilter=False, peakshapes=peakshapes,tailranges=tailranges,Minimization="Poisson",Poisson_Minapprox=True,Minimization_algorithm="Powell",Termtol=Termtol,include_countweights=False,fitmincounts=0,baserate_correction=True,br_scaling=True,PHAmin=500,Epq_stopstep_min=Epq_stopstep_min,check_fitarea=False,Plot_start_model=False,Plot_2dcontour=Plot_2dcontour,CBlog=True, Plot_residuals=False, absmax=1.,Plot_tofproj=False,plot_tof_errbars=False,Plot_tofproj_log=False, plot_Eproj=False,save_figures=False, save_countdata=False, chidata_filename="chidata_test",plot_datacontour=False)
        
        else:
            print("no valid vswfilter selected (only vswfilter=[330,340] and vswfilter=[330,340] are currently implemented)") 
            fitresult_data=None 
                
        return fitresult_data

            

    def analyze_longterm_veldist(self,fitresult_data,ions_plot=["C4+","O6+","Si7+","Si8+","Si9+"],peakmodel="kappa_moyalpar_Easym",vswfilter=[500,510],Yrange=None,save_meanvels=False,save_totalcounts=False,filepath="",filename="Test",counts_filename="Test_counts",savefigure=False,plot_steps=False,cmult=True,scale_ions=False,plot_436=False,labelsize=20,legsize=18,ticklabelsize=16):

        #slow
        if vswfilter==[320,330]:
            Steps=arange(28,97,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=325.,runavg=3,MAX_velmin=200, MAX_velmax=450,stopstep=90, cmult=cmult,plot_evalrange=True, Xrange=[240,430],Yrange=Yrange,Xrange_log=[110,750],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=15,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        if vswfilter==[330,340]:
            Steps=arange(35,95,1)#check why 35! and stopstep 87!
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=335.,runavg=3,MAX_velmin=200, MAX_velmax=470,stopstep=95, cmult=cmult,plot_evalrange=True, Xrange=[270,400],Yrange=Yrange,Xrange_log=[240,560],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=0.1,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=15,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions,save_totalcounts=save_totalcounts,counts_filename=counts_filename)

        if vswfilter==[340,350]:
            Steps=arange(27,95,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=345.,runavg=3,MAX_velmin=200, MAX_velmax=480,stopstep=87, cmult=cmult,plot_evalrange=True, Xrange=[240,430],Yrange=Yrange,Xrange_log=[110,750],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=15,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions,save_totalcounts=save_totalcounts,counts_filename=counts_filename)

        if vswfilter==[350,360]:

            Steps=arange(24,93,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=355.,runavg=3,MAX_velmin=200, MAX_velmax=480,stopstep=82, cmult=cmult,plot_evalrange=True, Xrange=[240,490],Yrange=Yrange,Xrange_log=[110,750],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=15,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)

        if vswfilter==[360,370]:
            Steps=arange(19,93,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=365.,runavg=3,MAX_velmin=200, MAX_velmax=480,stopstep=83, cmult=cmult,plot_evalrange=True, Xrange=[240,470],Yrange=Yrange,Xrange_log=[110,750],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=15,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)

        if vswfilter==[370,380]:
            Steps=arange(19,91,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=375.,runavg=3,MAX_velmin=200, MAX_velmax=490,stopstep=81, cmult=cmult,plot_evalrange=True, Xrange=[240,490],Yrange=Yrange,Xrange_log=[110,750],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=15,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)

        #intermediate
        if vswfilter==[380,390]:
            Steps=arange(19,89,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=385.,runavg=4,MAX_velmin=270, MAX_velmax=500,stopstep=79, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        if vswfilter==[390,400]:
            Steps=arange(25,87,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=395.,runavg=4,MAX_velmin=270, MAX_velmax=520,stopstep=78, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        if vswfilter==[400,410]:
            Steps=arange(22,86,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=405.,runavg=4,MAX_velmin=270, MAX_velmax=540,stopstep=77, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        if vswfilter==[410,420]:
            Steps=arange(21,85,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=415.,runavg=4,MAX_velmin=280, MAX_velmax=550,stopstep=76, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        if vswfilter==[420,430]:
            Steps=arange(19,86,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=425.,runavg=4,MAX_velmin=290, MAX_velmax=560,stopstep=75, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        if vswfilter==[430,440]:
            Steps=arange(20,84,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=435.,runavg=4,MAX_velmin=300, MAX_velmax=570,stopstep=74, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)

        if vswfilter==[440,450]:
            Steps=arange(24,84,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=445.,runavg=4,MAX_velmin=300, MAX_velmax=590,stopstep=73, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)


        if vswfilter==[450,460]:
            Steps=arange(21,81,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=455.,runavg=4,MAX_velmin=300, MAX_velmax=610,stopstep=72, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)

        if vswfilter==[460,470]:
            Steps=arange(20,81,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=465.,runavg=4,MAX_velmin=300, MAX_velmax=630,stopstep=71, cmult=cmult,plot_evalrange=True, Xrange=[240,650],Yrange=Yrange,Xrange_log=[110,1000],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=None,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)

        #fast
        elif vswfilter==[470,480]:
            #Steps=arange(20,72,1)
            Steps=arange(20,77,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=475.,runavg=5,MAX_velmin=370, MAX_velmax=2000,stopstep=71, cmult=cmult,plot_evalrange=True, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=475,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=True,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        elif vswfilter==[480,490]:
            #Steps=arange(19,76,1)#for Gaussian case
            Steps=arange(16,76,1)#doe Gaussian-Moyal case 
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=485.,runavg=5,MAX_velmin=380, MAX_velmax=2000,stopstep=70, cmult=cmult,plot_evalrange=True, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=485,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=True,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)

    
        elif vswfilter==[490,500]:
            #Steps=arange(20,72,1)
            Steps=arange(16,76,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=495.,runavg=5,MAX_velmin=390, MAX_velmax=2000,stopstep=69, cmult=cmult,plot_evalrange=True, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=495,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=True,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        

        elif vswfilter==[500,510]:
            #Steps=arange(20,72,1)
            Steps=arange(18,75,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=505.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=68, cmult=cmult,plot_evalrange=False, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=0.1,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=505,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=False,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions,plot_436=plot_436)
        
        elif vswfilter==[510,520]:
            #Steps=arange(20,72,1)
            Steps=arange(18,71,1)
            stopstep=67
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=515.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=stopstep, cmult=cmult,plot_evalrange=True, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=515,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=True,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        elif vswfilter==[520,530]:
            #Steps=arange(20,72,1)
            Steps=arange(18,72,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=525.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=66, cmult=cmult,plot_evalrange=True, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=525,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=True,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        elif vswfilter==[530,540]:
            #Steps=arange(20,72,1)
            Steps=arange(17,71,1)
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=535.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=65, cmult=cmult,plot_evalrange=True, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=535,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=True,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        elif vswfilter==[540,550]:
            Steps=arange(18,71,1)
            stopstep=64
            #Steps=arange(13,71,1)#for stopstep=0
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=545.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=stopstep, cmult=cmult,plot_evalrange=True, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=545,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=True,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)

        elif vswfilter==[550,560]:
            stopstep=0
            Steps=arange(15,60,1)#for stopstep=0
            velmean_data=self.data.analyze_veldist(ionlist=fitresult_data[-1],Chi=fitresult_data,modelnumber=0,steps=Steps,ions_plot=ions_plot,cfracs=[0.61,0.32,0.14],velref=545.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=stopstep, cmult=cmult,plot_evalrange=True, Xrange=[350,720],Yrange=Yrange,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=legsize,labelsize=labelsize,ticklabelsize=ticklabelsize,vproton=545,figtitle="",savefigure=savefigure,figpath="",figname="test",peakshape=peakmodel,plot_errorbars=False, Nboot=1000,plot_steps=plot_steps,Plot=True, figformat_autom=True, fitgauss=True,vth_fitguess=45,save_meanvels=save_meanvels,filepath=filepath,filename=filename,scale_ions=scale_ions)
        
        
        else:
            velmean_data=None
            print("no valid vswfilter selected (only vswfilter=[330,340] and vswfilter=[330,340] are currently implemented)")  
            
        return velmean_data
    


Ionlist_select=["C4+","C5+","N6+","O6+","O7+","Ne8+","Mg6+","Mg7+", "Mg9+", "Mg10+","Si7+","Si8+", "Si9+","S7+", "Ca10+","Fe8+", "Fe9+","Fe10+","Fe11+","Fe12+"]

Ionlist_valid=["C4+","C5+","N6+","O5+","O6+","O7+","Ne7+","Ne8+","Mg6+","Mg7+","Mg8+", "Mg9+","Mg10+","Si7+","Si8+", "Si9+","Si10+","S7+", "Ca10+","Fe7+","Fe8+", "Fe9+","Fe10+","Fe11+","Fe12+"]

#Ionlist_select=["C4+","C5+","N6+","O5+","O6+","O7+","Ne8+","Mg6+","Mg7+","Mg8+", "Mg9+","Mg10+","Si7+","Si8+", "Si9+","Si10+","S7+", "Ca10+","Ca11+","Fe7+","Fe8+", "Fe9+","Fe10+","Fe11+","Fe12+"]#for AGU



longterm_cycledata=loadtxt("longterm_cycles_valid.txt",unpack=True)
cycle_vswmin=longterm_cycledata[0]#in km/s, edges included in interval
cycle_vswmax=longterm_cycledata[1]#in km/s, edges included in interval
cycle_minstopstep=longterm_cycledata[2]#in Epq-step, this step is still included in to calculate longterm_cycles_valid
Ncycle_valid=longterm_cycledata[3]#number of valid CTOF 5-min cycles within the filtered vsw-range which reached at least minimum stop-step indicated above 

#Figure sizes
figx_full=13.8
figx_half=6.8
figy_full=7
figy_half=5
        

    
class Ionmeanvels(object):
    
    def __init__(self,elements=["carbon","oxygen","silicon","iron"],max_meanveldiff=15,filepath="Veldist_moments/Longterm/Test/",include_systemerrors=False,ionlist_select=None):
        """
        Docstring:
        Class to claculate the overall mean speed for the different ion species from the different (1, 1.5, 2)-sigma mean peeds obtained from the longterm-data VDFs.
        
        Parameters:
        max_meanveldiff: maximum difference between the (1, 1.5, 2)-sigma mean speeds to obtain a valid overall mean speed. This is done to calculate a more robust mean speed (as explained in Janitzek, PhD Thesis (2020). The maximum value should be on the order of 1 Epw-step (in the fast or slow wind, respectively), i.e between ~7 and 20 km/s.   
        
        Examples:
        I=Ionmeanvels(max_meanveldiff=20,filepath="Veldists_moments_prefinal/Slow/Gauss/Full/",elements=["carbon_valid_ggm","nitrogen_valid_ggm","oxygen_valid_ggm","neon_valid_ggm","magnesium_valid_ggm","silicon_valid_ggm","sulfur_valid_ggm","calcium_valid_ggm","iron_valid_ggm"])
         
        I=Ionmeanvels(max_meanveldiff=20,filepath="Veldists_moments_prefinal/Slow/Gaussmoyal/Full/",elements=["carbon_valid_ggm","nitrogen_valid_ggm","oxygen_valid_ggm","neon_valid_ggm","magnesium_valid_ggm","silicon_valid_ggm","sulfur_valid_ggm","calcium_valid_ggm","iron_valid_ggm"],include_systemerrors=True)

        """
        #print "Ion mean speeds loaded (for optimal tail-scaling parameter A_C=0.00325 tofch^-1) for element:"
        self.elements=elements
        if include_systemerrors==True:
            self.Ions,self.Element_colors,self.M,self.MPQ,self.Elementlabel,self.Meanvels_sigmas=self.load_meanvel_data(filepath=filepath+"Optimal/")
            self.Meanvels_valid,self.Meanvels_sigmas_valid,self.M_valid,self.MPQ_valid,self.Elementlabel_valid,self.Ions_valid,self.Ions_invalid, self.Element_colors_valid =self.calc_velmeans_valid(max_meanveldiff=max_meanveldiff,tailpar="Optimal")
            
            #print "\n"
            #print "Ion mean speeds loaded (for estimated upper tail-scaling parameter A_C=0.00375 tofch^-1) for element:"
            self.elements_up=elements
            self.Ions_up,self.Element_colors_up,self.M_up,self.MPQ_up,self.Elementlabel_up,self.Meanvels_sigmas_up=self.load_meanvel_data(filepath=filepath+"Up/")
            self.Meanvels_up_valid,self.Meanvels_sigmas_up_valid,self.M_up_valid,self.MPQ_up_valid,self.Elementlabel_up_valid,self.Ions_up_valid,self.Ions_up_invalid, self.Element_colors_up_valid =self.calc_velmeans_valid(max_meanveldiff=max_meanveldiff, tailpar="Up")
            
            #print "\n"
            #print "Ion mean speeds loaded (for estimated lower tail-scaling parameter A_C=0.00275 tofch^-1) for element:"
            self.elements_low=elements
            self.Ions_low,self.Element_colors_low,self.M_low,self.MPQ_low,self.Elementlabel_low,self.Meanvels_sigmas_low=self.load_meanvel_data(filepath=filepath+"Low/")
            self.Meanvels_low_valid,self.Meanvels_sigmas_low_valid,self.M_low_valid,self.MPQ_low_valid,self.Elementlabel_low_valid,self.Ions_low_valid,self.Ions_low_invalid, self.Element_colors_low_valid =self.calc_velmeans_valid(max_meanveldiff=max_meanveldiff, tailpar="Low")
        else:     
            self.Ions,self.Element_colors,self.M,self.MPQ,self.Elementlabel,self.Meanvels_sigmas=self.load_meanvel_data(filepath=filepath)
            self.Meanvels_valid,self.Meanvels_sigmas_valid,self.M_valid,self.MPQ_valid,self.Elementlabel_valid,self.Ions_valid,self.Ions_invalid, self.Element_colors_valid =self.calc_velmeans_valid(max_meanveldiff=max_meanveldiff,ionlist_select=ionlist_select,tailpar="Optimal")
            
        
    def load_meanvel_data(self,filepath="Veldist_moments/Longterm/Test/"):
        
        Ions=[]
        Element_colors=[]
        M=[]
        MPQ=[]
        Element_label=[]
        Meanvels_sigmas=empty(shape=(0,3))
        for element in self.elements:
            #print element
            
            #load ion names
            meanvels_file=open(filepath+"%s"%(element),"r")
            header=meanvels_file.readlines()[1].split()
            #print header
            velkey="vmean"
            for s in header:
                if len(s)>5 and s[0:6]=="vmean_":
                    #print s
                    ion=s[6:]
                    #print ionname 
                    Ions.append(ion)
                    mass=iondict[ion][0]
                    Element_colors.append(Element_colordict[mass])
                    charge=iondict[ion][1]
                    mpq=mass/float(charge)
                    element_label=Element_labeldict[mass]
                    M.append(mass)
                    MPQ.append(mpq)
                    Element_label.append(element_label)
            #load ion mean speeds
            meanvel_data=loadtxt(filepath+"%s"%(element),skiprows=1,delimiter=" ",unpack=True)
            meanvels_sigmas=meanvel_data[1:]
            Meanvels_sigmas=vstack([Meanvels_sigmas,meanvels_sigmas])
        Ions=array(Ions)
        M=array(M)
        MPQ=array(MPQ)
        Element_colors=array(Element_colors)
        Element_label=array(Element_label)
        Meanvels_sigmas=array(Meanvels_sigmas)
        return Ions, Element_colors, M, MPQ, Element_label, Meanvels_sigmas
                    


        
    def calc_velmeans_valid(self,max_meanveldiff=15,ionlist_select=None,tailpar="Optimal"):
        
        self.veldelta_stat=max_meanveldiff
        
        if tailpar=="Optimal":
            Ions=self.Ions
            Meanvels_sigmas=self.Meanvels_sigmas
            M=self.M
            MPQ=self.MPQ
            Elementlabel=self.Elementlabel
            Element_colors=self.Element_colors
        elif tailpar=="Up":
            Ions=self.Ions_up
            Meanvels_sigmas=self.Meanvels_sigmas_up
            M=self.M_up
            MPQ=self.MPQ_up
            Elementlabel=self.Elementlabel_up
            Element_colors=self.Element_colors_up
        elif tailpar=="Low":
            Ions=self.Ions_low
            Meanvels_sigmas=self.Meanvels_sigmas_low
            M=self.M_low
            MPQ=self.MPQ_low
            Elementlabel=self.Elementlabel_low
            Element_colors=self.Element_colors_low
        
        if ionlist_select==None:
            Ions_select=Ions
        else:
            Ions_select=ionlist_select
        
        selmask=in1d(self.Ions,Ions_select)
        print("selmask:", selmask)
        M=M[selmask]
        MPQ=MPQ[selmask]
        Element_colors=Element_colors[selmask]
        Elementlabel=Elementlabel[selmask]
        Meanvels_sigmas=Meanvels_sigmas[selmask]
        
        
        Meanvels_valid=[]
        Ions_valid=[]
        Meanvels_sigmas_valid=[]
        Ions_invalid=[]
        Element_colors_valid=[]
        M_valid=[]
        MPQ_valid=[]
        Elementlabel_valid=[]
        for i,ion in enumerate(Ions_select): 
            meanvels_ion=Meanvels_sigmas[i]
            #if (abs(meanvels_ion[0]-meanvels_ion[1])<=max_meanveldiff)*(abs(meanvels_ion[1]-meanvels_ion[2])<=max_meanveldiff):
            if (max(meanvels_ion)-min(meanvels_ion))<=max_meanveldiff:
            
                meanvel_ion=average(meanvels_ion)
                Meanvels_valid.append(meanvel_ion)
                Ions_valid.append(ion)
                Meanvels_sigmas_valid.append(Meanvels_sigmas[i])
                Element_colors_valid.append(Element_colordict[M[i]])
                M_valid.append(M[i])
                MPQ_valid.append(MPQ[i])
                Elementlabel_valid.append(Elementlabel[i])
            else:
                Ions_invalid.append(ion)
        
        Meanvels_valid=array(Meanvels_valid)
        return Meanvels_valid,Meanvels_sigmas_valid, M_valid,MPQ_valid, Elementlabel_valid, Ions_valid,Ions_invalid, Element_colors_valid


    def calc_Zvalue(self,protonspeed_reference, add_mean_staterror=True, mean_systemerror=None): 
        """
        significance of differential streaming in units of sigma (test whether two Gaussians are different, see literature to Z-test)
        """
        
        if mean_systemerror==None:
            mean_systemerror=0
        
        if add_mean_staterror==True:
            meanspeeds=array([self.Meanvels_valid,self.Meanvels_valid,self.Meanvels_valid]).T
            #staterrors=amax(abs(meanspeeds-self.Meanvels_sigmas_valid),axis=1)
            staterrors=mean(abs(meanspeeds-self.Meanvels_sigmas_valid),axis=1)
            mean_staterror=mean(staterrors) 
        else: mean_staterror=0

        N_sample=len(self.Meanvels_valid)#sample size of valid ion sample 
        mean_ionspeed=mean(self.Meanvels_valid)#mean speed of valid ion sample
        std_ionspeed=std(self.Meanvels_valid,ddof=1)#standard deviation of valid ion sample
        
        #std_total=std_ionspeed+mean_staterror+mean_systemerror
        std_total=sqrt(std_ionspeed**2+mean_staterror**2+mean_systemerror**2)
        print("mean_staterror, mean_systemerror:", mean_staterror, mean_systemerror)
        
        #Z=(mean_ionspeed-mean_proton_speed)/sqrt(2*(std_total/sqrt(N_sample))**2)#old
        
        Z=sqrt(N_sample)*(mean_ionspeed-protonspeed_reference)/std_total
        return Z
      
      
    def select_ionspecies(self,ions_select=[]):
        Ions_select=[]
        Meanvels_select=[]
        Meanvels_sigmas_select=[]
        M_select=[]
        MPQ_select=[]
        Elementlabel_select=[]
        Element_colors_select=[]
        i=0
        while i<len(self.MPQ_valid):
            if self.Ions[i] in ions_select:
                Ions_select.append(self.Ions_valid[i])
                Meanvels_select.append(self.Meanvels_valid[i])
                Meanvels_sigmas_select.append(self.Meanvels_sigmas_valid[i])
                M_select.append(self.M_valid[i])
                MPQ_select.append(self.MPQ_valid[i])
                Elementlabel_select.append(self.Elementlabel_valid[i])
                Element_colors_select.append(self.Element_colors_valid[i])
            i+=1
        
        self.Ions_valid=array(Ions_select)
        self.Meanvels_valid=array(Meanvels_select)
        self.Meanvels_sigmas_valid=array(Meanvels_sigmas_select)
        self.M_valid=array(M_select)
        self.MPQ_valid=array(MPQ_select)
        self.Elementlabel_valid=array(Elementlabel_select)
        self.Element_colors_valid=array(Element_colors_select)
        


    def plot_velmeans(self,vswfilter=[500,510],Xrange=[1.5,8.5],Yrange=[305,360],plot_staterrors=True,plot_systemerrors=False, fontsize=16, ticklabelsize=14, legsize=13,plot_uncertainty_filter=False,plot_meanspeed=False, plot_meandiffspeed=True,plot_vstd=True,plot_legend=True, plot_vswrange_label=True, plot_Zvalue_upper=False,plot_Zvalue_lower=False, plot_Zvalue_mean=True,save_figure=False, figpath="", filename="meanspeeds_test",linefit=False,linefit_weights=True,plot_diffspeeds=False,plot_validcycles=True,panel_label=None,figurtitle=None,modeltext=None,figwidth="full"):
        """
        a1,b1,c1,d1=I.plot_velmeans(vswfilter=[330,340],Yrange=[285,365],plot_uncertainty_filter=False,save_figure=False,filename="meanspeed_staterrors_slow_ggm_gauss",plot_staterrors=True,plot_systemerrors=False)
        
        a3,b3,c3=I.plot_velmeans(vswfilter=[330,340],Yrange=[285,365],plot_uncertainty_filter=False,save_figure=False,filename="meanspeed_systemerrors_slow_gaussmoyal_ggm",plot_staterrors=False,plot_systemerrors=True)
        """
        
        Elementlabel_valid=self.Elementlabel_valid
        MPQ_valid=self.MPQ_valid
        Meanvels_valid=self.Meanvels_valid
        Meanvels_Error_valid=zeros((len(self.Meanvels_valid),2))
        
        vswmean=(vswfilter[0]+vswfilter[1])/2.
        if plot_diffspeeds==True:
            vsw_sub=vswmean
        else:
            vsw_sub=0
        
        if figwidth=="full":
            fig, ax = plt.subplots(1,1,figsize=(figx_full, 0.5*figx_full))
            fontsize=20
            ticklabelsize=16
            legsize=20
            markersize=8
            elinewidth=2.5, 
            capsize=4
            capthick=2.5
        elif figwidth=="half":
            fig, ax = plt.subplots(1,1,figsize=(figx_half, figy_half))
            fontsize=16
            ticklabelsize=14
            legsize=14
            markersize=5
            elinewidth=1.5, 
            capsize=2
            capthick=2        
        i=0
        elements_labeled=[]
        
        Nions_valid=len(self.MPQ_valid)
        while i<len(self.MPQ_valid):
            
            if plot_staterrors==True:
                lower_error_stat=self.Meanvels_valid[i]-min(self.Meanvels_sigmas_valid[i])
                upper_error_stat=max(self.Meanvels_sigmas_valid[i])-self.Meanvels_valid[i]
                stat_error = array([[lower_error_stat, upper_error_stat]]).T
                Meanvels_Error_valid[i]=stat_error.T[0]
                
            elif plot_systemerrors==True:
                lower_error_system=self.Meanvels_valid[i]-self.Meanvels_low_valid[i]
                upper_error_system=self.Meanvels_up_valid[i]-self.Meanvels_valid[i]
                system_error=array([[lower_error_system, upper_error_system]]).T
                Meanvels_Error_valid[i]=system_error.T[0]
                #print self.Ions_valid[i], self.Meanvels_low_valid[i], self.Meanvels_valid[i], self.Meanvels_up_valid[i], lower_error_system, upper_error_system
                #return system_error
            
            if self.M_valid[i] in elements_labeled: 
                if plot_staterrors==True:
                    ax.errorbar([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub],yerr=stat_error, linestyle="None", marker="o",markersize=markersize,elinewidth=elinewidth, capsize=capsize, capthick=capthick, color=self.Element_colors_valid[i])
                elif plot_systemerrors==True:
                    ax.errorbar([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub],yerr=system_error, linestyle="None", marker="o",markersize=markersize,elinewidth=elinewidth, capsize=capsize, capthick=capthick, color=self.Element_colors_valid[i])
                    if system_error[0]+system_error[1]>=0:
                        #ax.plot([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub+system_error[1]+2.3], linestyle="None", marker="^",markersize=markersize, color="k",zorder=10)#for fast wind
                        ax.plot([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub+system_error[1]+1.5], linestyle="None", marker="^",markersize=markersize, color="k",zorder=10)
                else:
                    ax.plot([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub], linestyle="None", marker="o", color=self.Element_colors_valid[i])
            else:
                if plot_staterrors==True:
                    ax.errorbar([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub],yerr=stat_error, linestyle="None", marker="o",markersize=markersize,elinewidth=elinewidth, capsize=capsize, capthick=capthick, color=self.Element_colors_valid[i],label="%s"%(self.Elementlabel_valid[i]))
                elif plot_systemerrors==True:
                    ax.errorbar([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub],yerr=system_error, linestyle="None", marker="o", markersize=markersize,elinewidth=elinewidth, capsize=capsize, capthick=capthick, color=self.Element_colors_valid[i],label="%s"%(self.Elementlabel_valid[i]))
                    if system_error[0]+system_error[1]>=0:
                        #ax.plot([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub+system_error[1]+2.3], linestyle="None", marker="^",markersize=markersize, color="k",zorder=10)#for fast wind
                        ax.plot([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub+system_error[1]+1.5], linestyle="None", marker="^",markersize=markersize, color="k",zorder=10)
                else:
                    ax.plot([self.MPQ_valid[i]],[self.Meanvels_valid[i]-vsw_sub], linestyle="None", marker="o", markersize=markersize, color=self.Element_colors_valid[i],label="%s"%(self.Elementlabel_valid[i]))
                elements_labeled.append(self.M_valid[i])
            
            print(vsw_sub, self.Ions_valid[i])
            if self.Ions_valid[i]=="C5+":
                vdiff_C5=self.Meanvels_valid[i]-vsw_sub
                vdifferr_C5=Meanvels_Error_valid[i]
            if self.Ions_valid[i]=="O6+":
                vdiff_O6=self.Meanvels_valid[i]-vsw_sub
                vdifferr_O6=Meanvels_Error_valid[i]
            if self.Ions_valid[i]=="Ne8+":
                vdiff_Ne8=self.Meanvels_valid[i]-vsw_sub
                vdifferr_Ne8=Meanvels_Error_valid[i]
            if self.Ions_valid[i]=="Si7+":
                vdiff_Si7=self.Meanvels_valid[i]-vsw_sub
                vdifferr_Si7=Meanvels_Error_valid[i]
            if self.Ions_valid[i]=="Si8+":
                vdiff_Si8=self.Meanvels_valid[i]-vsw_sub
                vdifferr_Si8=Meanvels_Error_valid[i]
            if self.Ions_valid[i]=="Fe9+":
                vdiff_Fe9=self.Meanvels_valid[i]-vsw_sub
                vdifferr_Fe9=Meanvels_Error_valid[i]
            if self.Ions_valid[i]=="Fe10+":
                vdiff_Fe10=self.Meanvels_valid[i]-vsw_sub
                vdifferr_Fe10=Meanvels_Error_valid[i]
            i+=1
        
        ax.plot([Xrange[0],Xrange[1]],[vswmean-vsw_sub,vswmean-vsw_sub],color="k")
        ax.plot([Xrange[0],Xrange[1]],[vswfilter[0]-vsw_sub,vswfilter[0]-vsw_sub],color="k",linestyle="--")
        ax.plot([Xrange[0],Xrange[1]],[vswfilter[1]-vsw_sub,vswfilter[1]-vsw_sub],color="k",linestyle="--")
        ax.set_xlabel(r"$ \rm{mass-per-charge \ [amu/e]}$",fontsize=fontsize)
        
        
        if plot_diffspeeds==True:
            #ax.set_ylabel(r"$ \rm{differential \ speed \ [km/s]}$",fontsize=fontsize)
            ax.set_ylabel(r"$ \rm{v_{i} \ - \ v_{p} \ \ [km/s]}$",fontsize=fontsize)
        else:
            ax.set_ylabel(r"$ \rm{ion \ mean \ speed \ [km/s]}$",fontsize=fontsize)
            
        
        if panel_label!=None:
            ax.text(-0.1,0.9," %s "%(panel_label), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize)        
        
        if figurtitle!=None:
            ax.set_title(r"$\rm{%s}$"%(figurtitle),fontsize=fontsize)
        
        
        if plot_uncertainty_filter==True:
            ax.set_title(r"$\rm{statistical \ uncertainty \ } \delta \rm{ v } \leq %i \rm{\ km/s}$"%(self.veldelta_stat),fontsize=fontsize)
        
        if plot_legend==True:
            ax.legend(loc="upper center",ncol=5,prop={'size': legsize-2})
            ypos_offset=0.
        else:    
            ypos_offset=0.15
        
        if modeltext!=None:
            ax.text(.99,.02,r"$\rm{%s}$"%(modeltext), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize)        
        
        
        if plot_meanspeed==True:
            vmean_ions=mean(self.Meanvels_valid)
            ax.text(.99,.75+ypos_offset,r"$\langle \rm{v}_{i} \rangle_s \ = %.1f\,\rm{km/s} $"%(vmean_ions), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize-2)        
        
            if plot_vstd==True:
                vstd_ions=std(self.Meanvels_valid,ddof=1)
                ax.text(.99,.65+ypos_offset,r"$\sigma_s \ (\rm{v}_{i}) \ = %.1f\,\rm{km/s} $"%(vstd_ions), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize-2)        
        
        elif plot_meandiffspeed==True:
            vmean_ions=mean(self.Meanvels_valid)
            vdiff_mean=vmean_ions-vswmean
            ax.text(.99,.75+ypos_offset,r"$\langle \Delta \rm{v}_{i,p} \rangle_s \ = %.1f\,\rm{km/s} $"%(vdiff_mean), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize-2)        
        
            if plot_vstd==True:
                vstd_ions=std(self.Meanvels_valid,ddof=1)
                ax.text(.99,.65+ypos_offset,r"$\sigma_s \ (\rm{v}_{i}) \ = %.1f\,\rm{km/s} $"%(vstd_ions), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize-2)        
            
        if plot_Zvalue_mean==True:
            Zvalue_mean=self.calc_Zvalue(protonspeed_reference=vswmean, add_mean_staterror=True, mean_systemerror=5.)
            ax.text(.99,.55+ypos_offset,r"$\rm{Z_{mean}} \ = %.1f$"%(Zvalue_mean), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize-2)
        
        if plot_Zvalue_upper==True:
            Zvalue_upper=self.calc_Zvalue(protonspeed_reference=vswfilter[-1], add_mean_staterror=True, mean_systemerror=5.)
            ax.text(.99,.45+ypos_offset,r"$\rm{Z_{up}} \ = \ %.1f$"%(Zvalue_upper), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize-2)
        
        if plot_Zvalue_lower==True:
            Zvalue_lower=self.calc_Zvalue(protonspeed_reference=vswfilter[0], add_mean_staterror=True, mean_systemerror=5.)
            ax.text(.99,.45+ypos_offset,r"$\rm{Z_{low}} \ = \ %.1f$"%(Zvalue_lower), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize-2)
        
        if plot_vswrange_label==True:
            ax.text(.03,.75+ypos_offset,r"$\rm{v_p} \in [%i\ \rm{km/s}, \ %i \ \rm{km/s}]$"%(vswfilter[0],vswfilter[-1]), horizontalalignment='left',transform=ax.transAxes,fontsize=fontsize-2)
        
        if plot_validcycles==True:
            n=where((cycle_vswmin==vswfilter[0])*(cycle_vswmax==vswfilter[-1]))[0][0]
            Ncycle=Ncycle_valid[n]
            #Ncycle=63#only for vp515 an no minimum Epq-stop-step condition
            ax.text(.03,.65+ypos_offset,r"$\rm{N_{cycles}} \ = \ %i$"%(Ncycle), horizontalalignment='left',transform=ax.transAxes,fontsize=fontsize-2)
        

        MPQ_valid=array(MPQ_valid)
        Meanvels_valid=array(Meanvels_valid)
        Meanvels_Error_valid=array(Meanvels_Error_valid)
        Elementlabel_valid=array(Elementlabel_valid)
        
        Meanvels_Error_valid_mean=mean(Meanvels_Error_valid,axis=1)
        vmean_errweight=average(Meanvels_valid,weights=1./Meanvels_Error_valid_mean)
        

        if linefit==True:
            
            xdata=MPQ_valid
            ydata=Meanvels_valid				
            fitfunc = lambda p, x: linfunc(p,x)
            #fitfunc = lambda p, x: constfunc(p,x)
            #mean_measerr=(stat_error[0]+stat_error[1])/2.
            #return MPQ_valid, Meanvels_valid, Meanvels_Error_valid_mean
            
            err=Meanvels_Error_valid_mean
            err[err<1]=1.
            if linefit_weights==True:
                errfunc = lambda p, x, y: ((fitfunc(p, x) - y)/err) 
            else:
                errfunc = lambda p, x, y: (fitfunc(p, x) - y)
            p0 = [0.,500.]
            #p0 = [500.]
            
            #return xdata, ydata,weights
            args = optimize.leastsq(errfunc, p0[:], args=(xdata, ydata), full_output=1)
            p1 = args[0]
            xrun=arange(min(MPQ_valid),max(MPQ_valid),0.01)
            yfit=fitfunc(p1,xdata)
            ax.plot(xrun,fitfunc(p1,xrun)-vsw_sub,linewidth=2,linestyle=":",color="darkviolet")
            
            chi2=sum(((ydata-yfit)/err)**2)
            chi2_red=chi2/(len(ydata)-1.)
            
            ax.text(.99,.03,r"$\rm{\chi^2_{red}} \ = %.1f$"%(chi2_red), horizontalalignment='right',transform=ax.transAxes,fontsize=fontsize-2,color="darkviolet")
            
            ax.text(.03,.03,r"$\rm{grad(v_{ion})} \ = %.2f \ \rm{ (km \cdot e)/(s \cdot amu)}$"%(p1[0]), horizontalalignment='left',transform=ax.transAxes,fontsize=fontsize-2,color="darkviolet")
            
            
            print(xdata)
            print(ydata)
            print(err)
            print(min(Meanvels_valid), max(Meanvels_valid), min(err), max(err))
            print("lin. fit parameter (gradient, offset):",p1)
            print("chi2_red:", chi2_red)

        ax.tick_params(axis="x", labelsize=ticklabelsize)
        ax.tick_params(axis="y", labelsize=ticklabelsize)
        #ax[1].grid(which='major')
        ax.set_xlim(Xrange[0],Xrange[1])
        ax.set_ylim(Yrange[0],Yrange[1])
        
        if save_figure==True:
            plt.savefig(figpath+filename+".png",bbox_inches='tight')
        
        
        """
        #return vmean_errweight, Meanvels_valid, vmean_errweight-Meanvels_valid, (vmean_errweight-Meanvels_valid)/(Meanvels_Error_valid_mean+3)
        chi2=sum(((vmean_errweight-Meanvels_valid)/Meanvels_Error_valid_mean)**2)
        chi2_red=chi2/(len(Meanvels_valid)-1.)
        
        print "chi2_test", vmean_errweight,chi2,chi2_red 
        """
        
        return MPQ_valid, Meanvels_valid, Meanvels_Error_valid,Elementlabel_valid
        #return Nions_valid, vdiff_mean, vstd_ions, Zvalue_lower, Zvalue_mean, Zvalue_upper,vdiff_C5,vdifferr_C5,vdiff_O6,vdifferr_O6,vdiff_Ne8,vdifferr_Ne8, vdiff_Si7,vdifferr_Si7,vdiff_Si8,vdifferr_Si8,vdiff_Fe9,vdifferr_Fe9,vdiff_Fe10,vdifferr_Fe10
    
    
    
def plot_velmeans_arrays(MPQ_array, Meanvels_array, Meanvels_Error_array,Elementlabels_array, vswfilter=[330,340],customtitle="ion mean speeds (incl. statistical uncertainties)", Xrange=[1.0,9.0],Yrange=[305,360], figx=13,figy=10,markersize=10, elinewidth=2,alpha=0.6,fontsize=20, ticklabelsize=16, legsize=18,save_figure=False,figpath="", filename="meanspeeds_array_test", plot_vswrange_label=True, plot_ggmtitle=True,plot_legend=True):#
    """
    Examples:
    plot_velmeans_arrays(MPQ_array=[a1,a2],Meanvels_array=[b1,b2],Meanvels_Error_array=[c1,c2],Elementlabels_array=[d3,d4],Yrange=[285,365],legsize=18.,save_figure=True,filename="meanspeeds_gauss_gaussmoyal_staterrors",customtitle="ion mean speeds (incl. statistical uncertainties)")
    """
    
    fig, ax = plt.subplots(1,1,figsize=(figx, figy))
    marker_colors=["b","r","orange"]
    marker_symbols=["^","x","+","s","^","o","o","X","s"]
    mfc=[["b","r","orange"],["b","r","orange"],["b","r","orange"],["b","r","orange"],["w","w","w"],["w","w","w"],["b","r","orange"],["b","r","orange"],["w","w","w"]]
    lc=[["k","k","k"],["k","k","k"],["k","k","k"],["k","k","k"],["w","w","w"],["w","w","w"],["k","k","k"],["k","k","k"],["w","w","w"]]
    
    #markersize=8
    #elinewidth=2.5, 
    #capsize=4
    #capthick=2.5
    markeredgewidth=2.5
    
    if plot_ggmtitle==True:
        ax.text(.03,.92,r"$\rm{CTOF}$", horizontalalignment='left', transform=ax.transAxes,fontsize=fontsize+6,color="k")
        
        ax.text(.03+0.13,.92,r"$\rm{Gaussian}$", horizontalalignment='left', transform=ax.transAxes,fontsize=fontsize,color="b")
    
        ax.text(.165+0.13,.92,r"$\rm{vs}$", horizontalalignment='left', transform=ax.transAxes,fontsize=fontsize,color="k")
        
        ax.text(.205+0.13,.92,r"$\rm{Gaussian-Moyal}$", horizontalalignment='left', transform=ax.transAxes,fontsize=fontsize,color="r")
    
    
    if plot_vswrange_label==True:
        ax.text(.97,.92,r"$\rm{v_p} \in [%i\ \rm{km/s}, \ %i \ \rm{km/s}]$"%(vswfilter[0],vswfilter[-1]), horizontalalignment='right', transform=ax.transAxes,fontsize=fontsize)

    
    j=0
    while j<len(MPQ_array):
    
        for i,Elabel in enumerate(unique(Elementlabels_array)):
            #return Elementlabels_array[j],Elabel
            Elabelmask=(Elementlabels_array[j]==Elabel)
            MPQ=MPQ_array[j][Elabelmask]
            Meanvels=Meanvels_array[j][Elabelmask]
            Meanvels_Error=(Meanvels_Error_array[j][Elabelmask]).T
            #return Elabelmask, MPQ, Meanvels, Meanvels_Error
            
            if j==2:
                ax.errorbar(MPQ, Meanvels, yerr=Meanvels_Error,linestyle="None", marker=marker_symbols[i],markersize=0,elinewidth=10,capsize=5,markerfacecolor=mfc[i][j], markeredgewidth=0,markeredgecolor=0,ecolor=marker_colors[j],alpha=alpha-0.3)
            
            else:
                ax.errorbar(MPQ, Meanvels, yerr=Meanvels_Error,linestyle="None", marker=marker_symbols[i],markersize=markersize,elinewidth=elinewidth,markerfacecolor=mfc[i][j], markeredgewidth=markeredgewidth,markeredgecolor=marker_colors[j],ecolor=marker_colors[j],alpha=alpha)
            
            
            
            if j==0:
                ax.errorbar(MPQ, zeros((len(Meanvels))), yerr=zeros((len(Meanvels))),linestyle="None", marker=marker_symbols[i],markersize=markersize,elinewidth=elinewidth,markerfacecolor=lc[i][j], markeredgewidth=markeredgewidth,markeredgecolor="k",ecolor="k",alpha=alpha,label="%s"%(Elabel))
                
           
        j+=1

    """
    handles, labels = ax.get_legend_handles_labels()
    for h in handles:
        h[0].set_color("k")
    """       
    v_alfven_estim=70/1.4
    vswmean=(vswfilter[0]+vswfilter[1])/2.
    ax.plot([Xrange[0],Xrange[1]],[vswmean,vswmean],color="k",linewidth=2)
    ax.plot([Xrange[0],Xrange[1]],[vswmean+v_alfven_estim,vswmean+v_alfven_estim],color="k",linestyle="--",linewidth=2)
    ax.plot([Xrange[0],Xrange[1]],[vswfilter[0],vswfilter[0]],color="k",linestyle="--")
    ax.plot([Xrange[0],Xrange[1]],[vswfilter[1],vswfilter[1]],color="k",linestyle="--")
    if plot_legend==True:
        leg=ax.legend(loc="lower right",ncol=5,prop={'size': legsize})
        leg.legendHandles[0].set_color('k')
        leg.legendHandles[1].set_color('k')
    ax.set_xlabel(r"$ \rm{mass-per-charge \ [amu/e]}$",fontsize=fontsize)
    #ax.set_ylabel(r"$ \rm{speed \ [km/s]}$",fontsize=fontsize)
    ax.set_ylabel(r"$ \rm{differential \ speed \ / \ Alfv\'en \ speed}$",fontsize=fontsize)
    if customtitle !=None:
        ax.set_title(customtitle,fontsize=fontsize)
    
    
    ax.tick_params(axis="x", labelsize=ticklabelsize)
    ax.tick_params(axis="y", labelsize=ticklabelsize)
    #ax[1].grid(which='major')

    ax.set_xlim(Xrange[0],Xrange[1])
    ax.set_ylim(Yrange[0],Yrange[1])
    
    if save_figure==True:
        plt.savefig(figpath+filename+".png",bbox_inches='tight')
    
    
    
        
        
def plot_velmeans_errorcolor(max_meanveldiffs=[30,20,10,5],filepath="Veldists_moments_prefinal/Fast/Gaussmoyal/Full/Optimal/", elements=["carbon_valid", "nitrogen_valid", "oxygen_valid","neon_valid","magnesium_valid","silicon_valid", "sulfur_valid", "calcium_valid", "iron_valid"], vswfilter=[500,510],Xrange=[1.5,8.5],Yrange=[475,595],plot_staterrors=True,plot_vswrange_label=True,plot_validcycles=True, save_figure=False,figurtitle=None, figpath="",filename="meanspeeds_errorcolor_test",figwidth="full"):
    """
    plot_velmeans_arrays(MPQ_array=[a1,a2],Meanvels_array=[b1,b2],Meanvels_Error_array=[c1,c2],Elementlabels_array=[d3,d4],Yrange=[285,365],legsize=18.,save_figure=False,filename="meanspeeds_gauss_gaussmoyal_staterrors",customtitle="ion mean speeds (incl. statistical uncertainties)")
    
    plot_velmeans_arrays(MPQ_array=[a3,a4],Meanvels_array=[b3,b4],Meanvels_Error_array=[c3,c4],Elementlabels_array=[d3,d4],Yrange=[285,365],legsize=18.,save_figure=False,filename="meanspeeds_gauss_gaussmoyal_systemerrors",customtitle="ion mean speeds (incl. systematic peak-tail-model uncertainties)") 
    """
    
    if figwidth=="full":
        fig, ax = plt.subplots(1,1,figsize=(figx_full, figy_full))
        fontsize=20
        ticklabelsize=16
        legsize=20
        markersize=8
        elinewidth=2.5, 
        capsize=4
        capthick=2.5
    elif figwidth=="half":
        fig, ax = plt.subplots(1,1,figsize=(figx_half, figy_half))
        fontsize=16
        ticklabelsize=14
        legsize=13.3
        markersize=5
        elinewidth=1.5, 
        capsize=2
        capthick=2        

    if figurtitle!=None:
        ax.set_title(r"$\rm{%s}$"%(figurtitle),fontsize=fontsize)

    ypos_offset=0.
    if plot_vswrange_label==True:
        ax.text(.03,.75+ypos_offset,r"$\rm{v_p} \in [%i\ \rm{km/s}, \ %i \ \rm{km/s}]$"%(vswfilter[0],vswfilter[-1]), horizontalalignment='left',transform=ax.transAxes,fontsize=fontsize-2)

    if plot_validcycles==True:
        n=where((cycle_vswmin==vswfilter[0])*(cycle_vswmax==vswfilter[-1]))[0][0]
        Ncycle=Ncycle_valid[n]
        #Ncycle=63#only for vp515 an no minimum Epq-stop-step condition
        ax.text(.03,.65+ypos_offset,r"$\rm{N_{cycles}} \ = \ %i$"%(Ncycle), horizontalalignment='left',transform=ax.transAxes,fontsize=fontsize-2)
    

    
    #figx_half=6.8
    #figy=5#
    #fig, ax = plt.subplots(1,1,figsize=(figx_half, figy_))
    marker_colors=["m","r","b","k"]
    
    j=0
    while j<len(max_meanveldiffs):
        I=Ionmeanvels(max_meanveldiff=max_meanveldiffs[j],filepath=filepath,elements=elements)
        print("max_meanveldiff, N_ions_valid :", max_meanveldiffs[j], len(I.Meanvels_valid))
    
        if plot_staterrors==True:
            Stat_error=zeros((len(I.MPQ_valid),2))
            i=0
            while i<len(I.MPQ_valid):
                lower_error_stat=I.Meanvels_valid[i]-min(I.Meanvels_sigmas_valid[i])
                upper_error_stat=max(I.Meanvels_sigmas_valid[i])-I.Meanvels_valid[i]
                #stat_error = array([[lower_error_stat, upper_error_stat]]).T
                stat_error = array([lower_error_stat, upper_error_stat])
                #return Stat_error, stat_error 
                
                Stat_error[i]=stat_error
                i+=1
            Stat_errorT=Stat_error.T
            if j in arange(0,len(max_meanveldiffs)-1,1):
                ax.errorbar(I.MPQ_valid,I.Meanvels_valid, yerr=Stat_errorT,linestyle="None", marker="o",color=marker_colors[j],markersize=markersize,elinewidth=elinewidth, capsize=capsize, capthick=capthick,label=r"$ %i \rm{\ km/s} < \delta \rm{ v_{stat} } \leq %i \rm{\ km/s}$"%(max_meanveldiffs[j+1], max_meanveldiffs[j]))
            else:
                ax.errorbar(I.MPQ_valid,I.Meanvels_valid, yerr=Stat_errorT,linestyle="None", marker="o",color=marker_colors[j],markersize=markersize,elinewidth=elinewidth, capsize=capsize, capthick=capthick,label=r"$ \delta \rm{ v_{stat} } \leq %i \rm{\ km/s}$"%(max_meanveldiffs[j]))
        else:    
            ax.plot(I.MPQ_valid,I.Meanvels_valid, linestyle="None", marker="o",color=marker_colors[j], markersize=markersize,label=r"$ \delta \rm{ v_{stat} } \leq %i \rm{\ km/s}$"%(max_meanveldiffs[j]))
        j+=1


    vswmean=(vswfilter[0]+vswfilter[1])/2.
    ax.plot([Xrange[0],Xrange[1]],[vswmean,vswmean],color="k")
    ax.plot([Xrange[0],Xrange[1]],[vswfilter[0],vswfilter[0]],color="k",linestyle="--")
    ax.plot([Xrange[0],Xrange[1]],[vswfilter[1],vswfilter[1]],color="k",linestyle="--")
    ax.legend(loc="upper center",ncol=2,prop={'size': legsize-2})
    ax.set_xlabel(r"$ \rm{mass-per-charge \ [amu/e]}$",fontsize=fontsize)
    ax.set_ylabel(r"$ \rm{ion \ mean \ speed \ [km/s]}$",fontsize=fontsize)
    #if plot_uncertainty_filter==True:
    #    ax.set_title(r"$\rm{statistical \ uncertainty \ } \delta v \leq %i \rm{\ km/s}$"%(self.veldelta_stat))

    ax.tick_params(axis="x", labelsize=ticklabelsize)
    ax.tick_params(axis="y", labelsize=ticklabelsize)
    #ax[1].grid(which='major')

    ax.set_xlim(Xrange[0],Xrange[1])
    ax.set_ylim(Yrange[0],Yrange[1])
    
    if save_figure==True:
        plt.savefig(figpath+filename+".png",bbox_inches='tight')


def plot_countscor_total(path="Veldists_moments_prefinal/Slow_345/Gaussmoyal/Full/Optimal/",filename="silicon_counts",figurtitle="response \ model: \ Gaussian-Moyal",Yrange=None,normalize_counts=False, iron_efficiencies_included=False,plot_CSArnaud=True,save_figure=False, outfilename="Gaussmoyal_charge_states_silicon_slow345_test"):


    fontsize=20
    ticklabelsize=16
    legsize=20
    markersize=8
    elinewidth=2.5, 
    capsize=4
    capthick=2.5

    #approximate iron TC efficiencies for v=335 km/s (from Janitzek_PHD/Aellig_PHD 
    eff_Fe7=0.14
    eff_Fe8=0.16
    eff_Fe9=0.175
    eff_Fe10=0.19
    eff_Fe11=0.205
    eff_Fe12=0.22
    eff_Fe13=0.23
    eff_Fe14=0.24


    countdata=loadtxt(path+filename,unpack=True,skiprows=0)
    #return countdata
    cfracs=countdata[0]
    
    if filename=="magnesium_counts":
        counts_61=countdata.T[0][1:]
        counts_32=countdata.T[1][1:]
        counts_14=countdata.T[2][1:]
    
        T0_Arnaud=array([10**(-1.17),10**(-0.62),10**(-0.60),10**(-0.65),10**(-0.68)])
        
        T1_Arnaud=array([10**(-2.44),10**(-1.53),10**(-1.08),10**(-0.65),10**(-0.18)])
        T2_Arnaud=array([10**(-3.83),10**(-2.60),10**(-1.78),10**(-0.94),10**(-0.06)])
    
    
    if filename=="silicon_counts":
        counts_61=countdata.T[0][1:]
        counts_32=countdata.T[1][1:]
        counts_14=countdata.T[2][1:]
    
        T1_Arnaud=array([10**(-0.97),10**(-0.44),10**(-0.41),10**(-0.98),10**(-1.60)])
        T2_Arnaud=array([10**(-1.86),10**(-0.98),10**(-0.53),10**(-0.62),10**(-0.71)])
        
    
    elif filename=="iron_counts":
        counts_61=countdata.T[0][1:-1]
        counts_32=countdata.T[1][1:-1]
        counts_14=countdata.T[2][1:-1]

        T0_Arnaud=array([10**(-0.72),10**(-0.49),10**(-0.57),10**(-0.85),10**(-1.44),10**(-2.18),10**(-3.44)])
        T1_Arnaud=array([10**(-1.41),10**(-0.90),10**(-0.63),10**(-0.55),10**(-0.71),10**(-1.01),10**(-1.74),10**(-2.66)])
        T1_Arnaud=T1_Arnaud[:-1]
        T2_Arnaud=array([10**(-2.58),10**(-1.81),10**(-1.24),10**(-0.83),10**(-0.63),10**(-0.53),10**(-0.81)])
        
        T0_Arnaud_1992=array([10**(-1.68),10**(-0.51),10**(-0.49),10**(-0.64),10**(-1.01),10**(-1.66),10**(-2.93),10**(-4.44)])
        T1_Arnaud_1992=array([10**(-2.56),10**(-1.19),10**(-0.81),10**(-0.57),10**(-0.53),10**(-0.75),10**(-1.50),10**(-2.42)])

    fig, ax = plt.subplots(1,1,figsize=(figx_half, figy_half))
    if filename=="silicon_counts":
        charge_states=[7,8,9,10,11]
        barcolor="orange"
    elif filename=="magnesium_counts":    
        charge_states=[6,7,8,9,10]
        barcolor="cyan"
        if normalize_counts==True:
            counts_14=counts_14/sum(counts_14)
            ax.set_ylabel(r"$\rm{relative \ abundance}$",fontsize=fontsize)
            if plot_CSArnaud==True:
                ax.plot(charge_states,T0_Arnaud,linestyle="--",marker="o",color="g")
                ax.plot(charge_states,T1_Arnaud,linestyle="-",marker="o",color="b")
                ax.plot(charge_states,T2_Arnaud,linestyle="--",marker="o",color="gray")
        ax.bar(charge_states,counts_14,color=barcolor,label=r"$\rm{silicon}$")
        ax.set_ylabel(r"$\rm{measured \ counts }$",fontsize=fontsize)
        
    elif filename=="iron_counts":
        charge_states=[7,8,9,10,11,12,13]
        if iron_efficiencies_included==True:
            counts_14[0]=counts_14[0]*1/eff_Fe7
            counts_14[1]=counts_14[1]*1/eff_Fe8
            counts_14[2]=counts_14[2]*1/eff_Fe9
            counts_14[3]=counts_14[3]*1/eff_Fe10
            counts_14[4]=counts_14[4]*1/eff_Fe11
            counts_14[5]=counts_14[5]*1/eff_Fe12
            counts_14[6]=counts_14[6]*1/eff_Fe13
            #counts_14[7]=counts_14[7]*1/eff_Fe14
            
            counts_32[0]=counts_32[0]*1/eff_Fe7
            counts_32[1]=counts_32[1]*1/eff_Fe8
            counts_32[2]=counts_32[2]*1/eff_Fe9
            counts_32[3]=counts_32[3]*1/eff_Fe10
            counts_32[4]=counts_32[4]*1/eff_Fe11
            counts_32[5]=counts_32[5]*1/eff_Fe12
            counts_32[6]=counts_32[6]*1/eff_Fe13
            #counts_14[7]=counts_14[7]*1/eff_Fe14

            counts_61[0]=counts_61[0]*1/eff_Fe7
            counts_61[1]=counts_61[1]*1/eff_Fe8
            counts_61[2]=counts_61[2]*1/eff_Fe9
            counts_61[3]=counts_61[3]*1/eff_Fe10
            counts_61[4]=counts_61[4]*1/eff_Fe11
            counts_61[5]=counts_61[5]*1/eff_Fe12
            counts_61[6]=counts_61[6]*1/eff_Fe13
            #counts_14[7]=counts_14[7]*1/eff_Fe14

            
        if normalize_counts==True:
            counts_14=counts_14/sum(counts_14)
            counts_32=counts_32/sum(counts_32)
            counts_61=counts_61/sum(counts_61)
            ax.set_ylabel(r"$\rm{relative \ abundance}$",fontsize=fontsize)
        else:
            ax.set_ylabel(r"$\rm{measured \ counts }$",fontsize=fontsize)
        ax.bar(charge_states,counts_14,color="brown",alpha=1.,label=r"$\rm{iron}$")
        #ax.bar(charge_states,counts_32,color="gray",alpha=0.5,label=r"$\rm{iron}$")
        #ax.bar(charge_states,counts_61,color="beige",alpha=0.5,label=r"$\rm{iron}$")
        
        Counts_max=zeros((len(counts_14)))
        Counts_min=zeros((len(counts_14)))
        l=0
        while l<len(counts_14):
            counts_sigma=array([counts_14[l],counts_32[l],counts_61[l]])
            print(counts_sigma)
            
            counts_err=array([[abs(counts_14[l]-min(counts_sigma)),abs(max(counts_sigma)-counts_14[l])]]).T
            if normalize_counts==True:
                ax.errorbar(charge_states[l],counts_14[l],yerr=counts_err,linestyle="None",marker="o",color="k",elinewidth=elinewidth, capsize=capsize, capthick=capthick)
                print(counts_err)
                
            #Counts_max[j]=max(counts_sigma)
            #Counts_min[j]=min(counts_sigma)
            
            l+=1 
        #print Counts_min,Counts_max
        #Counts_err=array(zip(Counts_min,Counts_max)).T   
        #print Counts_err
        #ax.errorbar(charge_states,counts_14,yerr=Counts_err,linestyle="None",marker="o",color="m")
        ax.set_xticks([7, 8, 9, 10, 11, 12, 13])
        ax.set_xticklabels(["7", "8", "9", "10", "11", "12", "13"])
        
        if plot_CSArnaud==True:
            ax.plot(charge_states,T0_Arnaud,linestyle="--",marker="o",color="g")
            ax.plot(charge_states,T1_Arnaud,linestyle="-",marker="o",color="b")
            ax.plot(charge_states,T2_Arnaud,linestyle="--",marker="o",color="gray")
            #ax.plot(charge_states,T0_Arnaud_1992[:-1],linestyle="--",marker="o",color="b")
            
        
            print(sum(abs(T1_Arnaud-counts_14)))
    
    msc=average(charge_states,weights=counts_14)
    if normalize_counts==True:
        ax.text(.62,.75,r"$\langle \rm{Q} \rangle\ = \ %.2f$"%(msc), horizontalalignment='left',transform=ax.transAxes,fontsize=fontsize-2)
    elif normalize_counts==False:
        ax.text(.62,.75,r"$\langle \rm{Q^*} \rangle\ = \ %.2f$"%(msc), horizontalalignment='left',transform=ax.transAxes,fontsize=fontsize-2)
    
    leg=ax.legend(loc="upper right",ncol=1,prop={'size': legsize-2})        
    ax.set_xlabel(r"$\rm{charge \ state }$",fontsize=fontsize)        
    ax.tick_params(axis="x", labelsize=ticklabelsize)
    ax.tick_params(axis="y", labelsize=ticklabelsize)
    if Yrange!=None:
        ax.set_ylim(Yrange)        

    if figurtitle!=None:
        ax.set_title(r"$\rm{%s}$"%(figurtitle),fontsize=fontsize)

    if save_figure==True:
        plt.savefig(outfilename+".png",bbox_inches='tight')
        
    return sum(counts_14.T)









class All_Ionmeanvels(object):
    
    def __init__(self,proton_speeds=[385,395],ions_plot=["O6+","Si8+","Fe10+"], max_meanveldiff=15,include_systemerrors=False,ionlist_select=None):
        
        self.Diffspeed_mean=[]
        self.Diffspeed_sigma=[]
        self.Diffspeed_error=[]
        self.Zlow=[]
        self.Zmean=[]
        self.Zup=[]
        
        self.Vdiff_C5=[]
        self.Vdiff_O6=[]
        self.Vdiff_Ne8=[]
        self.Vdiff_Si7=[]
        self.Vdiff_Si8=[]
        self.Vdiff_Fe9=[]
        self.Vdiff_Fe10=[]
        
        self.Vdifferr_C5=[]
        self.Vdifferr_O6=[]
        self.Vdifferr_Ne8=[]
        self.Vdifferr_Si7=[]
        self.Vdifferr_Si8=[]
        self.Vdifferr_Fe9=[]
        self.Vdifferr_Fe10=[]
        
        for vp in proton_speeds:
            
            if vp<385:
                speedregime="Slow"
            elif (vp>=385) and (vp<475):
                speedregime="Inter"
            elif vp>=475:
                speedregime="Fast"
            
            I=Ionmeanvels(max_meanveldiff=max_meanveldiff,filepath="Veldists_moments_prefinal/%s_%i/Gaussmoyal/Full/Optimal/"%(speedregime,vp), elements=["carbon_valid","nitrogen_valid","oxygen_valid","neon_valid","magnesium_valid","silicon_valid","sulfur_valid","calcium_valid","iron_valid"],include_systemerrors=include_systemerrors,ionlist_select=ionlist_select)
            
            vp_width=10
            Nions_valid,diffspeed_mean,diffspeed_sigma,zlow,zmean,zup,vdiff_C5,vdifferr_C5,vdiff_O6,vdifferr_O6,vdiff_Ne8,vdifferr_Ne8,vdiff_Si7,vdifferr_Si7,vdiff_Si8,vdifferr_Si8,vdiff_Fe9,vdifferr_Fe9,vdiff_Fe10,vdifferr_Fe10, = I.plot_velmeans(vswfilter=[vp-vp_width/2.,vp+vp_width/2.],Yrange=[-125,65], plot_uncertainty_filter=False,save_figure=False, filename="test",plot_staterrors=True, plot_systemerrors=False,linefit=True,linefit_weights=True,plot_legend=False,plot_Zvalue_lower=True,plot_Zvalue_mean=True,plot_Zvalue_upper=True,plot_diffspeeds=True,figwidth="half",panel_label=None,plot_validcycles=False)

            self.Proton_speed=proton_speeds
            
            self.Diffspeed_mean.append(diffspeed_mean)
            self.Diffspeed_sigma.append(diffspeed_sigma)
            self.Diffspeed_error.append(diffspeed_sigma/sqrt(Nions_valid))
            
            self.Zmean.append(zmean)
            self.Zup.append(zup)
            self.Zlow.append(zlow)
    
            self.Vdiff_C5.append(vdiff_C5)
            self.Vdiff_O6.append(vdiff_O6)
            self.Vdiff_Ne8.append(vdiff_Ne8)
            self.Vdiff_Si7.append(vdiff_Si7)
            self.Vdiff_Si8.append(vdiff_Si8)
            self.Vdiff_Fe9.append(vdiff_Fe9)
            self.Vdiff_Fe10.append(vdiff_Fe10)
        
            self.Vdifferr_C5.append(vdifferr_C5)
            self.Vdifferr_O6.append(vdifferr_O6)
            self.Vdifferr_Ne8.append(vdifferr_Ne8)
            self.Vdifferr_Si7.append(vdifferr_Si7)
            self.Vdifferr_Si8.append(vdifferr_Si8)
            self.Vdifferr_Fe9.append(vdifferr_Fe9)
            self.Vdifferr_Fe10.append(vdifferr_Fe10)
        
        
            
        
        self.Vdifferr_C5=array(self.Vdifferr_C5).T
        self.Vdifferr_O6=array(self.Vdifferr_O6).T
        self.Vdifferr_Ne8=array(self.Vdifferr_Ne8).T
        self.Vdifferr_Si7=array(self.Vdifferr_Si7).T
        self.Vdifferr_Si8=array(self.Vdifferr_Si8).T
        self.Vdifferr_Fe9=array(self.Vdifferr_Fe9).T
        self.Vdifferr_Fe10=array(self.Vdifferr_Fe10).T

    
    
    def plot_diffspeed(self,hspace=0.1,save_figure=False,filename="test"):
        
        fig, axs = plt.subplots(2,1,figsize=(figx_full, figy_full+4),gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        fig.subplots_adjust(hspace=hspace)
        fontsize=20
        ticklabelsize=16
        legsize=20
        markersize=8
        elinewidth=2.5, 
        capsize=4
        capthick=2.5
    
        
        axs[0].plot([320,550],[5,5],linewidth=2,linestyle="--",color="k")
        axs[0].plot([320,550],[0,0],linewidth=2,linestyle="--",color="k")
        axs[0].plot([320,550],[-5,-5],linewidth=2,linestyle="--",color="k")
        
        
        
        axs[0].errorbar(self.Proton_speed,self.Vdiff_C5,yerr=self.Vdifferr_C5, marker="o", linewidth=2, color="b",label=r"$ \Delta \rm{v_{C5+} } $")
        axs[0].errorbar(self.Proton_speed,self.Vdiff_O6,yerr=self.Vdifferr_O6, marker="o", linewidth=2, color="r",label=r"$ \Delta \rm{v_{O6+} } $")
        axs[0].errorbar(self.Proton_speed,self.Vdiff_Ne8,yerr=self.Vdifferr_Ne8, marker="o", linewidth=2, color="lime",label=r"$ \Delta \rm{v_{Ne8+} } $")
        axs[0].errorbar(self.Proton_speed,self.Vdiff_Si7,yerr=self.Vdifferr_Si7, marker="o", linewidth=2, color="gold",label=r"$ \Delta \rm{v_{Si7+} } $")
        axs[0].errorbar(self.Proton_speed,self.Vdiff_Si8,yerr=self.Vdifferr_Si8, marker="o", linewidth=2, color="orange",label=r"$ \Delta \rm{v_{Si8+} } $")
        axs[0].errorbar(self.Proton_speed,self.Vdiff_Fe9,yerr=self.Vdifferr_Fe9, marker="o", linewidth=2, color="brown",label=r"$ \Delta \rm{v_{Fe9+} } $")
        axs[0].errorbar(self.Proton_speed,self.Vdiff_Fe10,yerr=self.Vdifferr_Fe10, marker="o", linewidth=2, color="sienna",label=r"$ \Delta \rm{v_{Fe10+} } $")
        axs[0].errorbar(self.Proton_speed,self.Diffspeed_mean,yerr=self.Diffspeed_error,marker="o", linewidth=2, color="k", label=r"$\langle \Delta \rm{ v_{ip}} \rangle $")
        
        
        leg=axs[0].legend(loc="upper left",ncol=3,prop={'size': legsize})
        #axs[0].set_xlabel(r"$\rm{proton \ mean \ speed \ [km/s]}$",fontsize=fontsize)
        axs[0].set_ylabel(r"$\rm{differential \ speed \ [km/s]}$",fontsize=fontsize)
        
        axs[0].tick_params(axis="x", labelsize=ticklabelsize)
        axs[0].tick_params(axis="y", labelsize=ticklabelsize)
    
        axs[0].set_xlim(320,550)
        axs[0].set_ylim(-18,40)

        
        
        #fig, ax = plt.subplots(1,1,figsize=(figx_full, 4))
        
        axs[1].plot([320,550],[0,0],linewidth=2,color="k")
        axs[1].plot([320,550],[-3,-3],linewidth=2,linestyle="--",color="k")
        axs[1].plot([320,550],[3,3],linewidth=2,linestyle="--",color="k")
        
        self.Diffspeed_mean=array(self.Diffspeed_mean)
        self.Zlow=array(self.Zlow)
        self.Zup=array(self.Zup)
        meandiff_mask_zup=self.Diffspeed_mean>5
        meandiff_mask_zlow=self.Diffspeed_mean<-5
        
        axs[1].plot(self.Proton_speed[meandiff_mask_zlow],self.Zlow[meandiff_mask_zlow],linewidth=2,marker="o",color="orange", label=r"$\rm{Z_{low} }$")
        axs[1].plot(self.Proton_speed,self.Zmean ,linewidth=2,marker="o",color="b", label=r"$\rm{Z_{mean} }$")
        axs[1].plot(self.Proton_speed[meandiff_mask_zup],self.Zup[meandiff_mask_zup],linewidth=2,marker="o",color="r", label=r"$\rm{Z_{up} }$")
        
        leg=axs[1].legend(loc="upper left",ncol=3,prop={'size': legsize})
        axs[1].set_xlabel(r"$\rm{proton \ mean \ speed \ [km/s]}$",fontsize=fontsize)
        #axs[1].set_ylabel(r"$\rm{ Z-value} \ [\sigma]}$",fontsize=fontsize)
        axs[1].set_ylabel(r"$\rm{ Z-value }$",fontsize=fontsize)
        
        
        axs[1].tick_params(axis="x", labelsize=ticklabelsize)
        axs[1].tick_params(axis="y", labelsize=ticklabelsize)
    
        axs[1].set_xlim(320,550)
        #ax.set_ylim(-15,40)

        if save_figure==True:
            plt.savefig(filename+".png",bbox_inches='tight')
