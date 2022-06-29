from numpy import *
from scipy import pi,sin,cos
#from scipy import stats
from Libsoho.libctof import getionvel
#from scipy import optimize
#from scipy.special import gamma,gammainc


"""
1b) Define specific instrument functions (from CTOF calibration)
"""
#define phase space correction for CTOF
def ps_correction(N,v,v_ref):#counts as integer, v in  km/s
    N_cor=N*(v_ref/v)# 1D-correction for solar wind ions!
    return N_cor
    
def tofch_to_ns(tof_ch):
    a=2.00723e-1
    b=-1.46909
    tof_ns=a*tof_ch+b
    return tof_ns	

def Epq(step):
    U_0=0.331095#in kV	
    r=1.040926	
    Epq=U_0*r**(116-step)
    return Epq#in keV/e

def step_to_E(step,q):#Energy before postacceleration	
    E=Epq(step)*q
    return E#in keV	


def calc_Esig(step,q):
    E=step_to_E(step)
    #print "Ion Energies:",E
    Esig=0.024*E
    return Esig

def step_to_E_acc(step,q):#Energy after postacceleration
    U_acc=23.85
    E_pc=(Epq(step)+U_acc)*q
    return E_pc

def E_to_v(E,m):#energy in keV, v in km/s
    return sqrt(2*(E*1.6*10**-16)/(m*1.66*10**-27))/1000.

def step_to_v_acc(step,q,m):#velocity after postacceleration
    U_acc=23.85
    E_pc=(Epq(step)+U_acc)*q
    v_pc=sqrt(2*(E_pc*1.6*10**-16)/(m*1.66*10**-27))/1000.#in km/s
    return v_pc
    
    
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

#define Hefti diffstream function for Fe9+:
def Hefti_O6(vmean_proton):
	return 1.13*vmean_proton-46
        
        
def Hefti_Si7(v_proton):
	return 0.92*(1.13*v_proton-46)+21

def Hefti_Fe9(v_proton):
	return 0.90*(1.13*v_proton-46)+28
				    
    
#start of new calibration ( including (1) 2D-Gauss Model and (2) 2D-Gausskappa Model ):

#2D-Gauss Model

#Individual ion calibration functions for the ion positions (in ToF and Energy): 

def tofch_cor(tofch):  
    """
    He2 tof position correction due to more accurate modelling of peak shape:
    """
    tofcor_grad=(-3.0-(-2.3))/(221.-241.5)#corrections observed for He2+ at steps 50 and 70, respectively
    tofcor_offset=-3.0-tofcor_grad*221.
    tofch_cor=tofcor_grad*tofch+tofcor_offset
    return tofch_cor

#Helium
def tof_He_uncor(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=4.0	
    A=5.88932130312
    B=17.7009155458
    C=0.0156280234352
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    return tofch

def tof_He(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=4.0	
    A=5.88932130312
    B=17.7009155458
    C=0.0156280234352
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    
    tofchcor=tofch_cor(tofch)
    
    #return tofch,tofch_cor
    #print("tofch_cor test",tofch_cor)
    #return tofch
    return tofch+tofchcor
		 	

#define ToF-dependent helium efficiencies:
def eff_He(xdata):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    a4=5.11732938203e-05#constants from fit of Helium after calibration with TRIM
    b4=-0.00292319019398#constants from fit of Helium after calibration with TRIM
    c4=0.0638742509087#constants from fit of Helium after calibration with TRIM
    d4=0.248060802039#constants from fit of Helium after calibration with TRIM
    Enuc_Hec=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*xdata+b)**-2
    #print(Enuc_Hec)			
    eff=a4*Enuc_Hec**3+b4*Enuc_Hec**2+c4*Enuc_Hec+d4
    #print("eff_He:")			
    #print(eff)
    return eff

def ESSD_He(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    ESSD_He=0.5098*eff_He(tof_ch)*4*Enuc
    
    #following correction due to more accurate modelling of peak shape:
    ESSD_He_cor=0.7*ESSD_He+5.5
    
    return ESSD_He_cor


#claibrated with He2, but in good agreement with iron too, and is therefore (currently) used for all ions
"""
def tofsig_left_He(tof_ch):
	tofsig_left_grad=(2.4-2.0)/(tof(step=70,m=4,q=2)-tof(step=40,m=4,q=2))
	tofsig_left_offset=2.4-tofsig_left_grad*tof(step=70,m=4,q=2)
	tofsig_left=tofsig_left_grad*tof_ch+tofsig_left_offset
	return tofsig_left
"""


#fits best with calibration of iron	
def tofsig_left_He(tof_ch):
	tofsig_left_grad=(2.4-2.1)/(tof(step=70,m=4,q=2)-tof(step=40,m=4,q=2))
	tofsig_left_offset=2.4-tofsig_left_grad*tof(step=70,m=4,q=2)
	tofsig_left=tofsig_left_grad*tof_ch+tofsig_left_offset
	return tofsig_left
			

#add ESSD

#Carbon
def tof_C(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=12.0	
    A=13.084899417
    B=21.9688211968
    C=0.0152375221992
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    return tofch

def ESSD_C(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.609*12*Enuc



#Nitrogen
def tof_N(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=14.0	
    A=13.1955015455
    B=14.6073929294
    C=0.0171410104879
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    return tofch

def ESSD_N(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.575*14*Enuc




#Oxygen
def Eres_rel_O(Eacc):
    A=17.3545360929
    B=45.9861325882
    C=0.0081593831844
    Eres=1.-(A/(Eacc+B)+C)
    return Eres


def tof_O(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=16.0	
    A=17.3545360929
    B=45.9861325882
    C=0.0081593831844
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    return tofch


def ESSD_O(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.542*16*Enuc



#Neon
def Eres_rel_Ne(Eacc):
    A=13.6266697519
    B=26.3409230996
    C=0.0158606370039
    Eres=1.-(A/(Eacc+B)+C)
    return Eres

def tof_Ne(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=20.0	
    A=13.6266697519
    B=26.3409230996
    C=0.0158606370039
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    return tofch

def ESSD_Ne(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.510*20*Enuc



#Magnesium
def Eres_rel_Mg(Eacc):
    A=8.23561596762
    B=6.92723675668
    C=0.0238165945195
    Eres=1.-(A/(Eacc+B)+C)
    return Eres

def Eres_rel_Mg(Eacc):
    A=8.23561596762
    B=6.92723675668
    C=0.0238165945195
    Eres=1.-(A/(Eacc+B)+C)
    return Eres


def tof_Mg(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=24.0
    A=8.23561596762
    B=6.92723675668
    C=0.0238165945195
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    return tofch

def ESSD_Mg(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.479*24*Enuc	



#Silicon
def Eres_rel_Si(Eacc):
    A=14.7154729103
    B=-7.05736444872
    C=0.0216687483439
    Eres=1.-(A/(Eacc+B)+C)
    return Eres


def tof_Si(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=28.0	
    A=14.7154729103
    B=-7.05736444872
    C=0.0216687483439
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    return tofch

def ESSD_Si(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.448*28*Enuc

    

#Sulfur
def tof_S(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=32.0
    A=13.8951103294
    B=1.12535827142
    C=0.0305098123556
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a
    return tofch
    
def ESSD_S(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.423*32*Enuc



#Calcium
#reperform TOF calibration for Calcium with TRIM

def ESSD_Ca(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.372*40*Enuc#only an approximation 



#Iron, 2D-Gauss
def Eres_rel_Fe(Eacc):
    A=12.2841240336
    B=-35.5347739111
    C=0.0186635244032
    Eres=1.-(A/(Eacc+B)+C)
    return Eres

def tof_Fe(Eacc):
    a=2.00723e-10
    b=-1.46909e-9
    m=56.0
    A=12.2841240336
    B=-35.5347739111
    C=0.0186635244032
    tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/((1-(A/(Eacc+B)+C))*Eacc*1.6*10**-16))	
    tofch=(tofsec-b)/a		
    return tofch

def ESSD_Fe(tof_ch):
    a=2.00723e-10#time conversion:tofch to tofs
    b=-1.46909e-9#time conversion:tofch to tofs		
    Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tof_ch+b)**-2	
    return 0.5098*0.296*56*Enuc


#ESSD-Width correction if the 'kappa_moyalpar_Easym'-model is used
 

#Corrections for the ion positions that have to be applied when the base-rate-corrected PHA data is used 

BRtofshift_He=0# for the He Priority range the BR reconstruction cannot be done properly, therefore no evaluation of the He2+ should be done (unless detailed alpha sensor rates are taken into account for the analysis). Thus, this neutral shift is selected as a dummy value. TODO: include this in the general description! 
BRtofshift_C=-2
BRtofshift_N=-1    
BRtofshift_O=-2    
BRtofshift_Ne=-3    
BRtofshift_Mg=-2    
BRtofshift_Si=-1    
BRtofshift_S=0    
BRtofshift_Ca=0    
BRtofshift_Fe=0    
BRtofshift_Ni=0    

BREfactor_He=1.# for the He Priority range the BR reconstruction cannot be done properly, therefore no evaluation of the He2+ should be done (unless detailed alpha sensor rates are taken into account for the analysis). Thus, this neutral factor is selected as a dummy value. TODO: include this in the general description!
BREfactor_C=39./42
BREfactor_N=39.2/42.
BREfactor_O=39.5/42
BREfactor_Ne=39.5/42
BREfactor_Mg=39.5/42
BREfactor_Si=40.5/42
BREfactor_S=40.7/42
BREfactor_Ca=40.82/42
BREfactor_Fe=1.
BREfactor_Ni=40.85/42

#General calibration functions for the ion positions based on the individual ion calibration functions, given above

def tof(step,m,q,tofpos_brcor=True,print_interpolation=False):#Calcium still missing!
    E_acc=step_to_E_acc(step,q)	
    if tofpos_brcor==True:
        if m==4: 
            tof_ch=tof_He(E_acc)+BRtofshift_He	
        elif m==12: 
            tof_ch=tof_C(E_acc)+BRtofshift_C
        elif m==14: 
            tof_ch=tof_N(E_acc)+BRtofshift_N	
        elif m==16: 
            tof_ch=tof_O(E_acc)+BRtofshift_O
        elif m==20:	
            tof_ch=tof_Ne(E_acc)+BRtofshift_Ne		
        elif m==24:	
            tof_ch=tof_Mg(E_acc)+BRtofshift_Mg
        elif m==28:	
            tof_ch=tof_Si(E_acc)+BRtofshift_Si
        elif m==32:	
            tof_ch=tof_S(E_acc)+BRtofshift_S
        elif m==56:	
            tof_ch=tof_Fe(E_acc)+BRtofshift_Fe		
        else:
            if m==40:
                tof_ch=interpol_tof(step,q,m)+BRtofshift_Ca
            if m==59:
                tof_ch=interpol_tof(step,q,m)+BRtofshift_Ni
                #TODO: add missing interpolated ions here!
            if print_interpolation==True:
                print("time-of-flight position is interpolated for this element!\nuncertainty can be calculated with method toferr_interpol!")
    else:
        if m==4: 
            tof_ch=tof_He(E_acc)	
        elif m==12: 
            tof_ch=tof_C(E_acc)
        elif m==14: 
            tof_ch=tof_N(E_acc)	
        elif m==16: 
            tof_ch=tof_O(E_acc)
        elif m==20:	
            tof_ch=tof_Ne(E_acc)		
        elif m==24:	
            tof_ch=tof_Mg(E_acc)
        elif m==28:	
            tof_ch=tof_Si(E_acc)
        elif m==32:	
            tof_ch=tof_S(E_acc)
        elif m==56:	
            tof_ch=tof_Fe(E_acc)		
        else:
            tof_ch=interpol_tof(step,q,m)
            if print_interpolation==True:
                print("time-of-flight position is interpolated for this element!\nuncertainty can be calculated with method toferr_interpol!")
    return tof_ch

def ESSD(tofch,m,Z,Epos_brcor=True,print_interpolation=False):#Helium and calcium still missing
    if Epos_brcor==True:
        if m==4:	
            ESSD_ch=ESSD_He(tofch)*BREfactor_He
        elif m==12:	
            ESSD_ch=ESSD_C(tofch)*BREfactor_C
        elif m==14:	
            ESSD_ch=ESSD_N(tofch)*BREfactor_N
        elif m==16:	
            ESSD_ch=ESSD_O(tofch)*BREfactor_O
        elif m==20:	
            ESSD_ch=ESSD_Ne(tofch)*BREfactor_Ne
        elif m==24:	
            ESSD_ch=ESSD_Mg(tofch)*BREfactor_Mg
        elif m==28:	
            ESSD_ch=ESSD_Si(tofch)*BREfactor_Si	
        elif m==32:	
            ESSD_ch=ESSD_S(tofch)*BREfactor_S	
        elif m==56:
            ESSD_ch=ESSD_Fe(tofch)*BREfactor_Fe
        else:
            if m==40:
                ESSD_ch=interpol_ESSD(tofch,m,Z)*BREfactor_Ca
            if m==59:
                ESSD_ch=interpol_ESSD(tofch,m,Z)*BREfactor_Ni
                #TODO: add missing interpolated ions here!
            if print_interpolation==True: 	
                print("residual energy position is interpolated linearly for this element!")
    else:
        if m==4:	
            ESSD_ch=ESSD_He(tofch)
        elif m==12:	
            ESSD_ch=ESSD_C(tofch)
        elif m==14:	
            ESSD_ch=ESSD_N(tofch)
        elif m==16:	
            ESSD_ch=ESSD_O(tofch)
        elif m==20:	
            ESSD_ch=ESSD_Ne(tofch)
        elif m==24:	
            ESSD_ch=ESSD_Mg(tofch)
        elif m==28:	
            ESSD_ch=ESSD_Si(tofch)	
        elif m==32:	
            ESSD_ch=ESSD_S(tofch)	
        elif m==56:
            ESSD_ch=ESSD_Fe(tofch)
        else:
            ESSD_ch=interpol_ESSD(tofch,m,Z)
            if print_interpolation==True: 	
                print("residual energy position is interpolated linearly for this element!")            
    return ESSD_ch


def phd(m):#Pulse Height Defect factor in dependance of mass, derived from SSD calibration 
    if m==4:	
        alpha_PHD=0.650#only approximation here, which should be sufficient for the peakshape (small tof width of He!), for true tod-dependent value see function: "eff_He"
    elif m==12:	
        alpha_PHD=0.609
    elif m==14:	
        alpha_PHD=0.575
    elif m==16:	
        alpha_PHD=0.542
    elif m==20:	
        alpha_PHD=0.510
    elif m==24:	
        alpha_PHD=0.479
    elif m==28:	
        alpha_PHD=0.448	
    elif m==32:	
        alpha_PHD=0.423	
    elif m==56:
        alpha_PHD=0.296
    else: 	
        print("No Pulse Height Defect factor prediction available for this element!")
    return alpha_PHD
    

def interpol_tof(step,q,m):
		#print "test"
		Eacc=step_to_E_acc(step=step,q=q)
		Eres_rel=Eres_rel_Fe(Eacc)#iron relative eenergy loss is approximately in between all calibrated elements
		#return Eres_rel

		a=2.00723e-10
		b=-1.46909e-9
		tofsec=sqrt(0.5*m*1.66*10**-27*(70.5*10**-3)**2/(Eres_rel*Eacc*1.6*10**-16))	
		tofch=(tofsec-b)/a		
		return tofch

def toferr_interpol(step,q,m):
		a=2.00723e-10#time conversion:tofch to tofs
		M=1.66e-27*m
		L=70.5e-3
		E=step_to_E_acc(step=step,q=q)*1.61e-16#in J
		#return E
		toferr=sqrt(M*L**2/(8.*E))*0.01#1 percent is estimated tof position error, when we interpolate \Delta E/E (=relative energy loss in foil)
		toferr_ch=1./a*toferr
		return toferr_ch

def interpol_ESSD(tofch,m,Z):
		a=2.00723e-10#time conversion:tofch to tofs
		b=-1.46909e-9#time conversion:tofch to tofs		
		Enuc=(0.5*1.66e-27*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a*tofch+b)**-2	
		phd_rel=interpol_phd(Z,tofch)[0]#phd is constant for all elements except for helium, therefore first array entry taken
		return 0.5098*phd_rel*m*Enuc



		  
#General calibration functions for the ion widths


def tofsig(step,m,q):
    Eacc=step_to_E_acc(step,q)
    vacc=E_to_v(Eacc,m)
    return -0.0040*vacc+10.28#from fitted widths of reference ions, valid for all ions with box rate method

def Esig(ESSD):
    return 0.1024*ESSD+2.368#preliminary#from fitted widths of reference ions, valid for all ions with box rate method

def Esig_iron_low(ESSDch):
		#return ESSDch*0+5.
		return (9.5-7.)/(46-28)*ESSDch

def Esig_iron_up(ESSDch):
		#sigE_grad=(6.0-9.5)/(500.-360.)
		#sigE_off=18.5
		#return sigE_grad*tofch+sigE_off	
		#return ESSDch*0+30.#test	
		#return ESSDch*0+8.		
		return (10.-7.)/(46-28)*ESSDch+2.5

def Esig_kappa_moyalpar(ESSD):#modification of Ewidth for Kappa-Moyal peakshape model, derived empirically from long-term fits of the whole (base-rate-corrected) ET-matrix
    Esig_Gauss=Esig(ESSD)
    return 0.85*Esig_Gauss

#(1) 2D-Gausskappa Model, preliminary, very accurate for Iron  
        
def tofsig_left(step,m,q):
    vacc=step_to_v_acc(step,q,m)
    mask_const=vacc>=1100
    mask_slope=vacc<1100
    tofsig_left_0=0*vacc[mask_const]+3.0
    tofsig_left_1=-0.010*vacc[mask_slope]+14.1
    tofsig_left=concatenate([tofsig_left_0,tofsig_left_1])
    return tofsig_left
        
def tofsig_right(step,m,q):
    vacc=step_to_v_acc(step,q,m)
    mask_const=vacc>=1140
    mask_slope=vacc<1140
    tofsig_right_0=0*vacc[mask_const]+7.0
    tofsig_right_1=-0.03125*vacc[mask_slope]+42.5
    tofsig_right=concatenate([tofsig_right_0,tofsig_right_1])
    return tofsig_right

#def tofkappa_interpol(step):
#	"""
#	old version
#	"""
#    return 0*step+1.2#kappaslope_0,from master thesis NJ
    
def tofkappa_interpol(mass,step):
	#return 0*step+1.2#kappaslope_0,from master thesis NJ
	
	tofkappa_light=100.
	tofkappa_intermediate=0.8
	tofkappa_heavy=0.5
	
	if mass==56:
		tofkappa=0*step+tofkappa_heavy
	elif mass==4:
		tofkappa=0*step+tofkappa_light
	else:	
		tofkappa=0*step+tofkappa_intermediate
		
	return tofkappa


def tofkappa_tofdep(tofchs):
	kappas=zeros((len(tofchs)))
	i=0
	for tofch in tofchs:
		#tofkappa=amax(array([8.0-0.02*tofch,array([0.01]*len(tofch))]),axis=0)#test
		if tofch<240:
			tofkappa=2.0
		elif (tofch>=240)*(tofch<320):
			tofkappa=-(2-0.8)/80.*tofch+5.6
		elif (tofch>=320)*(tofch<=1000):
			tofkappa=-(0.8-0.3)/80.*tofch+2.8
			#tofkappa=amax(array([-(0.8-0.)/80.*tofch+4.0,0.001]))
		#elif (tofch>=400):
			tofkappa=amax(array([-(0.3-0.0)/70.*tofch+2.0142857,0.01]))
		kappas[i]=tofkappa
		i+=1
	return kappas

def kappa_linear(tofch,tofrange,kapparange):
	"""
	is not allowed to become negative before tofch=550 (for step 116), or at least tofch 533 (for step90)
	"""
	kgrad=(kapparange[-1]-kapparange[0])/float(tofrange[-1]-tofrange[0])
	koff=kapparange[0]-kgrad*tofrange[0]
	kappa=kgrad*tofch+koff
	return kappa

"""
def kappa_linear(tofch,kgrad,koff):
	kappa=kgrad*tofch+koff
	return kappa
"""



def kappa_exp(tofrange,kappakoef):
	"""
	meaningful range for kappakoef[1]: kexp: [-0.1,-20]
	meaningful range for kappakoef[0]: kfact: [1,10]
	"""
	kfact=kappakoef[0]
	kexp=kappakoef[1]/1000.
	kappa=kfact*exp(kexp*(tofrange))*1./(exp(kexp*tofrange[0]))
	return kappa



def tofsigr_rel_linear(tofch,tofrange,sigrel_range):
	"""
	is not allowed to become negative before tofch=550 (for step 116), or at least tofch 533 (for step90)
	"""
	sigrel_grad=(sigrel_range[-1]-sigrel_range[0])/float(tofrange[-1]-tofrange[0])
	sigrel_off=sigrel_range[0]-sigrel_grad*tofrange[0]
	sigrel=sigrel_grad*tofch+sigrel_off
	return sigrel

"""
def tailscale_linear(tofch,tofrange,tailrange):
  tail_grad=(tailrange[-1]-tailrange[0])/float(tofrange[-1]-tofrange[0])
  tail_offset=tailrange[0]-tail_grad*tofrange[0]
  return tail_grad*tofch+tail_offset  
"""

def tailscale_linear(tofch,tail_grad,tail_offset):
	return tail_grad*tofch+tail_offset  
    
#def tailscale_exp(tofch,tail_fact,tail_exp):
#  return tail_fact*exp(tail_exp*tofch)  

def tailscale_exp(tofch,tofrange,tailrange):
  tail_exp=(log(tailrange[-1])-log(tailrange[0]))/float((tofrange[-1]-tofrange[0]))
  tail_fact=tailrange[0]/exp(tail_exp*tofrange[0])
  print(tail_fact,tail_exp)
  return tail_fact*exp(tail_exp*tofch)  

#New calibration ends here!
"""    
1b.1) interpolation of all other ions
"""    

def interpol_phd(Z,tofch,Epos_brcor=True):
    """
    interpolates linearly (relaitve) pulse height defects for elements 4<=Z<=26 (and extrapolates Z=27,28) from calibration elements, C,O,Si,Fe
    """
    #define interpolation:
    def alpha_He(Z):
        alpha_He=0*Z+0.650#only approximation here, which should be sufficient for the peakshape (small tof width of He!), for true tod-dependent value see function: "eff_He"
        return alpha_He

    def alpha_CO(Z): 
        CO_grad=(0.609-0.542)/(6.-8.)
        CO_off=0.609-6.*CO_grad
        return CO_grad*Z+CO_off

    def alpha_OSi(Z): 
        OSi_grad=(0.542-0.448)/(8.-14.)
        OSi_off=0.542-8.*OSi_grad
        return OSi_grad*Z+OSi_off

    def alpha_SiFe(Z): 
        SiFe_grad=(0.448-0.296)/(14.-26.)
        SiFe_off=0.448-14.*SiFe_grad
        return SiFe_grad*Z+SiFe_off

    #interpolate z-dependent
    if Z==2:
        #phd=eff_He(tofch)
        phd=array([alpha_He(Z)]*len(tofch))#preliminary
        
    if len(shape(tofch))!=0:
        if Z==2:
            pass #preliminary (see above)

        elif (Z>=6) and (Z<8):
            phd=array([alpha_CO(Z)]*len(tofch))

        elif (Z>=8) and (Z<14):
            phd=array([alpha_OSi(Z)]*len(tofch))

        elif (Z>=14) and (Z<=28):#change to 26, after nickel case is calculated with TRIM
            phd=array([alpha_SiFe(Z)]*len(tofch))

        else:
            print("no residual energy (ESSD) interpolation possible for this element!")

    else:
        if Z==2:
            pass #preliminary (see above)
        
        if (Z>=4) and (Z<6):
            phd=array([alpha_CO(Z)])

        elif (Z>=6) and (Z<14):
            phd=array([alpha_OSi(Z)])

        elif (Z>=14) and (Z<=28):#change to 26, after nickel case is calculated with TRIM
            phd=array([alpha_SiFe(Z)])

        else:
            print("no residual energy (ESSD) interpolation possible for this element!")

    if Epos_brcor==True:
        if Z==2:	
            phd=phd*BREfactor_He
        elif Z==6:	
            phd=phd*BREfactor_C
        elif Z==7:	
            phd=phd*BREfactor_N
        elif Z==8:	
            phd=phd*BREfactor_O
        elif Z==10:	
            phd=phd*BREfactor_Ne
        elif Z==12:	
            phd=phd*BREfactor_Mg
        elif Z==14:	
            phd=phd*BREfactor_Si	
        elif Z==16:	
            phd=phd*BREfactor_S	
        elif Z==20:
                phd=phd*BREfactor_Ca
        elif Z==26:
            phd=phd*BREfactor_Fe
        elif Z==28:
                phd=phd*BREfactor_Ni
    return phd

"""	
further CTOF-related instrumental functions
"""	
def calc_tofmin(m,q,step):
	Eacc=step_to_E_acc(step,q)*1.6e-16
	T_s=sqrt(0.5*m*1.66e-27*(70e-3)**2/Eacc)
	#return T_s 
	
	a=2.00723e-10#time conversion:tofch to tofs
	b=-1.46909e-9#time conversion:tofch to tofs		  
	T_ch=(T_s-b)/a
	return T_ch 
	
	