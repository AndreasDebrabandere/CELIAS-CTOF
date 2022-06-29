from numpy import *

#import time modules
import time 
import datetime as dt


def gauss2d(p,x,y):
    f=p[0]*exp(-(x-p[1])**2/(2*p[3]**2))*exp(-(y-p[2])**2/(2*p[4]**2))#all peak heights >= 0. 
    return f


def gauss_hyp_mask(ESSD_0,Esig,x,y):
    t0=time.process_time()
    #return alpha_PHD,mass
    #A0=0.468
    #A0=0.5098
    a0=2.00723e-1#time conversion in ns/ch 
    b0=-1.46909#time offset in ns		
    #return (A0*alpha_PHD*56*25750*(a0*x+b0)**(-2))
    #return exp(-0.5*((y-(A0*alpha_PHD*56*25750*(a0*x+b0)**(-2)))/a)**2)	
    #Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a0*x+b0)**-2#x0tofch	
    #return 0.5098*0.372*56*Enuc
    #print "test3"
    #return alpha_PHD
    #return a
    Epos=ESSD_0*(a0*x+b0)**-2
    t1=time.process_time()
    #print "gauss_hyp run time:", t1-t0
    #print "gauss1d param test:", [1.,Epos,Esig]
    M=exp(-0.5*((y-Epos)/Esig)**2)*(y-Epos)	
    Gmask=(M>=0)
    #return mask.astype(int)#for checks only
    return Gmask


def gauss_hyp_lm_tilt(ESSD_0,Esig,c1_tilt,c2_tilt,x,y):
    #return x
    t0=time.process_time()
    #return alpha_PHD,mass
    #A0=0.468
    #A0=0.5098
    a0=2.00723e-1#time conversion in ns/ch 
    b0=-1.46909#time offset in ns		
    #return (A0*alpha_PHD*56*25750*(a0*x+b0)**(-2))
    #return exp(-0.5*((y-(A0*alpha_PHD*56*25750*(a0*x+b0)**(-2)))/a)**2)	
    #Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a0*x+b0)**-2#x0tofch	
    #return 0.5098*0.372*56*Enuc
    #print "test3"
    #return alpha_PHD
    #return a
    Epos=c1_tilt*ESSD_0*(a0*x+b0)**-2+c2_tilt
   	#print "Epos_newmodel",ESSD_0*(a0*331.7+b0)**-2
    #return Epos
    t1=time.process_time()
    #print "gauss_hyp run time:", t1-t0
    #print "gauss1d param test:", [1.,Epos,Esig]
    #return y,Epos,Esig
    return exp(-0.5*((y-Epos)/Esig)**2)	



def kappa_moyalpar_hyp_2d(p,x,y):#function to put into lmfit method (see below) to fit peakheights
    t0=time.process_time()
    #print "p[5]",p[5]
    #return alpha_PHD,mass
    B=[p[0],p[1],p[2],p[3],p[4],p[5]]
    C=[p[6],p[7],p[8],p[9]]
    #return gauss_hyp_lm(ESSD_0=C[0],Esig=C[1],x=x,y=y)
    #return gauss_hyp_mask(ESSD_0=C[0],Esig=100*C[1],x=x,y=y)
    t00=time.process_time()
    hx=Kappa_Moyalpar(B,x)
    t01=time.process_time()
    hy=gauss_hyp_lm_tilt(ESSD_0=C[0],c1_tilt=C[2],c2_tilt=C[3],Esig=C[1],x=x,y=y)
    #return hx
    t02=time.process_time()
    #h=Kappa_Moyalpar(B,x)*gauss_hyp_lm_tilt(ESSD_0=C[0],c1_tilt=C[2],c2_tilt=C[3],Esig=C[1],x=x,y=y)
    h=hx*hy
    t1=time.process_time()
    #print "kappa_moyalpar_hyp_2d runtime",t1-t0,t01-t00,t02-t01,t1-t02
    return h 	


def Kappa_Moyalpar(p,x,FW_eq=0.5):#kappa_left_He2=1.5, 10 for all other elements
	#return x
	t0=time.process_time()
	h=float(p[0])
	#mu=float(find_nearest(x,p[1]))
	mu=float(p[1])
	#sigl=float(p[2])*convert_sigma(p[2],kappa_left,FW_eq)[-1]#early version include this conversion
	sigl=float(p[2])
	t01=time.process_time()
	#return p[2],sigl,convert_sigma(p[2],kappa_left,FW_eq)
	sigr=float(p[2])
	kappa_left=float(p[3])
	#print "kappa_left",kappa_left
	c=float(p[4])#high tof flank tail scale parameter
	tofch_min=float(p[5])
	#tofch_min=mu-2*p[2]#provisional for testing
	
	xmask_zero=(x<tofch_min)
	#print "tofch_min TOFpart only",tofch_min
	M2_zero=x[xmask_zero]*0
	
	xmask_low=(x>=tofch_min)*(x<mu)
	#z=arange(x[0],x[-1],1.)
	#z=arange(x[0],x[-1],0.01)
	z=arange(x[0],x[-1],0.1)
	M1=exp(-1/2.*(((z-mu)/(c*sigl))+exp(-((z-mu)/(sigl)))))#this is already the regular moyal function
	#return z,M1
	
	
	hmax=max(M1)
	#print "hmax zcheck",hmax
	t1=time.process_time()
	
	if hmax>0:
		indmax=where(M1==hmax)[0]
		zmax=z[indmax]
		zshift=mu-zmax
		#print indmax,zmax,zshift 
		M2_low=(1+(x[xmask_low]-mu)**2/(kappa_left*(sqrt(2)*sigl)**2))**(-kappa_left)
		
		xmask_high=(x>=mu)
		#M2_high=exp(-1/2.*(((x[xmask_high]-(mu+zshift))/(c*sigr))+exp(-((x[xmask_high]-(mu+zshift))/(sigr)))))
		M2_high=exp(-1/2.*(((x[xmask_high]-(mu))/(c*sigr))+exp(-((x[xmask_high]-(mu))/(sigr)))))#correct form!
		
		
		
		N_low=(1+(x[xmask_high][0]-mu)**2/(kappa_left*(sqrt(2)*sigl)**2))**(-kappa_left)
		N_high=float(M2_high[0])
		
		if len(M2_low)>0:
			#N2=max(M2_high)/float(max(M2_low))
			N=N_high/N_low
		else:
			N=1.
		#print "xmask test:", mu,x[xmask_low][-1],x[xmask_high][0],N_low,N_high,N
		#if x[xmask_low][-1]==x[xmask_high][0]:	
		#	M2=concatenate([M2_zero,M2_low[:-1],N2*M2_high])
		#else:
		#	M2=concatenate([M2_zero,M2_low,N2*M2_high])
		#N_overlap=len(x)-len(concatenate([M2_zero,M2_low,M2_high]))
		
		M2=concatenate([M2_zero,N*M2_low,M2_high])
		M3=h/float(max(M2))*M2
		#print len(x),len(M2_zero),len(M2_low),len(M2_high)
		#print len(x),len(M2),len((x*x[xmask_low])==mu),len((N2*M2_low)==M2_high[0])
		#print x[xmask_low],mu, len(where(xmask_low]==mu)>0)
	else:
		M3=zeros((len(x)))	
	t2=time.process_time()
	#print "kappa_moyalpar run times:",t2-t0,t01-t0,t1-t01,t2-t1
	#print "len(M2)",len(M2_zero),len(M2_low),len(M2_high)
	return M3


def kappa_Easym_hyp_lm_tilt(ESSD_0,Esig_low,Esig_up,Ekappa_low,Ekappa_up,c1_tilt,c2_tilt,x,y):    
    t0=time.process_time()
    #return alpha_PHD,mass
    #A0=0.468
    #A0=0.5098
    a0=2.00723e-1#time conversion in ns/ch 
    b0=-1.46909#time offset in ns		
    #return (A0*alpha_PHD*56*25750*(a0*x+b0)**(-2))
    #return exp(-0.5*((y-(A0*alpha_PHD*56*25750*(a0*x+b0)**(-2)))/a)**2)	
    #Enuc=(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))*(a0*x+b0)**-2#x0tofch	
    #return 0.5098*0.372*56*Enuc
    #print "test3"
    #return alpha_PHD
    #return a
    Epos=c1_tilt*ESSD_0*(a0*x+b0)**-2-2.0+c2_tilt#take out offset!
    #print "Epos_newmodel",ESSD_0*(a0*395.6+b0)**-2
    
    t1=time.process_time()
    #print "gauss_hyp run time:", t1-t0
    #return exp(-0.5*((y-Epos)/Esig)**2)	
    #print "gauss1d param test:", [1.,Epos,Esig_low,Esig_up]
    #print len(Epos),len(y)
    Gmask=gauss_hyp_mask(ESSD_0,Esig_up,x,y)
    Kup=(1+(y-Epos)**2/(Ekappa_up*(sqrt(2)*Esig_up)**2))**(-Ekappa_up)
    Klow=(1+(y-Epos)**2/(Ekappa_low*(sqrt(2)*Esig_low)**2))**(-Ekappa_low)
	  #Kup=exp(-0.5*((y-Epos)/Esig_up)**2)
    #Klow=exp(-0.5*((y-Epos)/Esig_low)**2)
    Kall=Kup*Gmask+Klow*(invert(Gmask))
    return Kall
    #return gauss1d_asym_array([1.,Epos,Esig_low,Esig_up],y)


def kappa_moyalpar_hyp_2d_Ekappa_asym(p,x,y):
    B=[p[0],p[1],p[2],p[3],p[4],p[5]]
    C=[p[6],p[7],p[8],p[9],p[10],p[11],p[12]]
    hx=Kappa_Moyalpar(B,x)
    hy=kappa_Easym_hyp_lm_tilt(ESSD_0=C[0],c1_tilt=C[5],c2_tilt=C[6],Esig_low=C[1],Esig_up=C[2],Ekappa_low=C[3],Ekappa_up=C[4],x=x,y=y)
    #print "length test hx,hy",len(hx),len(hy)
    #print "length test new", len(Kappa_Moyalpar(B,x)),len(kappa_Easym_hyp_lm_tilt(ESSD_0=C[0],c1_tilt=C[5],c2_tilt=C[6],Esig_low=C[1],Esig_up=C[2],Ekappa_low=C[3],Ekappa_up=C[4],x=x,y=y))
    #return Kappa_Moyalpar(B,x)*kappa_Easym_hyp_lm_tilt(ESSD_0=C[0],c1_tilt=C[5],c2_tilt=C[6],Esig_low=C[1],Esig_up=C[2],Ekappa_low=C[3],Ekappa_up=C[4],x=x,y=y)
    return hx*hy