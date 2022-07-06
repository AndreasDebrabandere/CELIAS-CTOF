from numpy import *
from scipy import constants

m_p=constants.m_p
e=constants.e

def convsecs70toDoY(t):
    from time import gmtime
    tmpt=gmtime(t)
    DoY=float(tmpt[7])+float(tmpt[3])/24.+float(tmpt[4])/(24.*60.)+float(tmpt[5])/(24.*60.*60.)
    return DoY

#Funktionen zur Umrechnung der ToF in m/q
def epq_step(step):
  """
  Returns energy per charge for ESA-step.
  step -> (list of) ESA-step(s)
  """
  Uo=0.331095e3     #lowest energy per charge
  r=1.040926        #scaling factor
  smax=116          #maximum step number
  epq=Uo*r**(smax-step)
  return epq  



def foil_loss(eq):
  """
  Returns parameter for Foil loss
  eq -> (list of) ESA-step(s)
  """
  return 9.12634e-6*eq**3-1.26741e-3*eq**2+4.55282e-2*eq-7.00777
  # Alt mit q
  #return -9.14732913e-09*eq**3+6.40784258e-07*eq**2-1.80375060e-05*eq-2.03481327e-04  #Parameter aus mq-tof-loop_neu.py

def mq_func(tof,epq,param):
  """
  Returns mq depending on ToF-channel, E/q, and foil loss
  tof -> (list of) ToF channel(s)
  epq -> (list of) E/q(s)
  param -> (list of) parameter(s) for Foil-loss
  """
  Un=22.69e3   # Post-acceleration voltage in KeV
  d=70.5e-3    # length of ctof ToF section in m
  
  return (5.78748e-7*tof+2.48478e-5)**2*2/d**2*(epq+Un)*(1-param)
  # Alt mit q
  #return 1./(d**2/((1.56558096094e-06*tof+7.75612861593e-05)**2*2*(epq+Un))+param)
  
def tof_to_mq(tof,step):
  """
  Returns mq depending on ToF-channel and ESA-step
  tof -> (list of) ToF channel(s)
  step -> (list of) ESA-step(s)
  """
  return mq_func(tof,epq_step(step),foil_loss(step))

def getionvel(mpc,step):
  """
  Returns ion velocity in km/s.
  mpc -> mass per charge of ion (He2+ -> 2., He1+ -> 1.)
  step -> (list of) ESA-step(s)
  """
  v=sqrt(epq_step(array(step))*2/(mpc*m_p/e))
  return v*1e-3

mpqarr=zeros([128,1024])
for step in range(128):
  for tof in range(1024):
    mpqarr[step,tof]=tof_to_mq(tof,step)

def hist_step(step):
  doubles[step,0:200]=0.
  x,y=histogram(mpqarr[step,:],linspace(0,56,231),weights=doubles[step,:])
  return x,y[:-1]+0.1


