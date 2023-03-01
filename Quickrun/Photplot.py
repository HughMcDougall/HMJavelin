import matplotlib.pyplot as plt
from Data_sim_Utils import *


dt=0.1
tmax=180*2*5
tau=400
T_season=180

T,Y = DRW_sim(dt, tmax, tau, siginf=1, x0=None, E0=0, method='square')

#PHOTOMETRIC DRAG
cad=8
Eav=4.3/100
Espread=1/100

TA,YA,EA    =   season_sample(T, Y, T_season, dt=cad, Eav=Eav, Espread=Espread, Emethod='gauss', garble=True, rand_space=True)
TB,YB,EB    =   season_sample(T, Y, T_season, dt=cad, Eav=Eav, Espread=Espread, Emethod='gauss', garble=True, rand_space=True)
TC,YC,EC    =   season_sample(T, Y, T_season, dt=cad, Eav=Eav, Espread=Espread, Emethod='gauss', garble=True, rand_space=True)

lag=200
blur=60
Yline = tophat_convolve(T,Y, tau, siginf=1, method='square',delay=lag,amp=1,width=blur, delay_from_center=True)
Yline_samp = season_sample(T, Y, T_season, dt=cad/4.33, Eav=Eav*10, Espread=Espread*10, Emethod='gauss', garble=True, rand_space=True)

delay1=40
delay2=80

plt.figure(figsize=(10,3))

plt.errorbar(TA,YA,EA, fmt='.',c='r')
plt.errorbar(TB+delay1,YB,EB, fmt='.',c='g')
plt.errorbar(TC+delay2,YC,EC, fmt='.',c='b')

tline=0
while tline<max(T):
    plt.axvline(tline,c='k',alpha=0.25,ls='--')
    tline+=180

plt.xlabel("Time (Days)")
plt.ylabel("Signal Arb Units")
plt.tight_layout()

plt.show()

#CONT AND RESPONSE LIGHTCURVE
fig,axs=plt.subplots(2,1,figsize=(10,5),sharex=True)
axs[0].plot(T,Y,c='b')
axs[1].plot(T,Yline,c='g')
axs[0].axvline(tmax/2,c='k',alpha=0.5)
axs[1].axvline(tmax/2+lag,c='k',alpha=0.5)

axs[1].set_xlabel("Time (Days)")
axs[1].set_ylabel("Signal Arb Units")
axs[0].set_ylabel("Signal Arb Units")

fig.tight_layout()
plt.show()


#WIREPLOT AND TFER FUNC
if True:
    tmax=500

    fig,axs=plt.subplots(2,1,figsize=(10,4),sharex=True,sharey=True)
    N=1024
    Ys=np.zeros([N,int(tmax//dt)+1])

    for i in range(N):
        T,Y= DRW_sim(dt, tmax, tau, siginf=1, x0=0, E0=0, method='square')

        Ys[i,:]=Y
        axs[1].plot(T,Y,c='k',lw=0.1,alpha=0.25)
    J=len(T)


    avs     = np.array([    np.average(Ys[:,j]) for j in range(len(T)) ] )
    sigmas  = np.array([    np.std(Ys[:,j])     for j in range(len(T)) ] )

    axs[1].plot(T,  avs,            c='r',  lw=2)
    axs[1].plot(T , avs- 2*sigmas,  c='r',  lw=2,   ls='--')
    axs[1].plot(T , avs+ 2*sigmas,  c='r',  lw=2,   ls='--')

    axs[0].plot(T,  avs,            c='r',  lw=2)
    axs[0].plot(T , avs- 2*sigmas,  c='r',  lw=2,   ls='--')
    axs[0].plot(T , avs+ 2*sigmas,  c='r',  lw=2,   ls='--')

    axs[0].plot(T,Y,c='k',lw=2)

    axs[1].set_xlabel("Time")

    fig.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)

    Y1=np.zeros(len(T))
    Y2=np.zeros(len(T))
    offset=tmax/2

    for i in range(len(T)):
        if abs(T[i]-lag-offset)<blur/2:Y2[i]=1

        if sum(Y1)==0 and T[i]>offset: Y1[i]=1

    axs[0].plot(T, Y1)
    axs[1].plot(T, Y2)
    axs[1].axvline(tmax / 2 + lag, c='k', alpha=0.5)
    fig.tight_layout()
    plt.show()


print("done")