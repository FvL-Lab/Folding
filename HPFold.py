def HPFold(N=20, Dim=2, Time=0, Temperature=0.5, Interaction=[-1,-0.1,-0.01], mode=12):
#--------------------------------------------------------------------------
# HPFold.py  -  Monte-Carlo simulation of a simple 2D HP=folding problem
#
# Call  [Seqence,minE,minF,Energy,Interaction] = 
#                              HPFold(N,Time,Temperature,Interaction,mode)
#
# Input    N           -  length of a random sequence (def:20) or
#                          sequence as binary vector with 1:=H, 0:=P
#                          for N<0 an N-times a-helix is produced
#          Dim         -  dimension of the model 2/3 (def:2)
#          Time        -  length of the simulation (def: 100*N)
#          Temperature -  initial temperature (see mode; def:0.5*E_HH)
#          Interaction -  interaction energies [E_HH, E_HP, E_PP]
#                          (def:[-1,0,0])
#          mode        -  (def:12)
#                         x0 - no intermediate graphics output
#                         x1 - intermediate graphics output
#                         x2 - extended intermediate graphics output
#                         0x - keep system at fixed 'Temperature'
#                         1x - stepwise cooldown starting from
#                              'Temperature'
#                         2x - StatPhys mode: decrease Temperature in 10
#                              steps
#
#
# Output   Sequence    -  the sequence used
#          minE        -  minimum energy
#          minF        -  folding sequence at minE
#                         the folding sequence is a vector of N-1 complex
#                         numbers representing the direction of the step
#                         between unit i and i+1 in the sequence. The
#                         direction in x is represented by the real, and in
#                         y by the imaginary part of the folding sequence.
#          Energy      -  development of the energy during the simulation
#          Interaction -  values for the three interaction energies used
#                          [E_HH, E_HP, E_PP]
#
#
# reference: Lau, K. F., and K. A. Dill. 1989. 
#            "A lattice statistical mechanics model of the conformational 
#            and sequence spaces of proteins." 
#            Macromolecules 22:3986-3997.
#
# a-Helix: [-H-H-P-P-H-H-P-]_n = numpy.tile([1,1,0,0,1,1,0],n)
# b-Sheet: [-H-P-]_n = numpy.tile([1,0],n)
#
#
# date: 11 Oct 2016
# author: ts
# version: <00.00> ts <20161012.0000>
# <00.10> ts <20161026.0000>
# <01.00> ts <20161028.0000> - Python version
#--------------------------------------------------------------------------
    import numpy as np
    from time import sleep
    
    #set some internal variables & interpret any input
    if N==None: N=20
    if Dim==None: Dim=2
    if Time==None: Time=0
    if Temperature==None: Temperature=0.5
    if Interaction==None: Interaction=[-1,-0.1,-0.01]        
    if np.size(N)==1:
        if N<0: #N times alpha-helix
            N = int(max(3,-N))
            Sequence = np.tile([1,1,0,0,1,1,0],N)
        else: #random sequence of length N
            N = int(max(5,N))
            Sequence = np.random.randint(2,size=N,dtype=int)
            Sequence[0] = 0
    else: #sequence as given
        Sequence = N
    N = np.size(Sequence)
    Dim = int(max(2,min(3,Dim)))
    Time = int(max(1000*N,Time))

    #internal convenience variables
    gMode = mode % 10                    #graphics mode
    pMode = int(np.floor(mode/10) % 10)  #program mode
    tShow = 10**(max(0,np.ceil(np.log10(Time))-2)) 
    if pMode==1:
        dtime = 1 
        dTemp = (Temperature-0.1*min(np.abs(Interaction)))/Time
    elif pMode==2:
        Time = min(Time,10000*N)
        dtime = Time/11
        dTemp = (Temperature-0.1*min(np.abs(Interaction)))/10
    else: 
        dtime = 1 
        dTemp = 0
    rowO  = np.ones((1,N)) #a row of ones
    colO  = np.ones((N,1)) #a colum of ones
    diaI  = np.diagflat(2*colO)
    
    #calculate masks for HH, HP, and PP interactions
    HH     = np.triu((colO*(Sequence==1))*((Sequence==1)[:,np.newaxis]*rowO),2)
    HP     = np.triu((colO*(Sequence==1))*((Sequence==0)[:,np.newaxis]*rowO) +
                     (colO*(Sequence==0))*((Sequence==1)[:,np.newaxis]*rowO),2)
    PP     = np.triu((colO*(Sequence==0))*((Sequence==0)[:,np.newaxis]*rowO),2)
    
    #start with a protein stretched in x
    #"Fold" contains the direction on a lattice to get to the next amino acid
    Fold   = np.hstack((np.ones((N,1)),np.zeros((N,Dim-1))))
    Fold[0,0] = 0
    Energy = np.zeros((Time+1,4))
    minE   = Energy[0,1]
    minF   = np.array(Fold)
    
    showState(Sequence,np.cumsum(minF,axis=0),Dim,0,0,minE)
    sleep(1)

    #--------------------------------------------------------------------------
    #the time-loop
    for iTime in range(1,Time+1):
        #iT = iTime+1;

        #randomly choose one element (amino acid) to change
        iFold = np.random.randint(low=2,high=N);

        #do a step
        Prob = -1;
        while Prob<np.random.rand(): #Metropolis algorithm to accept the last step 
            Fold[iFold,:] = nrand(Dim,Fold[iFold-1,:]); #get a new step, ie configuration
            Path = np.cumsum(Fold,axis=0); #calculate the new configuration
            Dist = np.array(diaI);
            for iDim in range(Dim): #calculate the Euklidian distance matrix
                Dist = Dist + (colO*Path[:,iDim]-Path[:,iDim][:,np.newaxis]*rowO)**2;
            #print(iTime, iFold, Fold[iFold,:], Dist.min())
            if Dist.min()==0: #overlap in position: try a different fold
                Prob = -1;
            else:
                T = max(0,Temperature-np.floor(iTime/dtime)*dTemp);
                Dist = Dist==1; #distances that are == 1
                E = Interaction[0]*HH*Dist + Interaction[1]*HP*Dist + Interaction[2]*PP*Dist #the interaction energy as dependent on the HH, HP, and PP interaction
                Energy[iTime,0] = iTime;
                Energy[iTime,1] = E.sum()
                Energy[iTime,2] = T;
                Energy[iTime,3] = np.var(Path, axis=0).sum();
                if Energy[iTime,1]<=minE: #keep track of the minimal energy and it's fold
                    minE = Energy[iTime,1];
                    minF = np.array(Fold);
                Prob = np.exp(-(Energy[iTime,1]-Energy[iTime-1,1])/T);
                #fprintf(1,'t=%d, E=%d, p=%.1f\n',iTime,Energy(iT,2),Prob);
        
        if (gMode>0) & (np.mod(iTime,tShow)==0):
             showState(Sequence,Path,Dim,iTime,T,Energy[iTime,1]);

    #--------------------------------------------------------------------------
    #report the result
    sleep(1)
    print("The sequence ", Sequence, " of length %d,"%N)
    print("containing %d H-elements"%np.sum(Sequence),"folded in %d steps"%Time)
    print("to an energy of %.1f."%minE, "\n");
    #show the configuration of lowest energy found
    showState(Sequence,np.cumsum(minF, axis=0),Dim,Time,0,minE)
    if gMode>1: ShowTimeCourse(Energy)
    return Sequence, minE, minF, Energy, Interaction

#=== subroutines ============================================================
#function generates a random step different to the one given
def nrand(Dim, Smin1=0): 
    import numpy as np
    s = -np.array(Smin1);
    while np.sum((s+Smin1)**2)==0: #do not fold back
        s = np.zeros((1,Dim))
        s[0,0] = 1
        s = (2*np.random.randint(2)-1) * np.roll(s,np.random.randint(Dim));
    return s

#--------------------------------------------------------------------------
#shows the given configuartion and some additional info
def showState(Sequence, Path, Dim, time=0, Temperature=0, Energy=0):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from time import sleep
    
    hFig = plt.figure('HPFold')
    plt.clf()
    sqN  = np.ceil((len(Sequence))**(1/Dim))
    Path = Path - np.mean(Path, axis=0)

    if Dim==2:
        hAx = hFig.add_subplot(111)
        hAx.plot(Path[:,0],Path[:,1],'-k',linewidth=2, gid='Path')
        hAx.plot(Path[np.where(Sequence==0),0],Path[np.where(Sequence==0),1],
                 'ob',markersize=12,gid='H')
        hAx.plot(Path[np.where(Sequence==1),0],Path[np.where(Sequence==1),1],
                 'or', markersize=12, gid='P')
        hAx.plot(Path[0,0],Path[0,1],'og',markersize=8, gid='0') 
        hAx.axis((np.tile([-sqN,+sqN],(1,Dim)))[0,:])
        hAx.text(sqN*(0.05-1),sqN*0.9,"T = %.2f"%Temperature)
        hAx.text(sqN*(0.05-1),sqN*0.8,"E = %.2f"%Energy)
        hAx.text(sqN*(0.05-1),sqN*0.7,"t = %d"%time)
    else:
        hAx3D = hFig.add_subplot(111, projection='3d')
        hAx3D.plot3D(Path[:,0],Path[:,1],Path[:,2],'-k',linewidth=2)
        hAx3D.plot3D(Path[np.where(Sequence==0),0][0,:],
                     Path[np.where(Sequence==0),1][0,:],
                     Path[np.where(Sequence==0),2][0,:],
                     'ob',markersize=12)
        hAx3D.plot3D(Path[np.where(Sequence==1),0][0,:],
                     Path[np.where(Sequence==1),1][0,:],
                     Path[np.where(Sequence==1),2][0,:],
                     'or', markersize=10)
        hAx3D.plot3D([Path[0,0],Path[0,0]],[Path[0,1],Path[0,1]],
                     [Path[0,2],Path[0,2]],'og',markersize=8) 
        hAx3D.set_xlim3d(-sqN, +sqN)
        hAx3D.set_ylim3d(-sqN, +sqN)
        hAx3D.set_zlim3d(-sqN, +sqN)
        hAx3D.text3D(sqN*(0.05-1),-sqN*0.5,sqN*0.90,"T = %.2f"%Temperature)
        hAx3D.text3D(sqN*(0.05-1),-sqN*0.5,sqN*0.75,"E = %.2f"%Energy)
        hAx3D.text3D(sqN*(0.05-1),-sqN*0.5,sqN*0.60,"t = %d"%time)

    plt.pause(0.05)
    return

#--------------------------------------------------------------------------
#shows the final timecourse of E, radius of gyration, Cv
def ShowTimeCourse(E):
    import numpy as np
    import matplotlib.pyplot as plt
    
    hFig = plt.figure('HPFold - result')
    plt.clf()
    
    N = int(np.ceil(len(E[:,1])/50));
    
    hAx1 = hFig.add_subplot(221)
    hAx1.plot(E[:,0],E[:,2])
    #hAx1.axis([E[1,1],E[end,1],0,1.05*max(E[:,3])])
    hAx1.set_xlabel('time')
    hAx1.set_ylabel('temperature')
 
    hAx2 = hFig.add_subplot(222)
    hAx2.plot(E[:,0],E[:,3])
    #hAx2.axis([E(1,1),E(end,1),0,mean(E(:,4))^2])
    hAx2.set_xlabel('time')
    hAx2.set_ylabel('radius of gyration')
 
    hAx3 = hFig.add_subplot(223)
    hAx3.plot(E[:,0],E[:,1])
    hAx3.set_xlabel('time')
    hAx3.set_ylabel('energy')

    hAx4 = hFig.add_subplot(224)
    hAx4.plot(E[1:,0],np.diff(np.convolve(E[:,1],np.ones(N)/N,mode='same')))
    hAx4.set_xlabel('time')
    hAx4.set_ylabel('specific heat')
    
    plt.show(block=False)
    plt.pause(0.05)
    return


#--------------------------------------------------------------------------
#run the program
HPFold()

