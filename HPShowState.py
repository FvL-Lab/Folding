def showState(Sequence, Path, Dim, time=0, Temperature=0, Energy=0):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from time import sleep
    
    if plt.fignum_exists('HPFold'):
        plt.close(plt.figure('HPFold'))
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

    # if time==0: plt.show(block=False)
    # else: plt.draw()
    plt.show(block=False)
    sleep(0.01)
    return
