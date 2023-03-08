'''
Quick file to shuffle jittertest files
'''


from reftable import *
rootfol= "./data/fakedata/jittertest3"

fols = ["/00-Hbetasource-B1-2971028700", "/01-CIVsource-A1-2940510474", "/02-Hbetasource-B6-2925858108", "/03-CIVsource-A7-2925523772"]
srcs = [source_B1, source_A1, source_B6, source_A7]

#Over-write with ozdes data
for dest in ["/dat-01","/dat-02"]:
    for fol in fols:
        for fol, src in zip(fols,srcs):

            targfol = rootfol + dest + fol
            #Load data from real source
            Tcs, Xcs, Ecs    = src.T_cont, src.X_cont, src.E_cont
            Tls1, Xls1, Els1 = src.T_line1, src.X_line1, src.E_line1
            Tls2, Xls2, Els2 = src.T_line2, src.X_line2, src.E_line2

            #Save to file
            np.savetxt(fname=targfol + "/cont.dat",     X=np.vstack([Tcs, Xcs, Ecs]).T)
            np.savetxt(fname=targfol + "/line1.dat",    X=np.vstack([Tls1, Xls1, Els1]).T)
            np.savetxt(fname=targfol + "/line2.dat",    X=np.vstack([Tls2, Xls2, Els2]).T)

            print("Saving source data to " + targfol)

#Shuffle y values of data
for dest in ["/dat-02","/sim-02"]:
    for fol in fols:
        for fol, src in zip(fols,srcs):

            targfol = rootfol + dest + fol

            #Load data from sources
            Tcs, Xcs, Ecs    = np.split(np.loadtxt(targfol + "/cont.dat"), 3, 1)
            Tls1, Xls1, Els1 = np.split(np.loadtxt(targfol + "/line1.dat"), 3, 1)
            Tls2, Xls2, Els2 = np.split(np.loadtxt(targfol + "/line2.dat"), 3, 1)

            #Shuffle em
            np.random.shuffle(Xcs)
            np.random.shuffle(Xls1)
            np.random.shuffle(Xls2)

            #Save to file
            np.savetxt(fname=targfol + "/cont.dat", X=np.hstack([Tcs, Xcs, Ecs]))
            np.savetxt(fname=targfol + "/line1.dat", X=np.hstack([Tls1, Xls1, Els1]))
            np.savetxt(fname=targfol + "/line2.dat", X=np.hstack([Tls2, Xls2, Els2]))

            print("Shuffling data in " + targfol)