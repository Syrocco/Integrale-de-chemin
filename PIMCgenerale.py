import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from numba import jit
import scipy.integrate as scp



#####################
######FONCTIONS######
#####################


#Carré du module du fondamental de la fonction d'onde d'une particule dans un potentiel harmonique
def gauss(x,w=1):
    return np.sqrt(w/np.pi)*np.exp(-w*x**2)  




#Différents potentiels
@jit(nopython=True,cache=True) 
def harmonicOscillator(x,w=1):
    return 0.5*(x*w)**2

@jit(nopython=True,cache=True) 
def quartOscillator(x):
    return 0.5*x**2+0.125*x**4

@jit(nopython=True,cache=True)
def doubleGaussian(x):
    return -3*x**2*np.exp(-0.5*x**2)

@jit(nopython=True,cache=True)
def atomic(x):
    return 12/(0.2*np.sqrt(np.pi))*np.exp(-(x/0.2)**2)-12/(0.8*np.sqrt(np.pi))*np.exp(-(x/0.8)**2)

@jit(nopython=True,cache=True)
def free(x):
    return 0
      
@jit(nopython=True,cache=True)
def harmonicInterrac(x1,x2):
    return 0.25*(x1-x2)**2  #mettre 0.25 plutot que 0.5 si il y a deux particules pour obtenir la bonne fonction d'onde. Il suffit de changer w dans gauss() pour avoir la bonne fonction d'onde avec un coeff diff de 0.25.

@jit(nopython=True,cache=True)
def gaussianWell(x):
    return -2.5*np.exp(-0.5*x**2)






#Génère i chemins allant de xa jusqu'à xb avec un pas maximum de dxMax
def path(size,xa,xb,i,M,dxMax=5):
    for j in range(i):
    	A=rd.randint(-dxMax,dxMax+1,size)
    	sumA=np.sum(A)
    	diff=xb-xa
    	if sumA>diff:           #permet d'avoir les bonnes conditions aux bords
    		for i in range(int(sumA-diff)):
    			randomIndex=rd.randint(1,len(A))
    			A[randomIndex]=A[randomIndex]-1
    	if sumA<diff:
    		for i in range(int(-sumA+diff)):
    			randomIndex=rd.randint(1,len(A))
    			A[randomIndex]=A[randomIndex]+1
    	randomIndex=rd.randint(1,len(A)-1,2)
    	A[randomIndex[0]]=A[randomIndex[0]]+A[0]
    	A[0]=0
    	path=np.zeros(len(A))
    	path[0]=xa
    	path[-1]=xb
    	for i in range(1,len(A)-1):
    		path[i]=path[i-1]+A[i]
    	if np.max(path)>=M or np.min(path)<=-M: #Au final la discretisation selon les x importe peu mais je laisse cette condition pour savoir si mon chemin part trop loin...
    		raise Exception('Le chemin sort de la grille prédéfinie, baissez le dxMax ou augmentez la taille de la grille')
    	else:
            if j==0:
                pathFinal=np.array([path])
            else:
                pathFinal=np.concatenate((pathFinal,np.array([path])),axis=0)
                
    return pathFinal
    





#Calcul de l'énergie d'un chemin. On différencie le cas où il y a une particule et le cas où il y en a plus: (len(pathP[0]) donne le nombre de particules).
@jit(nopython=True,cache=True) 	
def energie(pathP,dt):
    E=0
    if len(pathP)==1:
        for j in range(1,len(pathP[0])-1):
            E+=0.5*((pathP[0,j]-pathP[0,j-1])/dt)**2+potentiel1D((pathP[0,j]+pathP[0,j-1])/2)
    else:
        Ec=0
        Ep=0
        for k in range(len(pathP)): #Calcul de l'énergie cinétique du k-ème chemin
            for j in range(1,len(pathP[0])):
                Ec+=0.5*((pathP[k,j]-pathP[k,j-1])/dt)**2
        for i in range(len(pathP)):
            for k in range(i): #permet de faire une somme de nbParticule>i>k
                for j in range(1,len(pathP[0])):
                    Ep+=potentielND((pathP[i,j]+pathP[i,j-1])/2,(pathP[k,j]+pathP[k,j-1])/2)
        E=Ec+Ep 
    return E



#Fait une modification et la valide ou non en renvoyant le chemin modifié (ou non) ainsi que son énergie
@jit(nopython=True,cache=True) 		
def metropolis(pathP,u,epsilon,ener):
    particule=rd.randint(0,len(pathP))  #Permet de choisir un des chemins si il y en a plusieurs (nbParticule>1)
    a=u*(rd.random()*2-1)
    indexRandom=rd.randint(1,len(pathP[0,:])-1)
    pathP[particule,indexRandom]+=a     #Fait une proposition de modification
    eMetro=energie(pathP,epsilon)
    s=1 #Permet de compter le nombre de modifications que l'on valide (vient s'ajouter à la somme S)
    if eMetro>ener and np.exp(epsilon*(ener-eMetro))<=rd.random(): #Accepte ou non la modification
        pathP[particule,indexRandom]-=a
        s=0
        eMetro=ener
    return pathP,pathP[:,indexRandom],s,eMetro,particule






#Valeur moyenne de x^2
@jit(nopython=True,cache=True)
def meanXX(pathP):
    x2Mean=[]
    for i in range(len(pathP)):
        x2Mean.append((np.mean(pathP[i,1:-1]**2)))
    return np.array(x2Mean)

#Valeur moyenne de l'énergie pour des potentiels harmoniques (en passant par viriel)
@jit(nopython=True,cache=True)
def harmonicMoy(pathP):
    E=0
    L=len(pathP[0])
    nbParticule=len(pathP)
    X=0
    
    
    if nbParticule>1: 
        for i in range(nbParticule):
            X+=np.mean(pathP[i,1:-1]**2)
            for k in range(i):
                for j in range(1,L-1):
                    E+=potentielND(pathP[i,j],pathP[k,j])-pathP[k,j]*pathP[i,j] # Formule du PDF
        E=E/(L-2)+X*(nbParticule-1)/2    
        
        
        
    else:
        for j in range(1,L-1):
            E+=2*potentiel1D(pathP[0,j])
        E=E/(L-2)
    return E
    

def diffCurve1D(proba,nbPoint=25,nbBins=1000):   #Mesure la différence entre la densité de proba harmonique calculée et celle théorique
    z=np.zeros(nbPoint-1)
    xx=np.linspace(0,1,nbPoint-1)   #Nombre d'iterations normalisé
    for i in range(1,nbPoint):
        print(i,"/",nbPoint)
        b=int(i/nbPoint*len(proba))
        A=plt.hist(proba[:b],bins=nbBins,density=True)
        plt.clf()
        x=A[1]
        xxxx=np.zeros(len(x)-1)
        for j in range(len(x)-1):
            xxxx[j]=(x[j]+x[j+1])/2 #Crée un tableau contenant le milieu de chaque bins de l'histogramme
        z[i-1]=np.sqrt(scp.trapz((A[0]-gauss(xxxx))**2,xxxx)) 
    return xx,z



######################
######PARAMETRES######
######################
    
T=100 #Durée de la simulation
N=100 #Nombre de divisions de l'intervalle de temps T
L=6   #Taille considérée pour la longueur max du chemin 
M=100 #Nombre de divisions de l'intervalle d'espace L
nbParticule=1


x=np.linspace(-L/2,L/2,M)
delta=x[1]-x[0]  #longueur de chaque division d'espace (le dx)
t=np.linspace(0,T,N)
epsilon=t[1]   #durée de chaque division de temps (le dt)


a=0 #pt de départ
b=0 #pt d'arrivé

Nb=1000000  #Nombre d'iterations
nbLoop=1    #Nombre de fois que l'on fait la simulation




potentiel1D=harmonicOscillator   #Potentiel qui sera utilisé si il n'y a qu'une seule particule
potentielND=harmonicInterrac     #Potentiel qui sera utilisé si il y a plusieurs particules


proba=np.zeros((nbParticule,Nb*nbLoop)) #Le tableau qui, deviendra un histogramme des valeurs de metropolis

x2=np.zeros(nbParticule) #valeur moyenne de x^2
E=0 #Valeur moyenne de l'énergie
S=0 #Pourcentage de modifications acceptées

afficherVitesseConvergence=False #Marche uniquement si le potentiel est celui d'un oscillateur harmonique 1D
harmonic=(potentiel1D==harmonicOscillator and nbParticule==1) or (potentielND==harmonicInterrac and nbParticule>1)


############################
######CALCUL PRINCIPAL######
############################

for j in range(nbLoop):
    pathPosition=path(N,a,b,nbParticule,M) #Genère le chemin
    pathDist=pathPosition*delta #donne au chemin le bon ordre de grandeur
    
    for i in range(nbParticule):
        ax1=plt.subplot(1,2,2)
        ax1.plot(pathDist[i,:],t,label="chemin initial",color="C0") #Trace le chemin de départ
        
        
    ener=energie(pathDist,epsilon)
    for i in range(Nb):
        if i % int(Nb/10) == 0:
            print(int(i/Nb*10)+1,"/ 10 -",j+1,"boucle sur",nbLoop)
        pathDist,proba[:,j*Nb+i],s,ener,particulee=metropolis(pathDist,1.5,epsilon,ener)
        xx=meanXX(pathDist)
        x2+=xx
        if harmonic:
            e=harmonicMoy(pathDist)
            E+=e    
        S+=s            
    for i in range(nbParticule):
        ax1.plot(pathDist[i,:],t,label="chemin final",color="C1")  #Trace le chemin d'arrivé


######################
######AFFICHAGES######
######################      
    
print("Valeur moyenne de x²:",x2/(Nb*nbLoop)) 
if harmonic:
    print("Valeur moyenne de E",E/(Nb*nbLoop))     
print("% de valeurs acceptées:",S/(Nb*nbLoop))

plt.xlabel("position")
plt.ylabel("temps imaginaire")
plt.title("chemin")
plt.legend()



if nbParticule==1:
    plt.subplot(1,2,1)
    plt.hist(proba[0],bins=100,density=True,label=r'prédiction numérique de $|\psi_0(q)|^2$')
    plt.xlabel("position")
    plt.ylabel("probabilité")
    
    if potentiel1D==harmonicOscillator:
        plt.plot(x,gauss(x),label=r'courbe théorique de $|\psi_0(q)|^2$')
    plt.legend()
    if potentiel1D==harmonicOscillator and afficherVitesseConvergence:
        fig=plt.figure()
        X,Y=diffCurve1D(proba[0])
        plt.plot(X,Y)
    plt.show()
    
    
if nbParticule==2:
    plt.subplot(1,2,1)
    plt.hist2d(proba[0,:],proba[1,:],bins=100,cmap="gist_heat",normed=True) #pour une raison inconnue, density=True ne marche pas :^/
    plt.xlabel("position")
    plt.ylabel("position")
    plt.colorbar()
    plt.show()
    if potentielND==harmonicInterrac:
        fig = plt.figure()
        plt.hist((proba[0,:]-proba[1,:])/np.sqrt(2),bins=100, density=True)
        plt.plot(x,gauss(x),label=r"courbe théorique de la densité de probabilité selon $(q_b-q_a)/\sqrt{2}$")
        plt.xlabel(r"$(q_b-q_a)/\sqrt{2}$")
        plt.ylabel("probabilité")
        plt.legend()
        plt.show()



