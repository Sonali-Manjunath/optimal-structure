import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Problem Definition
A= 1
E= 200e9
d = 8000e3
x = (30**2 - 16)**0.5
nodeCords=np.array([[2,0], #A0
                    [2,30], #D1
                    [2,35], #E2
                    [x + 2,31], #F3
                    [x+2,30], #G4
                    [7,30], #C5
                    [7,0], #B6
                    [7,30+4.325]#9
                    ])
#Nodal Coordinates
elemNodes=np.array([
                    [0,1],#0
                    [1,2],#1
                    [2,7],#2
                    [3,7],#3
                    [3,4],#4
                    [4,5],#5
                    [5,6],#6
                    [6,0],#7
                    [3,5],#8
                    [1,5],#9
                    [1,6],#10
                    [4,7],#11
                    [5,7],#12
                    [2,5],#13
                    [0,5],#14
                    [1,7]#15
                    
                    
                    
                    ]) #Element connectivity: near node and far node
modE=np.array([
    [200e9],[200e9],[200e9],
    [200e9],[200e9],[200e9],
    [200e9],[200e9],[200e9],
    [200e9],[200e9],[200e9],
    [200e9],[200e9],[200e9],
    [200e9],[200e9],
    [200e9],
    [200e9]]) #Young's modulus
Area=np.array([
    [0.04329784244],
    [0.003471043703],
    [0.003891700862],
    [0.03464746377],
    [0.001157014568],
    [0.06358987414],
    [0.1203830186],
    [0.006942087405],
    [0.08591647419],
    [0.01577747138],
    [0.04454332189],
    [0.03041852368],
    [0.03002452803],
    [0.04908797078],
    [0.0444495464],
    # [0.004589429943],
    
    
    
    ]) #Cross section area
DispCon=np.array([[0,1,0],[0,2,0],[6,2,0]]) #Displacement constraints
Fval=np.array([[4,2,-98634]]) #Applied forc

scale = 1


# #Problem Initialization
nELEM=elemNodes.shape[0] # Number of elements
nNODE=nodeCords.shape[0] # Number of nodes
nDC=DispCon.shape[0] # Number og constrained DOF
nFval=Fval.shape[0] # Number of DOFs where forces are applied
NDOF=nNODE*2 # Total number of DOFs
uDisp=np.zeros((NDOF,1)) #Displacement vector is NDOF X 1
forces=np.zeros((NDOF,1)) # Forces vector is NDOF X 1
Stiffness=np.zeros((NDOF,NDOF)) # Stiffness matrix is NDOF X NDOF
Stress=np.zeros((nELEM)) # Stress is nELEM X 1 vector
kdof=np.zeros((nDC)) # All known DOFs nDC x 1 vector
xx=nodeCords[:,0] # All X-coordinates of the nodes nNODE X 1 vector
yy=nodeCords[:,1] # All Y-coordinates of the nodes nNODE X 1 vector
L_elem=np.zeros((nELEM)) # All lengths of trusses nELEM X 1 vector

# #Building the displacement array
for i in range(nDC): #looping over the number of known degrees of freedom
    indice=DispCon[i,:]
    v=indice[2] #value of the known displacement 
    v=v.astype(float)
    indice=indice.astype(int)
    kdof[i]=indice[0]*2+indice[1]-1 # The corresponding degree of freedom that is constrained is assigned to kdof[i]
    uDisp[indice[0]*2+indice[1]-1]=v # The corresponding displacement value is assigned to uDisp
    
# #Building the force array

for i in range(nFval): #looping over the dofs where forces are applied
    indice2=Fval[i,:]
    v=indice2[2];
    v=v.astype(float)
    indice2=indice2.astype(int)
    forces[indice2[0]*2+indice2[1]-1]=v # Assigning the value of the force in the forces vector



# #Identifying known and unknown displacement degree of freedom
kdof=kdof.astype(int) #Contains all degrees of freedom with known displacement
ukdof=np.setdiff1d(np.arange(NDOF),kdof) #Contains all degrees of freedom with unknown displacement


# #Loop over all the elements

for e in range(nELEM):
    indiceE=elemNodes[e,:] #Extracting the near and far node for element 'e'
    Y=modE[e]
    Ae=Area[e]
    elemDOF=np.array([indiceE[0]*2,indiceE[0]*2+1,indiceE[1]*2,indiceE[1]*2+1]) #Contains all degrees of freedom for element 'e'
    elemDOF=elemDOF.astype(int)
    print(elemDOF)
    xa=xx[indiceE[1]]-xx[indiceE[0]]
    ya=yy[indiceE[1]]-yy[indiceE[0]]
    len_elem=np.sqrt(xa*xa+ya*ya) #length of the element 'e'
    c=xa/len_elem #lambda x
    s=ya/len_elem #lambda y


    
#     # Step 1. Define elemental stiffness matrix
    k=(Y*Ae/len_elem)*np.array([[1,-1],[-1,1]])
    
#     # Step 2. Transform elemental stiffness matrix from local to global coordinate system
    T=np.array([[c,s,0,0],[0,0,c,s]])
    k2=np.matmul(T.transpose(),np.matmul(k,T))
    
#     # Step 3. Assemble elemental stiffness matrices into a global stiffness matrix
    Stiffness[np.ix_(elemDOF,elemDOF)] += k2
    
#     # Step 4. Partition the stiffness matrix into known and unknown dofs
    k11 = Stiffness[np.ix_(ukdof,ukdof)]
    k12 = Stiffness[np.ix_(ukdof,kdof)]
    k21 = k12.transpose()
    k22 = Stiffness[np.ix_(kdof,kdof)]


# # Step 4a. Solve for the unknown dofs and reaction forces
f_known = forces[ukdof]-np.matmul(k12, uDisp[kdof])

uDisp[np.ix_(ukdof)] = np.linalg.solve(k11, f_known)

forces[np.ix_(kdof)]=np.matmul(k21,uDisp[np.ix_(ukdof)])+np.matmul(k22,uDisp[np.ix_(kdof)])

plt.figure(300) # came in skeleton code

# # Step 5. Evaluating Internal Forces and stresses
for e in range(nELEM):
    indiceE=elemNodes[e,:]
    Y=modE[e]
    Ae=Area[e]
    elemDOF=np.array([indiceE[0]*2,indiceE[0]*2+1,indiceE[1]*2,indiceE[1]*2+1])
    elemDOF=elemDOF.astype(int)
    xa=xx[indiceE[1]]-xx[indiceE[0]]
    ya=yy[indiceE[1]]-yy[indiceE[0]]
    len_elem=np.sqrt(xa*xa+ya*ya)
    L_elem[e]=len_elem
    c=xa/len_elem
    s=ya/len_elem
    
    #Elemental Stiffness Matrix
    ke = (Y*Ae/len_elem)*np.array([[1,-1],[-1,1]])

    
    
    # Transformation Matrix    
    T = np.array([[c,s,0,0],[0,0,c,s]])
    
    #Internal forces
    Fint = np.matmul(ke, np.matmul(T, uDisp[np.ix_(elemDOF)]))
    # print("Internal Force of " + str(e) + ": " + str(Fint[1]))
    print(str(Fint[1]))
    #Stress
    Stress[e] = Fint[1]/Ae
    
    
    
  
    plt.plot(np.array([xx[indiceE[0]],xx[indiceE[1]]]),np.array([yy[indiceE[0]],yy[indiceE[1]]]))
    plt.plot(np.array([xx[indiceE[0]]+uDisp[indiceE[0]*2]*scale,xx[indiceE[1]]+uDisp[indiceE[1]*2]*scale]),np.array([yy[indiceE[0]]+uDisp[indiceE[0]*2+1]*scale,yy[indiceE[1]]+uDisp[indiceE[1]*2+1]*scale]),'--')


plt.xlim(min(xx)-abs(max(xx)/10), max(xx)+abs(max(xx)/10))
plt.ylim(min(yy)-abs(max(yy)/10), max(yy)+abs(max(xx)/10))
plt.gca().set_aspect('equal', adjustable='box')
pduDisp = pd.DataFrame({'disp': uDisp[:,0]})
pdforces=pd.DataFrame({'forces': forces[:,0]})
pdStress=pd.DataFrame({'Stress': Stress})

pdLen=pd.DataFrame({'Length': L_elem})
#Displaying the results
print(pduDisp)
print(pdforces)
print(pdStress)


plt.show()