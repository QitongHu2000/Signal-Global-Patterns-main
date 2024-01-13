# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.3
eta=0.3
alpha = 0.01
t_0=500
t_1=500
B=1
C=2
from scipy.io import loadmat,savemat

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def simulation(A):
    def F(A,x):
        return np.mat(B-C*x-alpha*np.multiply(x,A*x))
        # return np.mat(-B*np.multiply(1-x,x)+np.multiply(x,A*(1-1/x)))
    
    def Fun(t,x):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(t,x,source):
        x=np.mat(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[source]=0
        return dx
    
    def sim_first(A):
        x_0=np.ones(np.shape(A)[0])*0.1
        sol=solve_ivp(Fun, [0,t_0], x_0, rtol=1e-13, atol=1e-13) #odeint(Fun,x_0,t,args=(A,a,b))
        xs=sol.y.T
        t=sol.t
        x=xs[-1,:].tolist()
        return x, t[-1]
    
    def sim_second(A,x,t_0,source):
        x[source]*=(1+gamma)
        sol=solve_ivp(Fun_1, [t_0,t_0+t_1], x, rtol=1e-13, atol=1e-13, args=(source,))#odeint(Fun_1,x,t,args=(A,a,b,source),atol=1e-13,rtol=1e-13)
        xs=sol.y.T
        ts=sol.t
        return np.mat(xs), np.array(ts)
        
    def time(xs,ts,eta):
        xs=(xs-xs[0])/(xs[len(xs)-1]-xs[0])
        indexs=np.argmax(1/(eta-xs),axis=0).tolist()[0]
        times=[]
        for i in range(len(indexs)):
            len_1=xs[indexs[i]+1,i]-xs[indexs[i],i]
            len_2=eta-xs[indexs[i],i]
            times.append(ts[indexs[i]]+len_2/len_1*(ts[indexs[i]+1]-ts[indexs[i]]))
        return np.mat(times)
    
    x,t=sim_first(A)
    print(x)
    xs,ts=sim_second(A,x.copy(),t,source)
    times=time(xs.copy(),ts.copy(),eta).tolist()[0]
    print(times)
    return times,x, xs

def R(x):
    return (C*x-B)/x

def W(x):
    return x/(C*x-B)

def H2(x):
    return -alpha * x

def R_(x):
    return B/x**2

def W_(x):
    return -B/(C*x-B)**2

def H2_(x):
    return -alpha

def C_(x):
    return -alpha/(B/x**2)

def calculate_Wij(H,degrees,xs):
    count=0
    r_roots=dict()
    for u in H.nodes:
        r_root=0
        for v in list(dict(H[u]).keys()):
            r_root+=(xs[0,u]*W_(xs[0,u])*H2(xs[0,v]))
        r_roots[u]=r_root
    
    for (u,v) in H.edges:
        count+=1
        H.edges[u,v]['weight_hens']=float(degrees[v,0])**(0)
        
        H.edges[u,v]['Rij']=np.abs(W(xs[0,u])*H2_(xs[0,v])*xs[0,v]/r_roots[u])
        H.edges[u,v]['Cij']=H2_(xs[0,v])/R_(xs[0,u])
    return H

def calculate_shortest_path(H,source):
    paths_hens=dict()
    values_hens=dict()
    paths_L=dict()
    values_L=dict()
    for u in H.nodes:
        if(u==source):
            values_hens[u]=0
            values_L[u]=0
            continue
        
        value_hens=0
        paths_hens[u]=nx.shortest_path(H,source=source,target=u,weight='weight_hens')
        values_hens[u]=0
        for i in range(1,len(paths_hens[u])-1):
            value_hens=H.edges[paths_hens[u][i],paths_hens[u][i+1]]['weight_hens']
            values_hens[u]+=value_hens

        paths_L[u]=nx.shortest_path(H,source=source,target=u)
        values_L[u]=0
        for i in range(1,len(paths_L[u])-1):
            values_L[u]+=1
        
    return paths_hens,values_hens,paths_L,values_L

def calculate_multi_path(H,source):
    values_multi_hens=dict()
    values_multi=dict()
    values_multi_weight=dict()
    for (k,u) in enumerate(H.nodes):
        # if(k>1000):
        #     break
        if(u==source):
            values_multi_hens[u]=0
            values_multi[u]=0
            values_multi_weight[u]=0
            continue
        print(u)
        # paths=nx.all_shortest_paths(H,source=source,target=u)
        paths=nx.all_simple_paths(H, source=source, target=u, cutoff=int(nx.shortest_path_length(H,source=source,target=u)) + 1)
        
        value_multis_hens=list()
        value_multis=list()
        value_multis_weight=list()
        Rij_multis=list()
        Cij_multis=list()
        
        for path in paths:
            value_hens=0
            value=0
            value_weight=0
            
            Rij_multi=1
            Cij_multi=1
            weight=[0 for i in range(len(path)-1)]
            weight[-1]=1
            for i in range(len(path)-3,-1,-1):
                weight[i]=1+C_(xs_edge[0,path[i]])*C_(xs_edge[0,path[i+1]])/weight[i+1]
            for i in range(0,len(path)-1):
                Rij_multi*=H.edges[path[i+1],path[i]]['Rij']
                Cij_multi*=H.edges[path[i],path[i+1]]['Cij']
                value_hens+=H.edges[path[i],path[i+1]]['weight_hens']
                value+=H.edges[path[i],path[i+1]]['weight_hens']
                value_weight+=H.edges[path[i],path[i+1]]['weight_hens']*weight[i]
            value_multis_hens.append(value_hens)
            value_multis.append(value)
            value_multis_weight.append(value_weight)
            Rij_multis.append(Rij_multi)
            Cij_multis.append(Cij_multi)
        values_multi_hens[u]=np.sum([value*Rij_multi/np.sum(Rij_multis) for (value,Rij_multi) in zip(value_multis_hens, Rij_multis)])
        values_multi[u]=np.sum([value*Cij_multi/np.sum(Cij_multis) for (value,Cij_multi) in zip(value_multis, Cij_multis)])
        values_multi_weight[u]=np.sum([value*Cij_multi/np.sum(Cij_multis) for (value,Cij_multi) in zip(value_multis_weight, Cij_multis)])
    return values_multi_hens, values_multi, values_multi_weight

# n=1000
# m=10
# G_edge=nx.barabasi_albert_graph(n,m)
# G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
# A_edge=nx.to_numpy_matrix(G_edge)
# savemat('../Networks/BA.mat',{'A':A_edge})

A_edge=np.mat(loadmat('../Networks/PPI4.mat')['A'])
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=nx.to_numpy_matrix(G_edge)
degrees=np.sum(A_edge,axis=1)

G_edge=nx.from_numpy_matrix(A_edge)

k=1
sources=[1745, ] #np.random.choice(G_edge.nodes,k) #[196, 55, 107]
times_simulation_times=list()
theory_times_hens=list()
theory_times_L=list()
theory_times_multi=list()
theory_times_multi_hens=list()
theory_times_multi_weight=list()

count=0
for source in sources:
    print(source)
    times_simulation_time,x_edge, xs_edge =simulation(A_edge)
    
    H_edge=calculate_Wij(G_edge,degrees,xs_edge)
    shortest_path_hens,theory_time_hens,\
        shortest_path_L,theory_time_L=calculate_shortest_path(H_edge,source)
    values_multi_hens, values_multi, values_multi_weight = calculate_multi_path(H_edge,source)
    
    theory_time_hens=[theory_time_hens[i] for i in theory_time_hens]
    theory_time_L=[theory_time_L[i] for i in theory_time_L]
    theory_time_multi_hens=[values_multi_hens[i] for i in values_multi_hens]
    theory_time_multi=[values_multi[i] for i in values_multi]
    theory_time_multi_weight=[values_multi_weight[i] for i in values_multi_weight]

    times_simulation_time=[times_simulation_time[i]-t_0 for i in G_edge.nodes]
    
    times_simulation_times.extend(times_simulation_time)
    theory_times_hens.extend(theory_time_hens)
    theory_times_L.extend(theory_time_L)
    theory_times_multi.extend(theory_time_multi)
    theory_times_multi_hens.extend(theory_time_multi_hens)
    theory_times_multi_weight.extend(theory_time_multi_weight)
    
save_dict(np.mat(times_simulation_times), 'B_PPI4_global_time_'+str(int(100*B))+'_'+str(int(100*C)))
save_dict(np.mat(theory_times_hens), 'B_PPI4_global_degree_hens_'+str(int(100*B))+'_'+str(int(100*C)))
save_dict(np.mat(theory_times_L), 'B_PPI4_global_degree_L_'+str(int(100*B))+'_'+str(int(100*C)))
save_dict(np.mat(theory_times_multi_hens), 'B_PPI4_global_degree_multi_hens_'+str(int(100*B))+'_'+str(int(100*C)))
save_dict(np.mat(theory_times_multi), 'B_PPI4_global_degree_multi_'+str(int(100*B))+'_'+str(int(100*C)))
save_dict(np.mat(theory_times_multi_weight), 'B_PPI4_global_degree_multi_weight_'+str(int(100*B))+'_'+str(int(100*C)))