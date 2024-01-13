# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import bezier
from matplotlib.patches import Wedge
from scipy.io import loadmat
from scipy.integrate import solve_ivp
import pickle

class cycle_figure:
    def __init__(self, G, source, dis, eta, kernel_layer = None):
        self.G = nx.subgraph(G, max(nx.connected_components(G), key=len)).copy()
        A = nx.to_numpy_array(G)
        self.d = np.sum(A, axis=0)
        self.source = source
        self.dis = dis
        self.eta = eta
        self.kernel_layer = kernel_layer
        self.is_sub = kernel_layer != None
    
    def tree_generating(self):
        G = self.G
        source = self.source
        father_tree = {0 : {source : list()}}
        path_length_dict = {source : 0}
        path_dict = {source : [source, ]}
        for u in G.nodes:
            if(u == source):
                continue
            dots = [node for node in nx.shortest_path(G, source = source, target = u)]
            path_length = len(dots) - 1
            path_length_dict[u] = path_length
            path_dict[u] = dots
            if(path_length-1 not in father_tree):
                father_tree[path_length - 1] = dict()
            if(path_length not in father_tree):
                father_tree[path_length] = dict()
            father_node = dots[-2]
            self_node = dots[-1]
            self.G.edges[father_node, self_node]['weight'] = -1
            if(father_node not in father_tree[path_length - 1]):
                father_tree[path_length - 1][father_node] = list()
                father_tree[path_length - 1][father_node].append(u)
            else:
                father_tree[path_length - 1][father_node].append(u)
            if(self_node not in father_tree[path_length]):
                father_tree[path_length][self_node] = list()
        self.father_tree = father_tree
        self.path_length_dict = path_length_dict
        self.path_dict = path_dict
        
    def leaves_counting(self):
        father_tree = self.father_tree
        max_length = np.max(list(father_tree.keys())) + 1
        ans_leave_counts = dict()
        for k in range(max_length - 1, -1, -1):
            for node in father_tree[k]:
                if(len(father_tree[k][node]) == 0):
                    ans_leave_counts[node] = (1, k)
                else:
                    ans_leave_counts[node] = (np.sum([ans_leave_counts[child][0] for child in father_tree[k][node]]), k)
        
        leave_counts = dict()
        for k in range(max_length):
            leave_counts[k] = dict()
        for node in ans_leave_counts:
            count, length = ans_leave_counts[node]
            leave_counts[length][node] = count
        return leave_counts
    
    def index_sorting(self):
        source = self.source
        father_tree = self.father_tree
        path_length_dict = self.path_length_dict
        G = self.G
        graph_size = len(G.nodes)
        max_length = np.max(list(father_tree.keys())) + 1
        index_sorted = dict()
        for k in range(max_length):
            index_sorted[k] = list()
        
        have_sorted = list()
        next_nodes = list()
        is_first = True
        
        current_node = source
        while(len(have_sorted) < graph_size or is_first):
            is_first = False
            index_sorted[path_length_dict[current_node]].append(current_node)
            have_sorted.append(current_node)
            if(len(have_sorted) == graph_size):
                break
            for next_node in father_tree[path_length_dict[current_node]][current_node]:
                if ((next_node not in have_sorted) and (next_node not in next_nodes)):
                    next_nodes.insert(0, next_node)
            current_node = next_nodes[-1]
            
            next_nodes.pop()
        return index_sorted
    
    def nodes_positioning(self):
        node_positions = dict()
        self.tree_generating()
        father_tree = self.father_tree
        # d= self.d
        # path_dict = self.path_dict
        leave_counts = self.leaves_counting()
        index_sorted = self.index_sorting()
        max_length = np.max(list(index_sorted.keys())) + 1
        
        kernel_layer = None
        total_leave_counts = list()
        for k in range(max_length):
            total_leave_counts.append(np.sum([leave_counts[k][node] for node in leave_counts[k]]))
        kernel_layer = max_length if (total_leave_counts[-1] / total_leave_counts[-2] > 0.2) else max_length - 1
        kernel_layer = min(self.kernel_layer, kernel_layer) if self.kernel_layer!=None else kernel_layer
        
        for k in range(kernel_layer-1, -1, -1):
            k_total_leave_counts = np.sum([leave_counts[k][node] for node in leave_counts[k]])
            k_cumsum_leave_counts = 0
            for node in index_sorted[k]:
                # print(path_dict[node],node)
                r = self.theory_time[node]**self.dis#np.sum([np.log(d[dot]) for dot in path_dict[node][:-1]])
                if(k < kernel_layer - 1 and len(father_tree[k][node]) > 0):
                    theta = np.mean([node_positions[child][1] for child in father_tree[k][node]])
                else:
                    theta = 2 * np.pi * k_cumsum_leave_counts / k_total_leave_counts
                k_cumsum_leave_counts += leave_counts[k][node]
                node_positions[node] = [r, theta]
        return node_positions
    
    def bezier_fitting(self, source, target):
        r1, theta1 = source
        r2, theta2 = target
        
        x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
        x3, y3 = r2 * np.cos(theta2), r2 * np.sin(theta2)
        x2, y2 = (r2) * np.cos(theta1), (r2) * np.sin(theta1)
        
        ans_nodes = [[i, j] for (i, j) in zip(np.linspace(x1, x2, 8), np.linspace(y1, y2, 8))]
        ans_nodes.append([x3, y3])
        
        curve=bezier.BezierSegment(ans_nodes)
        bezier_dots = curve(np.linspace(0,1,30))
        return bezier_dots
    
    def cycle_plotting(self, path, vector, theory_time, colors):
        G = self.G
        self.theory_time=theory_time
        node_positions = self.nodes_positioning()
        path_length_dict = self.path_length_dict
        fig = plt.figure(figsize = (12,12))
        ax = fig.add_subplot(111)
        
        H = nx.subgraph(G, list(node_positions.keys()))
        
        for (u, v) in H.edges:
            if(path_length_dict[u] > path_length_dict[v]):
                u,v = v,u
            source = node_positions[u]
            target = node_positions[v]
            if('weight' in G.edges[u,v] and G.edges[u,v]['weight'] == -1):
                bezier_dots = self.bezier_fitting(source, target)
                ax.plot(bezier_dots[:, 0], bezier_dots[:, 1], c='gray', linewidth = 1.5, alpha = 0.3,zorder=1)
            # else:
            #     nx.draw_networkx_edges(G, location, edge_color='#895551', width=1, alpha=0.35, edgelist = [(u, v)], arrowstyle = '-', connectionstyle=f'arc3, rad = {0.3}')
        node_list1=list()
        node_list2=list()
        for node in H.nodes:
            r, theta = node_positions[node]
            x, y = r * np.cos(theta), r * np.sin(theta)
            s=self.d[node]**(1.3)+150
            if(vector[node] > self.eta):
                node_list1.append((x,y,s))
            else:
                node_list2.append((x,y,s))
                
        for (x,y,s) in node_list2:
            ax.scatter(x,y,s=s,c=colors[0],zorder=3)
        for (x,y,s) in node_list1:
            ax.scatter(x,y,s=s,c=colors[1],edgecolor='black',linewidth = 1.5,zorder=3)
            
        ax.scatter(0, 0, s=self.d[self.source]**(1.3)+150, c=colors[1],edgecolor='black',linewidth = 1.5,zorder=3)
        
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks([],[])
        plt.yticks([],[])
        plt.tight_layout()
        plt.axis('equal')
        # title=path.split('_')[0].split('.')[0].split('/')[-1]
        # plt.title(title, fontsize=40)
        plt.savefig(path,dpi=300, bbox_inches = 'tight')
        plt.show()

class signal_propagation:
    def __init__(self,t_selected):
        self.gamma=0.3
        self.eta=0.3
        self.alpha = 0.01
        self.t_0=200
        self.t_1=200
        self.t_selected=t_selected
        self.B=1
        
    def simulation(self,A):
        def F(A,x):
            return np.mat(-self.B*x+self.alpha*np.multiply(1-x,A*x))
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
            x_0=np.zeros(np.shape(A)[0])
            sol=solve_ivp(Fun, [0,self.t_0], x_0, rtol=1e-10, atol=1e-10) #odeint(Fun,x_0,t,args=(A,a,b))
            xs=sol.y.T
            t=sol.t
            x=xs[-1,:].tolist()
            return x, t[-1]
        
        def sim_second(A,x,t_0,source):
            x[source]+=self.gamma
            sol=solve_ivp(Fun_1, [self.t_0,self.t_0+self.t_1], x, rtol=1e-10, atol=1e-10, args=(source,))#odeint(Fun_1,x,t,args=(A,a,b,source),atol=1e-13,rtol=1e-13)
            xs=sol.y.T
            ts=sol.t
            return np.mat(xs), np.array(ts)
            
        def time(xs,ts,eta):
            index=np.min(np.where(ts>(self.t_0+self.t_selected))[0])
            scale=(xs[index]-xs[0])/(xs[-1]-xs[0])
            return scale
        
        x,t=sim_first(A)
        xs,ts=sim_second(A,x.copy(),t,source)
        scale=time(xs.copy(),ts.copy(),self.eta).tolist()[0]
        return scale,xs

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    path='../Networks/DNCEmail.mat'
    A=loadmat(path)['A']
    G=nx.from_numpy_matrix(A)
    G=nx.subgraph(G, max(nx.connected_components(G), key=len))
    A=nx.to_numpy_matrix(G)
    d=np.sum(A,axis=0)
    source = 1641
    B=1
    eta = 0.3
    t_selected=1.5
    
    theory_time_hens = load_dict('E_DNCEmail_global_degree_hens_'+str(int(100*B))).tolist()[0]
    theory_time_multi = load_dict('E_DNCEmail_global_degree_multi_hens_'+str(int(100*B))).tolist()[0]
    theory_time_ours = load_dict('E_DNCEmail_global_degree_multi_'+str(int(100*B))).tolist()[0]

    signal=signal_propagation(t_selected)
    scale,xs=signal.simulation(A)
    
    colors1 = ['#8DDFBF', '#328CA0']
    path1 = 'E_DNCEmail_{}_distance_network_hens.pdf'.format(int(100*t_selected))
    dis1=1
    figure1 = cycle_figure(G, source, dis1, eta)
    figure1.cycle_plotting(path1, scale, theory_time_hens, colors1)
    
    colors2 = ['#ABBAE1', '#526D92']
    path2 = 'E_DNCEmail_{}_distance_network_multi.pdf'.format(int(100*t_selected))
    dis2=2
    figure2 = cycle_figure(G, source, dis2, eta)
    figure2.cycle_plotting(path2, scale, theory_time_multi, colors2)
    
    colors3 = ['#EECB8E', '#DC8910']
    path3 = 'E_DNCEmail_{}_distance_network_ours.pdf'.format(int(100*t_selected))
    dis3=2
    figure3 = cycle_figure(G, source, dis3, eta)
    figure3.cycle_plotting(path3, scale, theory_time_ours, colors3)