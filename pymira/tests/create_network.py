# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 07:18:10 2017

@author: simon
"""

import numpy as np
arr = np.asarray
from pymira import spatialgraph

def segment():

    #nodes = [[317,1215,781],
    #         [370,1221,822],
    #        ]
            
    #nodes = [[306,1174,968],
    #         [445,1149,1273],
    #        ]
    nodes = [ [0,0,0],[0,-100,0] ]
            
    connectivity = [[0,1],
                    ]
                    
    vessel_type = [0,
                   ]                    
                    
    edgeradii = [ 5.,
                 ] 
                
    return nodes, connectivity, edgeradii, vessel_type
    

def bifurcation():

    nodes = [[0,0,0],
             [100,0,0],
             [200.,100,0],
             [200.,-100.,0],
            ]
            
    connectivity = [[0,1],
                    [1,2],
                    [1,3],
                    ]
                    
                    
    vessel_type = [0,
                   0,
                   0,
                   ]                    
                    
    edgeradii = [ 30.,
                  20.,
                  20.,
                 ] 
                
    return nodes, connectivity, edgeradii, vessel_type
    
def double_bifurcation():

    nodes = [[0,0,0],
             [100,0,0],
             [200.,100,0],
             [200.,-100.,0],
             [300.,200,0],
             [300.,100.,0],
             [300.,-200,0],
             [300.,-100.,0],
            ]
            
    connectivity = [[0,1],
                    [1,2],
                    [1,3],
                    [2,4],
                    [2,5],
                    [3,6],
                    [3,7],
                    ]
                    
                    
    vessel_type = [0,
                   0,
                   0,
                   
                   2,
                   2,
                   2,
                   2,
                   ]                    
                    
    edgeradii = [ 30.,
                  20.,
                  20.,
                  10.,
                  10.,
                  10.,
                  10.,
                 ] 
                
    return nodes, connectivity, edgeradii, vessel_type
    
def double_bifurcation_reconnected():

    nodes = [[0,0,0],
             [100,0,0],
             [200.,100,0],
             [200.,-100.,0],
             
             [300.,200,0],
             [300.,100.,0],
             [300.,-200,0],
             [300.,-100.,0],
             
             [400.,100,0],
             [400.,-100.,0],
             [500,0,0],
             [600,0,0],
            ]
            
    connectivity = [[0,1],
                    [1,2],
                    [1,3],
                    
                    [2,4],
                    [2,5],
                    [3,6],
                    [3,7],
                    
                    [4,8],
                    [5,8],
                    [6,9],
                    [7,9],
                    
                    [8,10],
                    [9,10],
                    [10,11],
                    ]
                    
    vessel_type = [0,
                   0,
                   0,
                   
                   2,
                   2,
                   2,
                   2,
                   
                   2,
                   2,
                   2,
                   2,
                   
                   1,
                   1,
                   1,
                   ]
                   
                    
    edgeradii = [ 30.,
                  20.,
                  20.,
                  
                  10.,
                  10.,
                  10.,
                  10.,
                  
                  10.,
                  10.,
                  10.,
                  10.,
                  
                  20.,
                  20.,
                  30.,
                 ] 
                
    return nodes, connectivity, edgeradii, vessel_type

def custom_graph():

    nodes = [[0,0,0],
             [100,0,0],
             [50.,0,0],
             [0,100.,0],
             [200,200,0],
             [500,500,0],
             [200,500,0],
            ]
            
    connectivity = [[0,1],
                    [2,3],
                    [1,4],
                    [4,5],
                    [4,6],
                    [0,0]
                    ]
                    
    vessel_type = np.zeros(len(connectivity))
                    
    edgeradii = [ 30.,
                  10.,
                  20.,
                  10.,
                  15.,
                  5.,
                 ] 
                 
    return nodes, connectivity, edgeradii, vessel_type

def kidney_cco():
    # Kidney CCO
    nodes = [[500,1600,1000], # origin
             [500,1400,1000], # origin
             [317,1215,781],
             [370,1221,822],
             [306,1174,968],
             [445,1149,1273],
            ]
            
    nodes = arr(nodes)
    
    

    connectivity = [[0,1],
                    [1,2],
                    [1,3],
                    [1,4],
                    [1,5],
                    ]            
                    
    # Extended into mesh...  
    for conn in connectivity[1:]:      
        i0,i1 = conn
        dir1 = nodes[i1,:]-nodes[i0,:]
        l1 = np.linalg.norm(dir1)
        nodes[i1,:] = nodes[i0,:] + dir1*1.5  
                    
    vessel_type = [0,
                   0,
                   0,
                   0,
                   0,
                   ]                    
                    
    edgeradii = [ 30.,
                  10.,
                  10.,
                  10.,
                  10,
                 ] 
    return nodes, connectivity, edgeradii, vessel_type
    
def kidney_cco_simple():
    # Kidney CCO
    nodes = [[500,1300,1000], # origin
             [445,1149,1000]
            ]
            
    nodes = arr(nodes)
    
    connectivity = [[0,1],
                    ]            
                    
    # Extended into mesh...  
    for conn in connectivity:      
        i0,i1 = conn
        dir1 = nodes[i1,:]-nodes[i0,:]
        l1 = np.linalg.norm(dir1)
        nodes[i1,:] = nodes[i0,:] + dir1*1.6
        print(l1,dir1)
                    
    vessel_type = [0,
                   ]                    
                    
    edgeradii = [ 20.,
                 ] 
    return nodes, connectivity, edgeradii, vessel_type    
    
#nodes, connectivity, edgeradii, vessel_type = double_bifurcation_reconnected()
#nodes, connectivity, edgeradii, vessel_type = bifurcation()
#nodes, connectivity, edgeradii, vessel_type = segment()
nodes, connectivity, edgeradii, vessel_type = kidney_cco()
#nodes, connectivity, edgeradii, vessel_type = kidney_cco_simple()

nodes = arr(nodes)
#nodes[:,0] -= 250
#nodes = -nodes
                
edgepoints,nedgepoints,radii,category = [],[],[],[]
for i,conn in enumerate(connectivity):
    edgepoints.append(nodes[conn[0]])
    edgepoints.append(nodes[conn[1]])
    nedgepoints.append(2)
    radii.append([edgeradii[i],edgeradii[i]])
    category.append([vessel_type[i],vessel_type[i]])
edgepoints = arr(edgepoints)
nedgepoints = arr(nedgepoints)
radii = arr(radii).flatten()
category = arr(category).flatten()

nodes = np.asarray(nodes,dtype='float')
nedgepoints = np.asarray(nedgepoints,dtype='int')
connectivity = np.asarray(connectivity,dtype='int')
edgepoints = np.asarray(edgepoints,dtype='float')
radii = np.asarray(radii,dtype='float')
category = np.asarray(category,dtype='int')

nnode = nodes.shape[0]
nedge = nedgepoints.shape[0]
npoint = edgepoints.shape[0]
    
graph = spatialgraph.SpatialGraph(initialise=True,scalars=['Radii','VesselType'])
graph.set_definition_size('VERTEX',nnode)
graph.set_definition_size('EDGE',nedge)
graph.set_definition_size('POINT',npoint)
graph.set_data(nodes,name='VertexCoordinates')
graph.set_data(connectivity,name='EdgeConnectivity')
graph.set_data(nedgepoints,name='NumEdgePoints')
graph.set_data(edgepoints,name='EdgePointCoordinates')
graph.set_data(radii,name='Radii')
graph.set_data(category,name='VesselType')

graph.sanity_check(deep=True)

#graph._print()
ofile = '/mnt/data2/kidney_cco/segment1.am'
tp = graph.plot_graph(show=False,block=False)
gmesh_artery = tp.cylinders_combined
ofile2 = ofile.replace('.am','.ply')
import open3d as o3d
o3d.io.write_triangle_mesh(ofile2,gmesh_artery)
tp.destroy_window()
graph.write(ofile)
print(f'Saved as {ofile}')

# Convert to JSON
from pymira.amirajson import convert
convert(ofile)
