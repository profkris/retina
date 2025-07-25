import numpy as np
arr = np.asarray
norm = np.linalg.norm
import open3d as o3d
import pyvista as pv
from tqdm import tqdm, trange
from pymira import geometry

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, 0.
    axis_ = np.cross(a, b)
    l = np.linalg.norm(axis_)
    if l>0.:
        axis_ = axis_ / np.linalg.norm(axis_)
        angle = np.arccos(np.dot(a, b))
    else:
        angle = 0.

    return axis_, angle

class TubePlot(object):

    def __init__(self,graph, parameter=None, cylinders=None, cylinders_combined=None, color=None, edge_color=None, 
                         min_radius=0.,domain_radius=None,radius_scale=1.,fixed_radius=None,domain_centre=None,radius_based_resolution=True,cyl_res=10,edge_filter=None,node_filter=None,
                         cmap_range=[None,None],bgcolor=[0.,0.,0.],cmap=None,win_width=6000,win_height=6000,grab_file=None,
                         edge_highlight=[],node_highlight=[],highlight_color=[1,1,1],scalar_color_name=None,log_color=False,
                         show=True,block=True,engine='open3d',domain=None,domain_type='cylinder',ignore_domain=False,additional_meshes=None):
        self.vis = None
        self.headless = False # If headless mode detected, don't try and display anything
        self.graph = graph
        self.cylinders = cylinders
        self.cylinders_combined = cylinders_combined
        self.additional_meshes = additional_meshes
        
        # Minimum vessel radius to plot (scalar)
        self.min_radius = min_radius
        # Domain
        self.domain = domain
        self.domain_type = domain_type
        self.ignore_domain = ignore_domain
        # Size of area to plot
        self.domain_radius = domain_radius
        # Factor to scale vessel radii by
        self.radius_scale = radius_scale
        self.fixed_radius = fixed_radius
        # Centre of plot domain
        self.domain_centre = None
        if domain_centre is None:
            if self.domain is not None:
                self.domain_centre = (self.domain[:,1]+self.domain[:,0])/2.
        else:
            self.domain_centre = domain_centre
        #self.domain_radius = None
        if domain_radius is None: # Use size of first domain
            if self.domain is not None:
                self.domain_radius = (self.domain[0,1]-self.domain[0,0])/2.
        else:
            self.domain_radius = domain_radius
        # Cylinder resolution based on radius (boolean)
        self.radius_based_resolution = radius_based_resolution
        # Cylinder resolution (scalar)
        self.cyl_res = cyl_res
        # Edges and nodes to include in plot (np.where result)
        self.edge_filter = edge_filter
        self.node_filter = node_filter
        
        # Backend (open3d) (pyvista no longer supported!)
        self.engine = 'open3d' #engine
        # Colour range for edges [None,None]
        self.cmap_range = cmap_range
        # Background colour
        self.bgcolor = bgcolor
        # Array of edge colours ([nedge])
        self.edge_color = edge_color
        # Edge colors ([nedge,3])
        self.color = color
        # Colour map name ('gray','jet')
        self.cmap = cmap
        # Window dimensions
        self.win_width = win_width
        self.win_height = win_height
        # Make window visible (boolean)
        self.show = show
        # Blocking behaviour (boolean)
        self.block = block
        
        # Array identifying edges to highlight (np.where result)
        self.edge_highlight = edge_highlight
        # Array identifying nodes to highlight
        self.node_highlight = node_highlight
        # Hightlight colour for above
        self.highlight_color = highlight_color
        # Scalar parameter for edge colours (radius by default)
        if parameter is not None:
            self.scalar_color_name = parameter
        else:
            self.scalar_color_name = scalar_color_name
        # Whether to log colour scale (boolean)
        self.log_color = log_color
        
        # Create cylinders if they have not been provided
        if self.cylinders is None and self.cylinders_combined is None:
            self.create_plot_cylinders()
     
        # Create plot window
        self.create_plot_window()
        if self.headless:
            #print('Headless mode! Cannot render image')
            return
            
        # Set colours (only if raw cylinders have been provided)
        if self.cylinders_combined is None: 
            print('Preparing graph (adding color and combining)...')
            if self.scalar_color_name is None:
                if 'VesselType' in graph.fieldNames:
                    self.scalar_color_name = 'VesselType'
                else:
                    radName = graph.get_radius_field()['name']
                    self.scalar_color_name = radName
            self.set_cylinder_colors()
            # Combine cylinders
            self.combine_cylinders()                

        if self.block:
            self._show_plot()
            
            
    def find_domain_intersection(self,start_coord,end_coord,epsilon=1e-6):
        # Find which face segment intersects
        if self.domain_type=='cuboid':
            # Point on each face
            # Normal to each face
            pn = arr([[0.,0.,1.],[0.,-1.,0],[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.],[-1.,0.,0.]])
            intersection = np.zeros([6,3]) * np.nan
            for i in range(6): 
                pnt = arr([(x[1]-x[0])/2. for x in self.domain])
                ind = np.where(pn[i]==1)
                if len(ind[0])>0:                
                    pnt[ind[0]] = self.domain[ind[0],0]
                ind = np.where(pn[i]==-1)
                if len(ind[0])>0:                
                    pnt[ind[0]] = self.domain[ind[0],1]
                    
                dir = end_coord - start_coord
                coll,pt = geometry.line_plane_intersection(pnt,pn[i],end_coord,dir)
                # See if intersection point is inside the domain
                if pt is not None and self.inside_domain(pt):
                    intersection[i,:] = pt
            # Find closest intersection that is on the surface of the cubic domain
            if np.all(~np.isfinite(intersection)):
                return None
            face = np.nanargmin([norm(x-end_coord) for x in intersection])
            coord = intersection[face,:]
            return coord 
        elif self.domain_type=='rectangle':
            # Point on each face
            # Normal to each face
            pn = arr([[0.,0.,1.],[0.,-1.,0],[0.,1.,0.],[1.,0.,0.]])
            intersection = np.zeros([4,3]) * np.nan
            for i in range(4): 
                pnt = arr([(x[1]-x[0])/2. for x in self.domain])
                ind = np.where(pn[i]==1)
                if len(ind[0])>0:                
                    pnt[ind[0]] = self.domain[ind[0],0]
                ind = np.where(pn[i]==-1)
                if len(ind[0])>0:                
                    pnt[ind[0]] = self.domain[ind[0],1]
                    
                dir = end_coord - start_coord
                coll,pt = geometry.line_plane_intersection(pnt,pn[i],end_coord,dir)
                # See if intersection point is inside the domain
                if pt is not None and self.inside_domain(pt):
                    intersection[i,:] = pt
            # Find closest intersection that is on the surface of the cubic domain
            if np.all(~np.isfinite(intersection)):
                return None
            face = np.nanargmin([norm(x-end_coord) for x in intersection])
            coord = intersection[face,:]
            return coord 
        else:  
            return None                   
            
    def inside_domain(self,coord,start_coord=None,epsilon=1e-6,**kwargs):
    
        if self.ignore_domain:
            return True, None

        #if self.domain is None:
        #    return True, None
        # Cuboid domain
        if self.domain_type=='cuboid' and self.domain is not None:
            if np.all(self.domain[:,0]<=coord) and np.all(self.domain[:,1]>=coord):
                return True, None
            elif start_coord is not None:
                dom_int = self.find_domain_intersection(start_coord,coord)
                return False, dom_int
            else:
                return False, None
        elif self.domain_type=='rectangle' and self.domain is not None:
            if np.all(self.domain[:2,0]<=coord[:2]) and np.all(self.domain[:2,1]>=coord[:2]):
                return True, None
            elif start_coord is not None:
                dom_int = self.find_domain_intersection(start_coord,coord)
                return False, dom_int
            else:
                return False, None
        elif self.domain_type=='cylinder':
            if self.domain_centre is None or self.domain_radius is None:
                return True, None
            in0 = np.linalg.norm(start_coord-self.domain_centre)<=self.domain_radius
            in1 = np.linalg.norm(coord-self.domain_centre)<=self.domain_radius
            if not in0 and not in1:
                return False, None
            elif in0 and in1:
                return True, None
            else:
                return False, None #self.find_domain_intersection(start_coord,coord)
        elif self.domain_type=='surface':
            # Assumes a plane oriented in x-y
            if np.all(self.domain[0:2,0]<=coord[0:2]) and np.all(self.domain[0:2,1]>=coord[0:2]):
                return True, None
            elif start_coord is not None:
                dom_int = self.find_domain_intersection(start_coord,coord)
                return False, dom_int
            else:
                return False, None
        elif self.domain_type=='sphere':
            radius = (self.domain[:,1]-self.domain[:,0])/2.
            centre = arr([0.,0.,0.])
            dist = np.linalg.norm(coord-centre)
            if dist>radius[0]:
                return False, None
            else:
                return True, None
        else:
            return True, None          
    
    # Updated the function
    def register_scalar_to_graph(graph, name):
        """Ensure scalar is available in graph.fields"""
        if name not in [f['name'] for f in graph.fields]:
            data = graph.get_data(name)
            if data is not None and np.any(np.isfinite(data)):
                graph.fields.append({
                    'name': name,
                    'definition': 'point',
                    'shape': [1],
                    'data': data
                })

    # Updated the function to run on headless system
    def set_cylinder_colors(self, edge_color=None, scalar_color_name=None, cmap=None, cmap_range=None, update=True, log_color=None):
        import matplotlib.pyplot as plt


        if scalar_color_name is not None:
            self.scalar_color_name = scalar_color_name
        if cmap is not None:
            self.cmap = cmap
        if cmap_range is not None:
            self.cmap_range = cmap_range
        if log_color is not None:
            self.log_color = log_color
        if edge_color is not None:
            self.edge_color = edge_color
            self.color = edge_color

        nedge = self.graph.nedge
        nedgepoint = self.graph.nedgepoint
        sind = self.cylinder_inds

        # Step 1: Grab edge_color from scalar name if not provided
        if edge_color is None and self.edge_color is None:
            scalars = self.graph.get_scalars()
            scalarNames = [x['name'] for x in scalars]
            for name in scalarNames:
                print(" -", name)
            if self.scalar_color_name in scalarNames:
                self.edge_color = self.graph.get_data(self.scalar_color_name)
            else:
                self.edge_color = np.ones(nedgepoint)
        elif edge_color is not None:
            self.edge_color = edge_color
        elif self.edge_color is not None:
            pass
        else:
            self.edge_color = np.ones(nedgepoint)

        if self.edge_color is None or not np.any(np.isfinite(self.edge_color)):
            self.edge_color = np.ones(nedgepoint)

        # Step 2: Log scaling if needed
        if self.log_color:
            self.edge_color = np.abs(self.edge_color)
            self.edge_color[self.edge_color <= 0.] = 1e-12
            self.edge_color = np.log(self.edge_color)

        # Step 3: Normalize colormap range
        if self.cmap_range is None:
            self.cmap_range = [None, None]
        self.cmap_range = arr(self.cmap_range)

        if self.cmap_range[0] is None or not np.isfinite(self.cmap_range[0]):
            self.cmap_range[0] = np.nanmin(self.edge_color)
        if self.cmap_range[1] is None or not np.isfinite(self.cmap_range[1]):
            self.cmap_range[1] = np.nanmax(self.edge_color)

        if self.cmap_range[0] == self.cmap_range[1]:
            self.cmap_range[1] += 1.0

        if self.cmap_range[0] > self.cmap_range[1]:
            print('[WARNING] Invalid cmap_range detected.')

        # Step 4: Generate RGB color values
        cols = None
        if self.scalar_color_name is not None:
            if self.scalar_color_name == 'VesselType':
                cols = np.zeros([nedgepoint, 3])
                s_art = np.where(self.edge_color == 0)
                cols[s_art[0], :] = [1., 0., 0.]
                s_vei = np.where(self.edge_color == 1)
                cols[s_vei[0], :] = [0., 0., 1.]
                s_cap = np.where(self.edge_color == 2)
                cols[s_cap[0], :] = [0.5, 0.5, 0.5]
                s_oth = np.where((self.edge_color > 2) | (self.edge_color < 0))
                cols[s_oth[0], :] = [1., 1., 1.]
            else:
                cmapObj = plt.cm.get_cmap(self.cmap)
                edge_color_clean = np.nan_to_num(self.edge_color, nan=self.cmap_range[0])
                col_inds = np.clip((edge_color_clean - self.cmap_range[0]) / (self.cmap_range[1] - self.cmap_range[0]), 0., 1.)
                cols = cmapObj(col_inds)[:, 0:3]
        else:
            cols = self.color
            if cols is not None and len(cols.shape) == 2 and cols.shape[-1] == 3:
                if np.max(cols) > 1.:
                    cols = cols / np.max(cols)

        if cols is None:
            print("[ERROR] Color assignment failed. Defaulting to white.")
            cols = np.ones((nedgepoint, 3))

        # Step 5: Apply highlights
        epi = self.graph.edgepoint_edge_indices()
        if len(self.edge_highlight) > 0:
            self.edge_highlight = arr(self.edge_highlight)
            for e in np.unique(self.edge_highlight):
                cols[epi == e] = self.highlight_color

        # Step 6: Paint cylinders
        for i in sind[0]:
            cyl = self.cylinders[i]
            if cyl is not None:
                if self.engine == 'open3d':
                    cyl.paint_uniform_color(cols[i])
                elif self.engine == 'pyvista':
                    pass

        self.combine_cylinders()

        #for s in self.graph.get_scalars():
            #print(" -", s['name'])

        # Optional: highlight nodes
        if len(self.node_highlight) > 0:
            nodes = self.graph.get_data('VertexCoordinates')
            self.vis.add_point_cloud(nodes[self.node_highlight], color=[1., 1., 1.])

        if update:
            self.update()


    def add_torus(self,centre=arr([0.,0.,0.]),color=arr([1.,1.,1.]),**kwargs):
        if self.headless:
            return
        torus = o3d.geometry.TriangleMesh.create_torus(**kwargs) # torus_radius=1.0, tube_radius=0.5, radial_resolution=30, tubular_resolution=20
        # Simple translation (TODO: Add rotation, etc.)
        torus = torus.translate(centre, relative=False)
        torus.paint_uniform_color(color)
        if self.additional_meshes is None:
            self.additional_meshes = torus
            self.vis.add_geometry(self.additional_meshes)
        else:
            self.additional_meshes += torus
        self.update()

    def get_numpy_image(self):
    # Only if you're using Open3D and offscreen rendering is available
        img = self.vis.capture_screen_float_buffer(do_render=True)
        img_np = np.asarray(img)
        img_np = (img_np * 255).astype(np.uint8)
        return img_np

    def add_sphere(self,centre=arr([0.,0.,0.]),color=arr([1.,1.,1.]),**kwargs):
        # kwargs: radius, resolution
        if self.headless:
            return
        sphere = o3d.geometry.TriangleMesh.create_sphere(**kwargs) # torus_radius=1.0, tube_radius=0.5, radial_resolution=30, tubular_resolution=20
        # Simple translation (TODO: Add rotation, etc.)
        sphere = sphere.translate(centre, relative=False)
        sphere.paint_uniform_color(color)
        if self.additional_meshes is None:
            self.additional_meshes = sphere
            self.vis.add_geometry(self.additional_meshes)
        else:
            self.additional_meshes += sphere
        self.update()
        
    def add_cylinder(self,centre=arr([0.,0.,0.]),color=arr([1.,1.,1.]),**kwargs):
        if self.headless:
            return
        cyl = o3d.geometry.TriangleMesh.create_cylinder(**kwargs)
        # Simple translation (TODO: Add rotation, etc.)
        cyl = cyl.translate(centre, relative=False)
        cyl.paint_uniform_color(color)
        # For now, add it in to the combined mesh. TODO: Have a dedicated set of additional meshes
        if self.additional_meshes is None:
            self.additional_meshes = cyl
            self.vis.add_geometry(self.additional_meshes)
        else:
            self.additional_meshes += cyl
        self.update()
        
    def add_mesh(self,mesh,color=arr([1.,1.,1.]),**kwargs):
        if self.headless:
            return
        mesh.paint_uniform_color(color)
        # For now, add it in to the combined mesh. TODO: Have a dedicated set of additional meshes
        if self.additional_meshes is None:
            self.additional_meshes = mesh
            self.vis.add_geometry(self.additional_meshes)
        else:
            self.additional_meshes += mesh
        self.update()
        
    def add_point_cloud(self,points,color=arr([1.,1.,1.]),**kwargs):
        if self.headless:
            return
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        
        if color.shape[0]==points.shape[0] and color.shape[1]==3:
            pointcloud.colors = o3d.utility.Vector3dVector(color)
        else:
            pointcloud.paint_uniform_color(color)
            
        #material = o3d.visualization.rendering.Material()
        #material.shader = "defaultUnlit"
        #material.base_color = color
        #material.point_size = 6
            
        # For now, add it in to the combined mesh. TODO: Have a dedicated set of additional meshes
        if self.additional_meshes is None:
            self.additional_meshes = pointcloud
            self.vis.add_geometry(self.additional_meshes)
        else:
            self.additional_meshes += pointcloud
        self.update()
        
    def create_plot_cylinders(self):
        if self.headless:
            return
    
        nc = self.graph.get_data('VertexCoordinates')
        points = self.graph.get_data('EdgePointCoordinates')
        npoints = self.graph.get_data('NumEdgePoints')
        conns = self.graph.get_data('EdgeConnectivity')
        radField = self.graph.get_radius_field()
        if radField is None:
            print('Could not locate vessel radius data!')
            radii = np.ones(points.shape[0])
        else:
            radii = radField['data']
    
        nedge = self.graph.nedge
        if self.edge_filter is None:
            self.edge_filter = np.ones(conns.shape[0],dtype='bool')
        if self.node_filter is None:
            self.node_filter = np.ones(nc.shape[0],dtype='bool')

        self.cylinders = np.empty(self.graph.nedgepoint,dtype='object') # [None]*self.graph.nedgepoint
            
        print('Preparing graph (creating cylinders)...')
        # Create cylinders
        excluded = []
        for i in trange(nedge):
            excl = True
            
            if self.edge_filter[i] and self.node_filter[conns[i,0]] and self.node_filter[conns[i,1]]:
                i0 = np.sum(npoints[:i])
                i1 = i0+npoints[i]
                coords = points[i0:i1]
                rads = radii[i0:i1]

                if np.any(rads>=self.min_radius): # (self.domain_radius is None or np.any(np.linalg.norm(coords-self.domain_centre)<=self.domain_radius)):
                    for j in range(1,coords.shape[0]):
                        if rads[j]>=self.min_radius:
                            x0,x1 = coords[j-1],coords[j]
                            vec = x1-x0
                            height = np.linalg.norm(x1-x0)
                            
                            inside,intersection = self.inside_domain(x0,start_coord=x1)
                            if height>0. and np.isfinite(height) and inside: # (self.domain_radius is None or (np.linalg.norm(x0-self.domain_centre<=self.domain_radius) and np.linalg.norm(x1-self.domain_centre<=self.domain_radius))):
                                vec = vec / height
                                if rads[j]<20. and self.radius_based_resolution:
                                    resolution = 4
                                else:
                                    resolution = self.cyl_res
                                
                                if self.fixed_radius is not None:
                                    rad_cur = self.fixed_radius    
                                elif self.radius_scale!=1.:
                                    rad_cur = rads[j] * self.radius_scale
                                else:
                                    rad_cur = rads[j]
                                    
                                if self.engine=='open3d':
                                    cyl = o3d.geometry.TriangleMesh.create_cylinder(height=height,radius=rad_cur, resolution=resolution)
                                    translation = x0 + vec*height*0.5
                                    cyl = cyl.translate(translation, relative=False)
                                    axis, angle = align_vector_to_another(np.asarray([0.,0.,1.]), vec)
                                    if angle!=0.:
                                        axis_a = axis * angle
                                        cyl = cyl.rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cyl.get_center()) 

                                    # Default - paint white
                                    cyl.paint_uniform_color([0.,0.,0.])
                                    
                                    self.cylinders[i0+j] = cyl
                                    
                                elif self.engine=='pyvista':
                                    pass
                                    #poly = pv.PolyData()
                                    #poly.points = coords
                                    #the_cell = np.arange(0, len(coords), dtype=np.int_)
                                    #the_cell = np.insert(the_cell, 0, len(coords))
                                    #poly.lines = the_cell
                                    #poly['radius'] = rads
                                    ##tube = poly.tube(radius=rads[0],n_sides=3) # scalars='stuff', 
                                    #tube = pv.Spline(coords, coords.shape[0]).tube(radius=rads[0])
                                    ##tube['color'] = np.linspace(1,1,tube.n_points)
                                    #self.cylinders[i0+j] = tube
                                    
                                excl = False

        # Combine all individual cylinders into a single mesh
        cyl_list = [cyl for cyl in self.cylinders if cyl is not None]
        if cyl_list:
            self.cylinders_combined = cyl_list[0]
            for cyl in cyl_list[1:]:
                self.cylinders_combined += cyl
        else:
            self.cylinders_combined = None


        self.cylinder_inds = np.where(self.cylinders)
        #breakpoint()
        #o3d.visualization.draw_geometries(self.cylinders[self.cylinders!=None][:50])
    
    # Updated the function to run on headless mode 
    def combine_cylinders(self):
        if self.headless:
            return
    
        if self.engine=='open3d':
            if self.vis is not None:
                self.vis.remove_geometry(self.cylinders_combined)
        
            # Combine (select active cylinder entries)
            sind = self.cylinder_inds
            if len(sind[0])>2:
                # Sum first two - otherwise combined variable becomes first cylinder reference
                self.cylinders_combined = self.cylinders[sind[0][0]] + self.cylinders[sind[0][1]]
                for cyl in self.cylinders[sind[0][2:]]:
                    if cyl is not None:
                        self.cylinders_combined += cyl
            elif len(sind[0])==2:
                self.cylinders_combined = self.cylinders[sind[0][0]] + self.cylinders[sind[0][1]] 
            elif len(sind[0])==1:
                self.cylinders_combined = self.cylinders[sind[0][0]] 
                
            if self.vis is not None:
                self.vis.add_geometry(self.cylinders_combined)
                
        elif self.engine=='pyvista':
            pass
            #blocks = pv.MultiBlock(self.cylinders[self.cylinder_inds].tolist())
            #self.cylinders_combined = blocks.combine()
            #self.vis.add_mesh(self.cylinders_combined, smooth_shading=True, scalar_bar_args={'title':self.scalar_color_name}) # scalars='length', 
            #self.vis.show()
    
    def create_plot_window(self, bgcolor=None, win_width=None, win_height=None):

        # Update window size or background color if specified
        if win_width is not None:
            self.win_width = win_width
        if win_height is not None:
            self.win_height = win_height
        if bgcolor is not None:
            self.bgcolor = bgcolor

        # Open3D Engine (Headless Mode) 
        if self.engine == 'open3d':
            self.headless = True  # We are running without GUI display

            # Export colored .ply of vessels
            if self.cylinders_combined is not None:
                self.export_colored_ply("cylinders_combined_colored.ply")
            else:
                print("[WARNING] No cylinders_combined found. Skipping PLY export.")
        elif self.engine == 'pyvista':
            pass  # You mentioned PyVista is not working, so leave this blank or disabled

    def show_plot(self):
        if self.vis is not None and self.headless==False:
            self.vis.run()
            self.vis.destroy_window()
        
    def update(self):
        if self.engine=='open3d':
            if self.vis is not None and self.headless==False:
                meshes = []
                #breakpoint()
                if self.additional_meshes is not None:
                    self.vis.update_geometry(self.additional_meshes)
                if self.cylinders_combined is not None: 
                    meshes.append(self.cylinders_combined)
                    self.vis.update_geometry(self.cylinders_combined)

    # Updated the function to run in headless mode
    def screen_grab(self, filename=None, return_image=False):
        if self.headless or self.vis is None:
            #print("screen_grab skipped: Headless mode active or vis is None.")
            return None

        image = self.vis.capture_screen_float_buffer(True)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        if filename is not None:
            import imageio
            imageio.imwrite(filename, image_np)
        if return_image:
            return image_np

    def destroy_window(self):
        if self.engine=='open3d':
            if self.vis is not None:
                control = self.vis.get_view_control()
                self.vis.destroy_window()  
                del control
                del self.vis            
    # Updated the function to run in headless mode
    def export_colored_ply(self, filename="cylinders_combined_colored.ply"):
        if self.cylinders_combined is None:
            print("No cylinders to export.")
            return

        # Ensure cylinder colors are up-to-date
        self.set_cylinder_colors(scalar_color_name='Flow',cmap='jet',log_color=True,update=False)

        # Build a flat list of vertex colors from all individual cylinders
        colors = []
        for cyl in self.cylinders:
            if cyl is not None:
                verts = np.asarray(cyl.vertices)
                col = cyl.vertex_colors[0] if len(cyl.vertex_colors) else [1, 1, 1]
                colors.extend([col] * verts.shape[0])

        # Assign the color data to the combined mesh
        self.cylinders_combined.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Write to PLY file
        o3d.io.write_triangle_mesh(filename, self.cylinders_combined)
        #print(f"[SAVE] Colored PLY written: {filename}")
    