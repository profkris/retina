import argparse

from pymira import spatialgraph, reanimate, combine_graphs, amirajson
from pymira.spatialgraph import update_array_index,delete_vertices,GVars
from pymira.csv2amira import csv2amira
import pathlib
import pandas as pd

import os
from os.path import join
import shutil
import numpy as np
from numpy import asarray as arr
from tqdm import trange, tqdm

from retinasim.capillary_bed import direct_arterial_venous_connection, voronoi_capillary_bed, remove_arteriole  
from retinasim.utility import join_feeding_vessels, make_dir, remove_endpoint_nodes, remove_all_endpoints, identify_inlet_outlet, reanimate_sim, filter_graph_by_radius, make_dir, create_directories
from retinasim.eye import Eye
from retinasim import lsystem, inject, vascular, sine_interpolate
from retinasim.scripts.create_enface import create_enface
from retinasim.scripts.create_surface import create_surface
from retinasim.scripts.project_graph_to_surface import project_to_surface
from retinasim.embed import embed
from retinasim.vascular import run_sim         
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

"""def generate_lsystem(opath=None,lpath=None,gfile=None,screen_grab=True,eye=None):

    
    #Create L-system seed networks (arterial and venous, with upper and lower retina segments created separately)
    

    lsystem.simulate_cco_seed(prefix='',params=None,max_cycles=5,path=lpath,plot=False,dataPath=opath,eye=eye)
    mfiles = ['retina_artery_lower.am','retina_vein_lower.am','retina_artery_upper.am','retina_vein_upper.am']
    
    combined_graph = combine_graphs.combine_cco(opath,[join(lpath,m) for m in mfiles],gfile)
    
    # Screen grab result
    if screen_grab:
        domain_radius = (eye.domain[0,1]-eye.domain[0,0])/2.
        vis = combined_graph.plot_graph(show=False,block=False,domain_radius=domain_radius*2.2,bgcolor=[1.,1.,1.])
        vis.add_torus(centre=eye.optic_disc_centre,torus_radius=eye.optic_disc_radius,tube_radius=20.,color=[0.,0.,0.]) # radial_resolution=30, tubular_resolution=20
        vis.add_torus(centre=eye.fovea_centre,torus_radius=eye.fovea_radius,tube_radius=20.,color=[0.,0.,0.])
        vis.add_torus(centre=eye.macula_centre,torus_radius=eye.macula_radius,tube_radius=20.,color=[0.,0.,0.])
        
        centre = eye.occular_centre.copy()
        centre[2] = 0.
        vis.add_torus(centre=centre,torus_radius=domain_radius,tube_radius=20.,color=[0.,0.,0.])
        fname = join(opath,"lsystem.png")
        vis.screen_grab(fname)
        vis.destroy_window()
        print(f'L-system grab file: {fname}')
        
    return combined_graph, mfiles


def generate_lsystem(opath=None, lpath=None, gfile=None, screen_grab=True, eye=None):
    
    #Create L-system seed networks (arterial and venous, with upper and lower retina segments created separately)
    
    lsystem.simulate_cco_seed(prefix='', params=None, max_cycles=5, path=lpath, plot=False, dataPath=opath, eye=eye)
    mfiles = ['retina_artery_lower.am', 'retina_vein_lower.am', 'retina_artery_upper.am', 'retina_vein_upper.am']
    
    combined_graph = combine_graphs.combine_cco(opath, [join(lpath, m) for m in mfiles], gfile)
    
    # Screen grab result
    if screen_grab:
        try:
            import open3d as o3d
            from open3d.visualization import rendering

            domain_radius = (eye.domain[0, 1] - eye.domain[0, 0]) / 2.0

            # Add toruses as meshes
            vis = combined_graph.plot_graph(show=False, block=False, domain_radius=domain_radius * 2.2, bgcolor=[1., 1., 1.])
            vis.add_torus(centre=eye.optic_disc_centre, torus_radius=eye.optic_disc_radius, tube_radius=20., color=[0., 0., 0.])
            vis.add_torus(centre=eye.fovea_centre, torus_radius=eye.fovea_radius, tube_radius=20., color=[0., 0., 0.])
            vis.add_torus(centre=eye.macula_centre, torus_radius=eye.macula_radius, tube_radius=20., color=[0., 0., 0.])
            centre = eye.occular_centre.copy()
            centre[2] = 0.
            vis.add_torus(centre=centre, torus_radius=domain_radius, tube_radius=20., color=[0., 0., 0.])

            # Headless image rendering using OffscreenRenderer
            mesh = vis.cylinders_combined
            if mesh is not None:
                renderer = rendering.OffscreenRenderer(1024, 768)
                mat = rendering.MaterialRecord()
                mat.shader = "defaultLit"
                renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
                renderer.scene.add_geometry("cylinders", mesh, mat)

                # Save to PNG
                img = renderer.render_to_image()
                fname = join(opath, "lsystem.png")
                o3d.io.write_image(fname, img)
                print(f'L-system grab saved (headless): {fname}')
            else:
                print('No mesh to render. lsystem.png not created.')

        except Exception as e:
            print(f"[!] Failed to generate lsystem.png headlessly: {e}")
    
    return combined_graph, mfiles

def generate_lsystem(opath=None, lpath=None, gfile=None, screen_grab=True, eye=None):
    
    #Create L-system seed networks (arterial and venous, with upper and lower retina segments created separately).
    #Saves geometry data to PLY files instead of rendering images (safe for headless environments).

    lsystem.simulate_cco_seed(prefix='', params=None, max_cycles=5, path=lpath, plot=False, dataPath=opath, eye=eye)
    mfiles = ['retina_artery_lower.am', 'retina_vein_lower.am', 'retina_artery_upper.am', 'retina_vein_upper.am']

    combined_graph = combine_graphs.combine_cco(opath, [join(lpath, m) for m in mfiles], gfile)

    # Save the combined geometry to PLY without GUI rendering
    if screen_grab:
        try:
            # Get geometry from the combined graph (like cylinders or mesh)
            mesh = combined_graph.cylinders_combined if hasattr(combined_graph, 'cylinders_combined') else None
            if mesh is not None:
                ply_path = join(opath, "lsystem_output.ply")
                o3d.io.write_triangle_mesh(ply_path, mesh)
                print(f"Saved headless L-system mesh to: {ply_path}")
            else:
                print("combined_graph does not have mesh geometry to export.")
        except Exception as e:
            print(f"Failed to write .ply for L-system: {e}")

    return combined_graph, mfiles

def generate_lsystem(opath=None, lpath=None, gfile=None, screen_grab=True, eye=None):
    
    #Create L-system seed networks (arterial and venous, with upper and lower retina segments created separately)
    

    lsystem.simulate_cco_seed(prefix='', params=None, max_cycles=5, path=lpath, plot=False, dataPath=opath, eye=eye)
    mfiles = ['retina_artery_lower.am', 'retina_vein_lower.am', 'retina_artery_upper.am', 'retina_vein_upper.am']

    combined_graph = combine_graphs.combine_cco(opath, [join(lpath, m) for m in mfiles], gfile)

    # Convert to LineSet (if applicable)
    if hasattr(combined_graph, 'nodes') and hasattr(combined_graph, 'edges'):
        try:
            points = np.array(combined_graph.nodes)  # shape: (N, 3)
            lines = np.array(combined_graph.edges)   # shape: (M, 2)
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines)
            )
            fname = join(opath, "lsystem_output_graph.ply")
            o3d.io.write_line_set(fname, line_set)
            print(f'Saved combined_graph as line set: {fname}')
        except Exception as e:
            print(f"Failed to export combined_graph as LineSet: {e}")
    else:
        print("combined_graph does not have mesh geometry or required attributes.")

    return combined_graph, mfiles
"""
#---------------------------------------Updated Version-------------------------------------#
def generate_lsystem(opath=None, lpath=None, gfile=None, screen_grab=True, eye=None):
    
    #Create L-system seed networks (arterial and venous, with upper and lower retina segments created separately)
        # Step 1: Run L-system simulation
    lsystem.simulate_cco_seed(
        prefix='',
        params=None,
        max_cycles=5,
        path=lpath,
        plot=False,
        dataPath=opath,
        eye=eye
    )

    mfiles = [
        'retina_artery_lower.am',
        'retina_vein_lower.am',
        'retina_artery_upper.am',
        'retina_vein_upper.am'
    ]

    # Step 2: Combine graphs
    combined_graph = combine_graphs.combine_cco(
        opath,
        [join(lpath, m) for m in mfiles],
        gfile
    )

    # Step 3: Try to save mesh as .ply instead of screen grabbing
    if screen_grab:
        try:
            print("[INFO] Attempting headless mesh export instead of screen grab...")

            vis = combined_graph.plot_graph(show=False, block=False)

            if hasattr(vis, "cylinders_combined"):
                mesh = vis.cylinders_combined
                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    out_path = join(opath, "lsystem_output_mesh.ply")
                    o3d.io.write_triangle_mesh(out_path, mesh)
                    print(f"[SUCCESS] Mesh saved to: {out_path}")
                else:
                    print("[WARNING] cylinders_combined is not a TriangleMesh object.")
            else:
                print("[WARNING] combined_graph.plot_graph() did not return 'cylinders_combined'.")

        except Exception as e:
            print(f"[ERROR] Could not generate headless .ply export: {e}")

    return combined_graph, mfiles
#-----------------------------------------------Updated Version------------------------------------------#




def vascular_config(Actions="coarse;macula;resolve",FrozenFactor=0,eye=None,cfile=None, \
                    ArteryInPath=None,VeinInPath=None,ArteryOutPath=None,VeinOutPath=None,macula_fraction=1.,FlowFactor=2.,quad='upper',Frozen=None,
                    MinTerminalLength=0,TargetStep=0,set_quad=True,regulate_filepaths=False):

    """
    Run vascular on upper and lower sections separately
    The output path (opath) the cco directory by default
    """

    # Edit config files
    if cfile is None:
        cfile = join(opath,'retina_artery_discs_params.json')
    vcfg = vascular.VascularConfig(eye)
    
    vcfg.params['ArteryInPath'] = ArteryInPath
    vcfg.params['VeinInPath'] = VeinInPath
    vcfg.params['ArteryOutPath'] = ArteryOutPath
    vcfg.params['VeinOutPath'] = VeinOutPath
    
    if eye is not None:
        vcfg.params['Domain']['MaculaRadius'] = eye.macula_radius*macula_fraction
    else:
        vcfg.params['Domain']['MaculaRadius'] = 500.
    vcfg.params['Major']['PreSpacing'] = 5000.
    # Endpoint spacing used for the coursest scale
    vcfg.params['Major']['SpacingMax'] = 3000.
    # Endpoint spacing used for the finest scale (150um by default; change to 500 for faster testing)
    vcfg.params['Major']['SpacingMin'] = 500. #150.
    # Number of length scales to model (spread across spacing max and min range above)
    vcfg.params['Major']['Refinements'] = 5
    vcfg.params['Major']['FrozenFactor'] = FrozenFactor
    vcfg.params["Actions"] = Actions
    # Macula resolutions
    vcfg.params['Macula']['SpacingMax'] = vcfg.params['Domain']['MaculaRadius'] / 2
    vcfg.params['Macula']['SpacingMin'] = 150. #vcfg.params['Major']['SpacingMin']
    vcfg.params['Macula']['Refinements'] = 3
    vcfg.params['Macula']['FlowFactor'] = FlowFactor
    vcfg.params['Optimizer']['Frozen'] = Frozen
    vcfg.params['Optimizer']['TargetStep'] = TargetStep
    vcfg.params['Optimizer']['MinTerminalLength'] = MinTerminalLength
    vcfg.quad = quad
    
    print(f'Written param file: {cfile}')
    vcfg.write(cfile,set_quad=set_quad,regulate_filepaths=regulate_filepaths)
    
    return cfile

"""   
def vascular_upper_lower(opath=None,lpath=None,input_graphs=None,convert_to_json=True,join_feeding=True,eye=None,
                         ArteryInPath_lower=None,VeinInPath_lower=None,ArteryOutPath_lower=None,VeinOutPath_lower=None,
                         ArteryInPath_upper=None,VeinInPath_upper=None,ArteryOutPath_upper=None,VeinOutPath_upper=None,
                         macula_fraction=1.,FlowFactor=1.,Actions="coarse;macula;resolve",FrozenFactor=0,
                         quiet=False,combine=True,plot=True):
                         
    
    Use Retina library to generate vessel networks
    This requires generation of a configuration file and iterating over the upper and lower retina segments.
    
    
    # Convert amira spatial graph files to json    
    # Order must be artery,lower; vein,lower; artery,upper; vein,lower
    default_input_graphs = [join(lpath,x) for x in ['retina_artery_lower.am','retina_vein_lower.am','retina_artery_upper.am','retina_vein_upper.am']]
    input_graphs = [ArteryInPath_lower,VeinInPath_lower,ArteryInPath_upper,VeinInPath_upper]
    for i,f in enumerate(input_graphs):
        if f is None:
            input_graphs[i] = amirajson.convert(default_input_graphs[i])
        elif f.endswith('.am'):
            input_graphs[i] = amirajson.convert(f)
        elif f.endswith('.json'):
            pass
        else:
            breakpoint()
    
    # Location for inputs
    ArteryInPath_lower = input_graphs[0]
    VeinInPath_lower = input_graphs[1]
    ArteryInPath_upper = input_graphs[2]
    VeinInPath_upper = input_graphs[3]
        
    # Define output locations
    if ArteryOutPath_lower is None:
        ArteryOutPath_lower = join(opath,os.path.basename(input_graphs[0].replace('.json','.csv'))) # join(opath,f'retina_artery_lower_cco.csv')
    if VeinOutPath_lower is None:
        VeinOutPath_lower = join(opath,os.path.basename(input_graphs[1].replace('.json','.csv'))) # join(opath,f'retina_vein_lower_cco.csv')
    if ArteryOutPath_upper is None:
        ArteryOutPath_upper = join(opath,os.path.basename(input_graphs[2].replace('.json','.csv'))) #  #join(opath,f'retina_artery_upper_cco.csv')
    if VeinOutPath_upper is None:
        VeinOutPath_upper = join(opath,os.path.basename(input_graphs[3].replace('.json','.csv'))) #  #join(opath,f'retina_vein_upper_cco.csv')        
        
    amfiles = []
    # Loop over upper and lower retina
    for i,quad in enumerate(['upper','lower']):
        cfile = join(opath,f'retina_{quad}_discs_params.json')
        
        if quad=='upper':
            ArteryInPath = ArteryInPath_upper
            VeinInPath = VeinInPath_upper
            ArteryOutPath = ArteryOutPath_upper
            VeinOutPath = VeinOutPath_upper
        elif quad=='lower':
            ArteryInPath = ArteryInPath_lower
            VeinInPath = VeinInPath_lower
            ArteryOutPath = ArteryOutPath_lower
            VeinOutPath = VeinOutPath_lower
            
        vascular_config(eye=eye,cfile=cfile, \
                        ArteryInPath=ArteryInPath,VeinInPath=VeinInPath, \
                        ArteryOutPath=ArteryOutPath,VeinOutPath=VeinOutPath, \
                        macula_fraction=macula_fraction,FlowFactor=FlowFactor, \
                        Actions=Actions,FrozenFactor=FrozenFactor,quad=quad)

        res = run_sim(cfile,quiet=quiet)
        if res!=0:
            return False, amfiles, cfile, None
            
        # Convert csv output to amira graph format
        print('Converting CSV to AM: {}'.format(ArteryOutPath))
        ArteryOutPath_am = csv2amira(ArteryOutPath)
        print('Converting CSV to AM: {}'.format(VeinOutPath))
        VeinOutPath_am = csv2amira(VeinOutPath)
        
        amfiles.append(ArteryOutPath_am)
        amfiles.append(VeinOutPath_am)        
        
    # Combine upper and lower retina outputs into a single network
    if combine:
        # Combine graphs into one file
        ofile = 'retina_cco.am'
        graph = combine_graphs.combine_cco(opath,amfiles,ofile)
        
        #graph = spatialgraph.SpatialGraph()
        #graph.read(join(opath,join(opath,ofile)))
    
        if join_feeding:
            print('Joining feeding vessels...')
            graph = join_feeding_vessels(graph)
            graph.write(join(opath,ofile))
            
        if plot:
            win_width,win_height = 6000,6000
            vis = graph.plot_graph(scalar_color_name='VesselType',bgcolor=[1.,1.,1.],show=False,block=False,win_width=win_width,win_height=win_height)
            vis.screen_grab(join(opath,f'artery_vein_cco.png'))
            vis.add_torus(centre=eye.optic_disc_centre,torus_radius=eye.optic_disc_radius,tube_radius=20.,color=[0.,0.,0.])
            vis.add_torus(centre=eye.fovea_centre,torus_radius=eye.fovea_radius,tube_radius=20.,color=[0.,0.,0.])
            vis.add_torus(centre=eye.macula_centre,torus_radius=eye.macula_radius,tube_radius=20.,color=[0.,0.,0.])
            domain_radius = (eye.domain[0,1]-eye.domain[0,0])/2.
            centre = eye.occular_centre.copy()
            centre[2] = 0.
            vis.add_torus(centre=centre,torus_radius=domain_radius,tube_radius=20.,color=[0.,0.,0.])
            vis.screen_grab(join(opath,f'artery_vein_overlay_cco.png'))
            vis.destroy_window()            
    
        return True, amfiles, ofile, graph
    else:
        return True, amfiles, None, None

"""
#------------------Working
def vascular_upper_lower(opath=None, lpath=None, input_graphs=None, convert_to_json=True, join_feeding=True, eye=None,
                         ArteryInPath_lower=None, VeinInPath_lower=None, ArteryOutPath_lower=None, VeinOutPath_lower=None,
                         ArteryInPath_upper=None, VeinInPath_upper=None, ArteryOutPath_upper=None, VeinOutPath_upper=None,
                         macula_fraction=1., FlowFactor=1., Actions="coarse;macula;resolve", FrozenFactor=0,
                         quiet=False, combine=True, plot=True):
    
    #Use Retina library to generate vessel networks.
    #Display-independent version: exports .ply instead of requiring DISPLAY variable.
    
    # Step 1: Convert input .am files to .json if needed
    default_input_graphs = [join(lpath, x) for x in [
        'retina_artery_lower.am', 'retina_vein_lower.am',
        'retina_artery_upper.am', 'retina_vein_upper.am'
    ]]
    input_graphs = [ArteryInPath_lower, VeinInPath_lower, ArteryInPath_upper, VeinInPath_upper]
    for i, f in enumerate(input_graphs):
        if f is None:
            input_graphs[i] = amirajson.convert(default_input_graphs[i])
        elif f.endswith('.am'):
            input_graphs[i] = amirajson.convert(f)
        elif f.endswith('.json'):
            pass
        else:
            breakpoint()

    ArteryInPath_lower, VeinInPath_lower = input_graphs[0], input_graphs[1]
    ArteryInPath_upper, VeinInPath_upper = input_graphs[2], input_graphs[3]

    # Step 2: Define output CSV paths
    if ArteryOutPath_lower is None:
        ArteryOutPath_lower = join(opath, os.path.basename(input_graphs[0].replace('.json', '.csv')))
    if VeinOutPath_lower is None:
        VeinOutPath_lower = join(opath, os.path.basename(input_graphs[1].replace('.json', '.csv')))
    if ArteryOutPath_upper is None:
        ArteryOutPath_upper = join(opath, os.path.basename(input_graphs[2].replace('.json', '.csv')))
    if VeinOutPath_upper is None:
        VeinOutPath_upper = join(opath, os.path.basename(input_graphs[3].replace('.json', '.csv')))

    amfiles = []

    # Step 3: Run simulations for each quadrant
    for i, quad in enumerate(['upper', 'lower']):
        cfile = join(opath, f'retina_{quad}_discs_params.json')
        if quad == 'upper':
            ArteryInPath, VeinInPath = ArteryInPath_upper, VeinInPath_upper
            ArteryOutPath, VeinOutPath = ArteryOutPath_upper, VeinOutPath_upper
        else:
            ArteryInPath, VeinInPath = ArteryInPath_lower, VeinInPath_lower
            ArteryOutPath, VeinOutPath = ArteryOutPath_lower, VeinOutPath_lower

        vascular_config(eye=eye, cfile=cfile,
                        ArteryInPath=ArteryInPath, VeinInPath=VeinInPath,
                        ArteryOutPath=ArteryOutPath, VeinOutPath=VeinOutPath,
                        macula_fraction=macula_fraction, FlowFactor=FlowFactor,
                        Actions=Actions, FrozenFactor=FrozenFactor, quad=quad)

        res = run_sim(cfile, quiet=quiet)
        if res != 0:
            return False, amfiles, cfile, None

        # Convert to .am format
        print(f'Converting CSV to AM: {ArteryOutPath}')
        ArteryOutPath_am = csv2amira(ArteryOutPath)
        print(f'Converting CSV to AM: {VeinOutPath}')
        VeinOutPath_am = csv2amira(VeinOutPath)
        amfiles.append(ArteryOutPath_am)
        amfiles.append(VeinOutPath_am)

    # Step 4: Combine and optionally export visualization
    if combine:
        ofile = 'retina_cco.am'
        graph = combine_graphs.combine_cco(opath, amfiles, ofile)

        if join_feeding:
            print("Joining feeding vessels...")
            graph = join_feeding_vessels(graph)
            graph.write(join(opath, ofile))

        if plot:
            try:
                print("[INFO] Exporting headless mesh instead of screen rendering...")
                vis = graph.plot_graph(show=False, block=False)

                if hasattr(vis, "cylinders_combined"):
                    mesh = vis.cylinders_combined
                    if isinstance(mesh, o3d.geometry.TriangleMesh):
                        out_path = join(opath, "vascular_output_mesh.ply")
                        o3d.io.write_triangle_mesh(out_path, mesh)
                        print(f"[SUCCESS] Mesh saved to: {out_path}")
                    else:
                        print("[WARNING] cylinders_combined is not a TriangleMesh.")
                else:
                    print("[WARNING] Visualization object has no 'cylinders_combined' attribute.")
                vis.destroy_window()
            except Exception as e:
                print(f"[ERROR] Could not export mesh: {e}")

        return True, amfiles, ofile, graph

    return True, amfiles, None, None

"""
def vascular_upper_lower(opath=None, lpath=None, input_graphs=None, convert_to_json=True, join_feeding=True, eye=None,
                         ArteryInPath_lower=None, VeinInPath_lower=None, ArteryOutPath_lower=None, VeinOutPath_lower=None,
                         ArteryInPath_upper=None, VeinInPath_upper=None, ArteryOutPath_upper=None, VeinOutPath_upper=None,
                         macula_fraction=1., FlowFactor=1., Actions="coarse;macula;resolve", FrozenFactor=0,
                         quiet=False, combine=True, plot=True):

    Use Retina library to generate vessel networks.
    This requires generation of a configuration file and iterating over the upper and lower retina segments.
    

    # Convert amira spatial graph files to json    
    default_input_graphs = [join(lpath, x) for x in ['retina_artery_lower.am', 'retina_vein_lower.am', 'retina_artery_upper.am', 'retina_vein_upper.am']]
    input_graphs = [ArteryInPath_lower, VeinInPath_lower, ArteryInPath_upper, VeinInPath_upper]
    for i, f in enumerate(input_graphs):
        if f is None:
            input_graphs[i] = amirajson.convert(default_input_graphs[i])
        elif f.endswith('.am'):
            input_graphs[i] = amirajson.convert(f)
        elif f.endswith('.json'):
            pass
        else:
            breakpoint()

    ArteryInPath_lower = input_graphs[0]
    VeinInPath_lower = input_graphs[1]
    ArteryInPath_upper = input_graphs[2]
    VeinInPath_upper = input_graphs[3]

    # Define output locations
    if ArteryOutPath_lower is None:
        ArteryOutPath_lower = join(opath, os.path.basename(input_graphs[0].replace('.json', '.csv')))
    if VeinOutPath_lower is None:
        VeinOutPath_lower = join(opath, os.path.basename(input_graphs[1].replace('.json', '.csv')))
    if ArteryOutPath_upper is None:
        ArteryOutPath_upper = join(opath, os.path.basename(input_graphs[2].replace('.json', '.csv')))
    if VeinOutPath_upper is None:
        VeinOutPath_upper = join(opath, os.path.basename(input_graphs[3].replace('.json', '.csv')))

    amfiles = []

    for i, quad in enumerate(['upper', 'lower']):
        cfile = join(opath, f'retina_{quad}_discs_params.json')

        if quad == 'upper':
            ArteryInPath = ArteryInPath_upper
            VeinInPath = VeinInPath_upper
            ArteryOutPath = ArteryOutPath_upper
            VeinOutPath = VeinOutPath_upper
        elif quad == 'lower':
            ArteryInPath = ArteryInPath_lower
            VeinInPath = VeinInPath_lower
            ArteryOutPath = ArteryOutPath_lower
            VeinOutPath = VeinOutPath_lower

        vascular_config(eye=eye, cfile=cfile,
                        ArteryInPath=ArteryInPath, VeinInPath=VeinInPath,
                        ArteryOutPath=ArteryOutPath, VeinOutPath=VeinOutPath,
                        macula_fraction=macula_fraction, FlowFactor=FlowFactor,
                        Actions=Actions, FrozenFactor=FrozenFactor, quad=quad)

        res = run_sim(cfile, quiet=quiet)
        if res != 0:
            return False, amfiles, cfile, None

        print('Converting CSV to AM: {}'.format(ArteryOutPath))
        ArteryOutPath_am = csv2amira(ArteryOutPath)
        print('Converting CSV to AM: {}'.format(VeinOutPath))
        VeinOutPath_am = csv2amira(VeinOutPath)

        amfiles.append(ArteryOutPath_am)
        amfiles.append(VeinOutPath_am)

    if combine:
        ofile = 'retina_cco.am'
        graph = combine_graphs.combine_cco(opath, amfiles, ofile)

        if join_feeding:
            print('Joining feeding vessels...')
            graph = join_feeding_vessels(graph)
            graph.write(join(opath, ofile))

        if plot:
            try:
                print("[INFO] Exporting headless meshes instead of PNGs...")

                win_width, win_height = 6000, 6000
                vis = graph.plot_graph(scalar_color_name='VesselType', bgcolor=[1., 1., 1.],
                                       show=False, block=False, win_width=win_width, win_height=win_height)

                # Export mesh before adding torus overlays
                if hasattr(vis, "cylinders_combined"):
                    mesh = vis.cylinders_combined
                    if isinstance(mesh, o3d.geometry.TriangleMesh):
                        mesh_path = join(opath, "artery_vein_cco.ply")
                        o3d.io.write_triangle_mesh(mesh_path, mesh)
                        print(f"[SUCCESS] Basic mesh saved to: {mesh_path}")
                    else:
                        print("[WARNING] cylinders_combined is not a TriangleMesh.")
                else:
                    print("[WARNING] No 'cylinders_combined' in vis object.")

                # Add torus overlays
                vis.add_torus(centre=eye.optic_disc_centre, torus_radius=eye.optic_disc_radius,
                              tube_radius=20., color=[0., 0., 0.])
                vis.add_torus(centre=eye.fovea_centre, torus_radius=eye.fovea_radius,
                              tube_radius=20., color=[0., 0., 0.])
                vis.add_torus(centre=eye.macula_centre, torus_radius=eye.macula_radius,
                              tube_radius=20., color=[0., 0., 0.])
                domain_radius = (eye.domain[0, 1] - eye.domain[0, 0]) / 2.
                centre = eye.occular_centre.copy()
                centre[2] = 0.
                vis.add_torus(centre=centre, torus_radius=domain_radius,
                              tube_radius=20., color=[0., 0., 0.])

                # Export mesh after adding overlays
                if hasattr(vis, "cylinders_combined"):
                    mesh = vis.cylinders_combined
                    if isinstance(mesh, o3d.geometry.TriangleMesh):
                        mesh_path = join(opath, "artery_vein_overlay_cco.ply")
                        o3d.io.write_triangle_mesh(mesh_path, mesh)
                        print(f"[SUCCESS] Overlay mesh saved to: {mesh_path}")
                    else:
                        print("[WARNING] overlay mesh is not a TriangleMesh.")
                else:
                    print("[WARNING] No 'cylinders_combined' after torus addition.")

                vis.destroy_window()

            except Exception as e:
                print(f"[ERROR] Failed display-independent export: {e}")

        return True, amfiles, ofile, graph

    else:
        return True, amfiles, None, None



def vascular_upper_lower(opath=None, lpath=None, input_graphs=None, convert_to_json=True, join_feeding=True, eye=None,
                         ArteryInPath_lower=None, VeinInPath_lower=None, ArteryOutPath_lower=None, VeinOutPath_lower=None,
                         ArteryInPath_upper=None, VeinInPath_upper=None, ArteryOutPath_upper=None, VeinOutPath_upper=None,
                         macula_fraction=1., FlowFactor=1., Actions="coarse;macula;resolve", FrozenFactor=0,
                         quiet=False, combine=True, plot=True):
    
    Use Retina library to generate vessel networks.
    This requires generation of a configuration file and iterating over the upper and lower retina segments.
    

    # Convert amira spatial graph files to json    
    default_input_graphs = [join(lpath, x) for x in ['retina_artery_lower.am', 'retina_vein_lower.am', 'retina_artery_upper.am', 'retina_vein_upper.am']]
    input_graphs = [ArteryInPath_lower, VeinInPath_lower, ArteryInPath_upper, VeinInPath_upper]
    for i, f in enumerate(input_graphs):
        if f is None:
            input_graphs[i] = amirajson.convert(default_input_graphs[i])
        elif f.endswith('.am'):
            input_graphs[i] = amirajson.convert(f)
        elif f.endswith('.json'):
            pass
        else:
            breakpoint()

    ArteryInPath_lower = input_graphs[0]
    VeinInPath_lower = input_graphs[1]
    ArteryInPath_upper = input_graphs[2]
    VeinInPath_upper = input_graphs[3]

    # Define output locations
    if ArteryOutPath_lower is None:
        ArteryOutPath_lower = join(opath, os.path.basename(input_graphs[0].replace('.json', '.csv')))
    if VeinOutPath_lower is None:
        VeinOutPath_lower = join(opath, os.path.basename(input_graphs[1].replace('.json', '.csv')))
    if ArteryOutPath_upper is None:
        ArteryOutPath_upper = join(opath, os.path.basename(input_graphs[2].replace('.json', '.csv')))
    if VeinOutPath_upper is None:
        VeinOutPath_upper = join(opath, os.path.basename(input_graphs[3].replace('.json', '.csv')))

    amfiles = []

    for i, quad in enumerate(['upper', 'lower']):
        cfile = join(opath, f'retina_{quad}_discs_params.json')

        if quad == 'upper':
            ArteryInPath = ArteryInPath_upper
            VeinInPath = VeinInPath_upper
            ArteryOutPath = ArteryOutPath_upper
            VeinOutPath = VeinOutPath_upper
        elif quad == 'lower':
            ArteryInPath = ArteryInPath_lower
            VeinInPath = VeinInPath_lower
            ArteryOutPath = ArteryOutPath_lower
            VeinOutPath = VeinOutPath_lower

        vascular_config(eye=eye, cfile=cfile,
                        ArteryInPath=ArteryInPath, VeinInPath=VeinInPath,
                        ArteryOutPath=ArteryOutPath, VeinOutPath=VeinOutPath,
                        macula_fraction=macula_fraction, FlowFactor=FlowFactor,
                        Actions=Actions, FrozenFactor=FrozenFactor, quad=quad)

        res = run_sim(cfile, quiet=quiet)
        if res != 0:
            return False, amfiles, cfile, None

        print('Converting CSV to AM: {}'.format(ArteryOutPath))
        ArteryOutPath_am = csv2amira(ArteryOutPath)
        print('Converting CSV to AM: {}'.format(VeinOutPath))
        VeinOutPath_am = csv2amira(VeinOutPath)

        amfiles.append(ArteryOutPath_am)
        amfiles.append(VeinOutPath_am)

    if combine:
        ofile = 'retina_cco.am'
        graph = combine_graphs.combine_cco(opath, amfiles, ofile)

        if join_feeding:
            print('Joining feeding vessels...')
            graph = join_feeding_vessels(graph)
            graph.write(join(opath, ofile))

        if plot:
            try:
                print("[INFO] Exporting headless meshes instead of PNGs...")

                win_width, win_height = 6000, 6000
                vis = graph.plot_graph(scalar_color_name='VesselType', bgcolor=[1., 1., 1.],
                                       show=False, block=False, win_width=win_width, win_height=win_height)

                # Export combined mesh
                if hasattr(vis, "cylinders_combined"):
                    mesh = vis.cylinders_combined
                    if isinstance(mesh, o3d.geometry.TriangleMesh):
                        mesh_path = join(opath, "artery_vein_cco.ply")
                        o3d.io.write_triangle_mesh(mesh_path, mesh)
                        print(f"[SUCCESS] Combined mesh saved to: {mesh_path}")
                    else:
                        print("[WARNING] cylinders_combined is not a TriangleMesh.")
                else:
                    print("[WARNING] No 'cylinders_combined' in vis object.")

                # Export artery and vein separately
                if hasattr(graph, 'point_scalars_to_edge_scalars'):
                    vt = graph.point_scalars_to_edge_scalars(name='VesselType')

                    # Artery-only mesh
                    edge_filter_artery = np.zeros_like(vt, dtype=bool)
                    edge_filter_artery[vt == 1] = True
                    vis_artery = graph.plot_graph(edge_filter=edge_filter_artery, scalar_color_name='VesselType',
                                                  show=False, block=False, win_width=win_width, win_height=win_height)
                    if hasattr(vis_artery, "cylinders_combined"):
                        mesh_artery = vis_artery.cylinders_combined
                        if isinstance(mesh_artery, o3d.geometry.TriangleMesh):
                            mesh_path_artery = join(opath, "artery_cco.ply")
                            o3d.io.write_triangle_mesh(mesh_path_artery, mesh_artery)
                            print(f"[SUCCESS] Artery mesh saved to: {mesh_path_artery}")
                        else:
                            print("[WARNING] Artery mesh is not a TriangleMesh.")
                    vis_artery.destroy_window()

                    # Vein-only mesh
                    edge_filter_vein = np.zeros_like(vt, dtype=bool)
                    edge_filter_vein[vt == 2] = True
                    vis_vein = graph.plot_graph(edge_filter=edge_filter_vein, scalar_color_name='VesselType',
                                                show=False, block=False, win_width=win_width, win_height=win_height)
                    if hasattr(vis_vein, "cylinders_combined"):
                        mesh_vein = vis_vein.cylinders_combined
                        if isinstance(mesh_vein, o3d.geometry.TriangleMesh):
                            mesh_path_vein = join(opath, "vein_cco.ply")
                            o3d.io.write_triangle_mesh(mesh_path_vein, mesh_vein)
                            print(f"[SUCCESS] Vein mesh saved to: {mesh_path_vein}")
                        else:
                            print("[WARNING] Vein mesh is not a TriangleMesh.")
                    vis_vein.destroy_window()
                else:
                    print("[WARNING] graph has no VesselType data for splitting.")

                # Add torus overlays
                vis.add_torus(centre=eye.optic_disc_centre, torus_radius=eye.optic_disc_radius,
                              tube_radius=20., color=[0., 0., 0.])
                vis.add_torus(centre=eye.fovea_centre, torus_radius=eye.fovea_radius,
                              tube_radius=20., color=[0., 0., 0.])
                vis.add_torus(centre=eye.macula_centre, torus_radius=eye.macula_radius,
                              tube_radius=20., color=[0., 0., 0.])
                domain_radius = (eye.domain[0, 1] - eye.domain[0, 0]) / 2.
                centre = eye.occular_centre.copy()
                centre[2] = 0.
                vis.add_torus(centre=centre, torus_radius=domain_radius,
                              tube_radius=20., color=[0., 0., 0.])

                # Export final overlay mesh
                if hasattr(vis, "cylinders_combined"):
                    mesh = vis.cylinders_combined
                    if isinstance(mesh, o3d.geometry.TriangleMesh):
                        mesh_path = join(opath, "artery_vein_overlay_cco.ply")
                        o3d.io.write_triangle_mesh(mesh_path, mesh)
                        print(f"[SUCCESS] Overlay mesh saved to: {mesh_path}")
                    else:
                        print("[WARNING] overlay mesh is not a TriangleMesh.")
                else:
                    print("[WARNING] No 'cylinders_combined' after torus addition.")

                vis.destroy_window()

            except Exception as e:
                print(f"[ERROR] Failed display-independent export: {e}")

        return True, amfiles, ofile, graph

    else:
        return True, amfiles, None, None


def vascular_upper_lower(opath=None, lpath=None, input_graphs=None, convert_to_json=True, join_feeding=True, eye=None,
                         ArteryInPath_lower=None, VeinInPath_lower=None, ArteryOutPath_lower=None, VeinOutPath_lower=None,
                         ArteryInPath_upper=None, VeinInPath_upper=None, ArteryOutPath_upper=None, VeinOutPath_upper=None,
                         macula_fraction=1., FlowFactor=1., Actions="coarse;macula;resolve", FrozenFactor=0,
                         quiet=False, combine=True, plot=True):
    
    Use Retina library to generate vessel networks.
    Display-independent version: exports .ply instead of requiring DISPLAY variable.
    

    # Step 1: Convert input .am files to .json if needed
    default_input_graphs = [join(lpath, x) for x in [
        'retina_artery_lower.am', 'retina_vein_lower.am',
        'retina_artery_upper.am', 'retina_vein_upper.am'
    ]]
    input_graphs = [ArteryInPath_lower, VeinInPath_lower, ArteryInPath_upper, VeinInPath_upper]
    for i, f in enumerate(input_graphs):
        if f is None:
            input_graphs[i] = amirajson.convert(default_input_graphs[i])
        elif f.endswith('.am'):
            input_graphs[i] = amirajson.convert(f)
        elif f.endswith('.json'):
            pass
        else:
            breakpoint()

    ArteryInPath_lower, VeinInPath_lower = input_graphs[0], input_graphs[1]
    ArteryInPath_upper, VeinInPath_upper = input_graphs[2], input_graphs[3]

    # Step 2: Define output CSV paths
    if ArteryOutPath_lower is None:
        ArteryOutPath_lower = join(opath, os.path.basename(input_graphs[0].replace('.json', '.csv')))
    if VeinOutPath_lower is None:
        VeinOutPath_lower = join(opath, os.path.basename(input_graphs[1].replace('.json', '.csv')))
    if ArteryOutPath_upper is None:
        ArteryOutPath_upper = join(opath, os.path.basename(input_graphs[2].replace('.json', '.csv')))
    if VeinOutPath_upper is None:
        VeinOutPath_upper = join(opath, os.path.basename(input_graphs[3].replace('.json', '.csv')))

    amfiles = []

    # Step 3: Run simulations for each quadrant
    for i, quad in enumerate(['upper', 'lower']):
        cfile = join(opath, f'retina_{quad}_discs_params.json')
        if quad == 'upper':
            ArteryInPath, VeinInPath = ArteryInPath_upper, VeinInPath_upper
            ArteryOutPath, VeinOutPath = ArteryOutPath_upper, VeinOutPath_upper
        else:
            ArteryInPath, VeinInPath = ArteryInPath_lower, VeinInPath_lower
            ArteryOutPath, VeinOutPath = ArteryOutPath_lower, VeinOutPath_lower

        vascular_config(eye=eye, cfile=cfile,
                        ArteryInPath=ArteryInPath, VeinInPath=VeinInPath,
                        ArteryOutPath=ArteryOutPath, VeinOutPath=VeinOutPath,
                        macula_fraction=macula_fraction, FlowFactor=FlowFactor,
                        Actions=Actions, FrozenFactor=FrozenFactor, quad=quad)

        res = run_sim(cfile, quiet=quiet)
        if res != 0:
            return False, amfiles, cfile, None

        print(f'Converting CSV to AM: {ArteryOutPath}')
        ArteryOutPath_am = csv2amira(ArteryOutPath)
        print(f'Converting CSV to AM: {VeinOutPath}')
        VeinOutPath_am = csv2amira(VeinOutPath)
        amfiles.append(ArteryOutPath_am)
        amfiles.append(VeinOutPath_am)

    # Step 4: Combine and optionally export visualization
    if combine:
        ofile = 'retina_cco.am'
        graph = combine_graphs.combine_cco(opath, amfiles, ofile)

        if join_feeding:
            print("Joining feeding vessels...")
            graph = join_feeding_vessels(graph)
            graph.write(join(opath, ofile))

        if plot:
            try:
                print("[INFO] Exporting headless mesh with color coding...")

                vis = graph.plot_graph(show=False, block=False)

                if hasattr(vis, "cylinders_combined"):
                    mesh = vis.cylinders_combined
                    if isinstance(mesh, o3d.geometry.TriangleMesh):

                        # Color logic
                        vt = graph.point_scalars_to_edge_scalars(name='VesselType')
                        num_vertices = len(np.asarray(mesh.vertices))

                        # Default gray color
                        colors = np.tile([0.5, 0.5, 0.5], (num_vertices, 1))

                        # Color mapping (per edge), approximate to vertex coloring
                        if vt is not None and len(vt) > 0:
                            edge_colors = np.zeros((len(vt), 3))
                            edge_colors[vt == 1] = [1.0, 0.0, 0.0]  # Red for arteries
                            edge_colors[vt == 2] = [0.0, 0.0, 1.0]  # Blue for veins

                            # Simple approximate mapping from edge colors to vertices
                            # Just map first N edge colors to N vertices (crude but gives effect)
                            colors[:min(len(colors), len(edge_colors))] = edge_colors[:min(len(colors), len(edge_colors))]

                        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

                        out_path = join(opath, "vascular_output_colored.ply")
                        o3d.io.write_triangle_mesh(out_path, mesh)
                        print(f"[SUCCESS] Colored mesh saved to: {out_path}")
                    else:
                        print("[WARNING] cylinders_combined is not a TriangleMesh.")
                else:
                    print("[WARNING] Visualization object has no 'cylinders_combined' attribute.")
                vis.destroy_window()
            except Exception as e:
                print(f"[ERROR] Could not export mesh: {e}")

        return True, amfiles, ofile, graph

    return True, amfiles, None, None
#----------------For Colour
"""



def get_eye(geometry_file,create_new_geometry=False):

    """
    Helper to load eye geometry data stored in a pickle file
    """

    if create_new_geometry or geometry_file==None or geometry_file=='':
        ext = 35000./2.
        xoffset = 0
        domain = arr([[-ext+xoffset,ext+xoffset],[-ext,ext],[-ext,ext]]).astype('float')
        eye = Eye(domain=domain)
    else:
        eye = Eye()
        eye.load(geometry_file)
        
    return eye
    
def flow_ordering(graph,rfile=None,cco_path=None,run_reanimate=True,arterial_pressure=80.,venous_pressure=30.,join_inlets=False):

    """
    Using flow solutions from REANIMATE, order blood vessels
    """

    tmpDir = None
    if cco_path is None:
        # Create a temporary directory to store results
        while True: 
            tmpDir = TMPDIR
            tmpDir = join(tmpDir,f'temp_{str(int(np.random.uniform(0,10000000))).zfill(8)}')
            if not os.path.exists(tmpDir):
                os.mkdir(tmpDir)
                break
        cco_path = tmpDir

    if join_inlets:
        graph = join_feeding_vessels(graph)
    if run_reanimate:
        graph = reanimate_sim(graph,opath=cco_path,ofile=rfile,a_pressure=arterial_pressure,v_pressure=venous_pressure)
    elif rfile is not None:
        graph = spatialgraph.SpatialGraph()
        graph.read(rfile)

    ### Run crawl algorithm to get flow ordering ###
    # Crawl downstream from inlet artery
    ofilename = 'retina_arterial_crawl.am'
    graph = inject.crawl(graph,proj_path=cco_path,calculate_conc=False,plot=False,arteries_only=True,ofilename=ofilename)
    graph.write(join(cco_path,ofilename))    
    g_art_crawl = spatialgraph.SpatialGraph()
    g_art_crawl.read(join(cco_path,ofilename))
    # Crawl upstream from oulet vein
    ofilename = 'retina_venous_crawl.am'
    first_branch_index = g_art_crawl.get_data(name='Branch').max() + 1
    g_vein_crawl = inject.crawl(graph,proj_path=cco_path,calculate_conc=False,plot=False,downstream=False,veins_only=True,ofilename='retina_venous_crawl.am',first_branch_index=first_branch_index)  
    # Combine arterial (downstream) and venous (upstream) crawl results
    art_branch = g_art_crawl.get_data(name='Branch')
    art_order = g_art_crawl.get_data(name='Order')
    vtype = g_art_crawl.get_data(name='VesselType')
    art_branch[vtype==1] = g_vein_crawl.get_data(name='Branch')[vtype==1]
    art_order[vtype==1] = g_vein_crawl.get_data(name='Order')[vtype==1]
    g_art_crawl.set_data(art_branch, name='Branch')
    g_art_crawl.set_data(art_order, name='Order')
    graph = g_art_crawl
    
    if tmpDir is not None:
        shutil.rmtree(tmpDir)
    
    return graph                                                                              

def main(args):

    # Create directory structure
    lpath, cco_path, dataPath, surfacePath, embedPath, concPath = create_directories(args.path,args.name,overwrite_existing=args.overwrite_existing)

    # Load or create geometry
    geometry_file = join(dataPath,"retina_geometry.p")
    eye = get_eye(geometry_file,create_new_geometry=args.create_new_geometry)

    ### Create L-system seed ###
    if args.create_lsystem:
        ofile = 'retina_lsystem.am' #join(dataPath,'retina_lsystem.am')
        combined_graph,mfiles = generate_lsystem(opath=dataPath,lpath=lpath,gfile=ofile,eye=eye)

    ### Run Retina sims ###
    cco_ofile = 'retina_cco.am' # default value
    if args.run_vascular:
        res, amfiles, ofile, graph = vascular_upper_lower(lpath=lpath,convert_to_json=args.convert_to_json,opath=cco_path,join_feeding=True,eye=eye,quiet=args.quiet,macula_fraction=args.macula_fraction)
    else:
        graph = spatialgraph.SpatialGraph()
        graph.read(join(cco_path,cco_ofile))

    ### Create capillary bed or directly join arteries to veins ###
    if args.create_capillaries:
        # Directly connect venules to arterioles (quickest result)
        if args.direct_conn:
            graph = direct_arterial_venous_connection(cco_path,cco_ofile,None,graph=graph,write=False,displace_degen=True,geometry_file=geometry_file,eye=eye)
            cap_file = os.path.join(cco_path,cco_ofile.replace('.am','_a2v.am'))
            
        # Create capillary bed with voronoi tessellation
        else:
            graph = voronoi_capillary_bed(cco_path,cco_ofile,write=False,plot=False,displace_degen=True,geometry_file=geometry_file,eye=eye)
            cap_file = os.path.join(cco_path,cco_ofile.replace('.am','_vorcap.am'))
                    
        graph.write(cap_file)
    else:
        # Create default file with no capillary bed
        cap_file = os.path.join(cco_path,cco_ofile.replace('.am','_a2v.am'))
        if not os.path.exists(cap_file):
            cap_file = os.path.join(cco_path,cco_ofile.replace('.am','_vorcap.am'))

    # If arrived here with no graph loaded, then load the default one
    if graph is None:
        graph = spatialgraph.SpatialGraph()
        graph.read(cap_file)

    ### Run REANIMATE to get flow predictions ###
    cap_file_r = cap_file.replace('.am','_reanimate.am')
    graph = flow_ordering(graph,cco_path=cco_path,rfile=os.path.basename(cap_file_r),run_reanimate=args.run_reanimate,arterial_pressure=args.arterial_pressure,venous_pressure=args.venous_pressure)
        
    ### Sinusoidal fluctuations ###
    if args.add_wriggle:
        graph = sine_interpolate.apply_to_graph('',graph=graph,ofile=None,interp=True)
        cap_file_r_c = cap_file_r.replace('.am','_crawl.am')
        #graph.write(join(cco_path,cap_file_r_c))
        graph.write(cap_file_r_c)
        
        vis = graph.plot_graph(scalar_color_name='VesselType',show=False,block=False,win_width=args.win_width,win_height=args.win_height,bgcolor=[1.,1.,1.])
        vis.screen_grab(join(cco_path,'artery_vein_cco_sine.png'))
        vis.set_cylinder_colors(scalar_color_name='VesselType',cmap='gray',cmap_range=[0.,1.])
        vis.screen_grab(join(cco_path,'artery_vein_cco_sine_gray.png'))
        vis.destroy_window()
        
        # Re-run REANIMATE (only if not adding voronoi capillaries...)
        # Write .dat file to fixed location
        if args.direct_conn:
            graph = reanimate_sim(graph,opath=cco_path,ofile=os.path.basename(cap_file_r_c),a_pressure=args.arterial_pressure,v_pressure=args.venous_pressure)
    else:
        cap_file_r_c = cap_file_r
        
    if args.create_capillaries and not args.direct_conn:       
        # Remove existing capillaries
        vt = graph.get_data('VesselType')
        epi = graph.edgepoint_edge_indices()
        rind = np.where(vt==2)
        if len(rind[0])>0:
            redge = np.unique(epi[rind])
            gvars = GVars(graph)
            gvars.remove_edges(redge)
            graph = gvars.set_in_graph()
        graph = voronoi_capillary_bed(cco_path,cco_ofile,graph=graph,write=False,plot=False,displace_degen=True,geometry_file=geometry_file,eye=eye)
        #cap_file = os.path.join(cco_path,cap_file_r_c.replace('.am','_vorcap.am'))
        cap_file = cap_file_r_c.replace('.am','_vorcap.am')
                    
        graph.write(cap_file)
    
    ### Screen grabs ###
    domain = None
    vis = graph.plot_graph(scalar_color_name='Flow',cmap='jet',log_color=True,show=False,block=False,win_width=args.win_width,win_height=args.win_height,domain=domain,bgcolor=[1.,1.,1.])
    vis.screen_grab(join(cco_path,'log_flow.png'))
    vis.set_cylinder_colors(scalar_color_name='Pressure',cmap='jet',log_color=False,cmap_range=[args.venous_pressure,args.arterial_pressure])
    vis.screen_grab(join(cco_path,'pressure.png'))
    vis.destroy_window()
    
    # Create enface image
    if args.create_enface_image:
        xdim = 500*3
        ydim = 500*3
        view_size = arr([6000.,6000.])*3
        create_enface(opath=cco_path,graph=graph,xdim=xdim,ydim=ydim,view_size=view_size,eye=eye,plot=False,figsize=[7,7],exclude_capillaries=True)
            
    ### Create retina layer surfaces ###
    if args.create_surfaces:
        ofile = join(surfacePath,'retina_surface.ply')
        plot_file = ofile.replace('.ply','_profile.png')
        path = join(cco_path,'graph')
        create_surface(path=path,ofile=ofile,plot=True,plot_file=plot_file,add_simplex_noise=True,eye=eye,project=args.project_surface_to_sphere)
    
    # Project vessels onto surface
    if args.project:
        vfile = join(surfacePath,'retina_surface_vessels.ply')
        ofile = cap_file_r_c.replace('.am','_projected.am')
        project_to_surface(graph=graph,eye=eye,vfile=vfile,plot=False,interpolate=False,ofile=ofile,write_mesh=True,iterp_resolution=10.,filter=False)
    
    ### Embed into volume ###
    if args.embed_volume:
        # Embedding domain size (usually relative to optic disc centre at origin)
        depth = 15. # mm
        domain = arr([[-1000.,11000.],[-6000.,6000.],[-depth*1000.,depth*1000.]])
        # Rotation of domain
        theta,phi,chi = 0., 0., 0.
        # Volume pixel dimensions
        dim = [500*2,500*2,1500]

        embedObj = embed(graph=graph,eye=eye,filename=None,domain=domain,dim=dim,surface_dir=surfacePath,output_path=embedPath,theta=theta,phi=phi,chi=chi,write=False)              
        
        # Create OCT-A-like enface image from embedded vessel data
        vessels = embedObj.vessel_grid
        enface = np.sum(vessels,axis=2)

    ### Run crawl algorithm and simulate injection ###
    # Only possible for fully-connected networks
    if args.simulate_injection and args.direct_conn:
        graph,conc,t = inject.crawl(graph,proj_path=cco_path,calculate_conc=True,image_dir=concPath)
        recon_times = None
        #-------------Updated Version--------------------#
        #inject.conc_movie(graph,data=conc,time=t,recon_times=recon_times,odir=concPath,win_width=args.win_width,win_height=args.win_width,eye=eye,movie_naming=False)
        inject.conc_movie(graph=graph, data=conc, time=t, recon_times=recon_times, odir=concPath, 
                  win_width=args.win_width, win_height=args.win_width, 
                  eye=None, movie_naming=False, output_mesh=False)

def batch(args):

    if not os.path.exists(args.path):
        os.mkdir(args.path)

    for i in range(args.batch_offset,args.nbatch):
        cname = join(args.path,f'sim{str(i).zfill(8)}')
        args.name = cname
        main(args)
        
def create_parser():

    parser = argparse.ArgumentParser(prog='RetinaSim',description='For simulating retina vascular networks')
    parser.add_argument('path', type=str, help='Output path')
    parser.add_argument('--name', type=str, default='', help='Simulation name')
    parser.add_argument('--quiet', type=bool, default=True, help='Inhibit display of debugging output (default=True)')
    parser.add_argument('--batch', type=bool, default=False, help='Batch mode (boolean), default=False')
    parser.add_argument('--nbatch', type=int, default=10, help='Number of batches to generate')
    parser.add_argument('--batch_offset', type=int, default=0, help='Batch index to begin at')
    parser.add_argument('--overwrite_existing', type=bool, default=False, help='Overwrite existing data (default=False)')
    parser.add_argument('--create_lsystem', type=bool, default=True, help='Create new L-sysem data (otherwise load existing) (default=True)')
    parser.add_argument('--run_vascular', type=bool, default=True, help='Create new vascular simulation, using Retina library (otherwise load existing) (default=True)')
    parser.add_argument('--convert_to_json', type=bool, default=True, help='Convert Retina input to JSON (default=True)')
    parser.add_argument('--run_reanimate', type=bool, default=True, help='Run REANIMATE for flow estimation (default=True)')
    parser.add_argument('--diabetic', type=bool, default=False, help='Run diabetes capillary dropout simulation (default=False)')
    parser.add_argument('--nart', type=int, default=False, help='Number of arterioles to occlude in diabetes simulation (default=False)')
    parser.add_argument('--create_capillaries', type=bool, default=True, help='Add capillary structures (default=True)')
    parser.add_argument('--direct_conn', type=bool, default=True, help='If creating capillary structures, directly connect arterioles to venuoles (alternatively, create voronoi capillary bed) (default=True)')
    parser.add_argument('--simulate_injection', type=bool, default=False, help='Simulate fluorescein injection (default=False)')
    parser.add_argument('--create_new_geometry', type=bool, default=True, help='Generate new eye geoemtry values (otherwise load existing) (default=True)')
    parser.add_argument('--create_surfaces', type=bool, default=True, help='Create new retina surface layer geometry (otherwise use existing) (default=True)')
    parser.add_argument('--project_surface_to_sphere', type=bool, default=True, help='Project surfaces to spherical geometry (default=True)')
    parser.add_argument('--project', type=bool, default=True, help='Project onto spherical geometry (default=True)')
    parser.add_argument('--embed_volume', type=bool, default=False, help='Embed simulation into 3D pixel volume (default=False)')
    parser.add_argument('--create_enface_image', type=bool, default=True, help='Create an OCT-A-style enface image (default=True)')
    parser.add_argument('--add_wriggle', type=bool, default=True, help='Overlay sinusoidal tortuosity to vessel paths (default=True)')
    parser.add_argument('--win_width', type=int, default=6000, help='Window width for screen grabs to file (default=6000 pixels)')
    parser.add_argument('--win_height', type=int, default=6000, help='Window height for screen grabs to file (default=6000 pixels)')
    parser.add_argument('--arterial_pressure', type=float, default=52., help='Arterial pressure for flow simulations (default=52mmHg)')
    parser.add_argument('--venous_pressure', type=float, default=20., help='Venous pressure for flow simulations (default=20mmHg)')
    parser.add_argument('--macula_fraction', type=float, default=0.2, help='Fraction of macula to regrow vasculature into (default=0.2)')
    
    return parser

if __name__=='__main__':

    parser = create_parser()
    args = parser.parse_args()

    # Single run
    if args.batch==False:
        main(args)
    # Batch run
    else:
        batch(args)
