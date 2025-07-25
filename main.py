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
from mpi4py import MPI
import sys
from datetime import datetime

# Updated the function to run in headless mode
def generate_lsystem(opath=None, lpath=None, gfile=None, screen_grab=True, eye=None):
    
    """Simulates retinal artery/vein seed networks and generates a merged colored PLY file.
    Arteries = red, Veins = blue. Includes torus annotations if eye is provided."""
    

    # Step 1: Simulate network
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

    combined_graph = combine_graphs.combine_cco(
        opath,
        [join(lpath, m) for m in mfiles],
        gfile
    )

    if screen_grab:
        try:
            print("[INFO] Generating merged colored mesh...")
            vtypeEdge = combined_graph.point_scalars_to_edge_scalars(name='VesselType')
            full_mesh = o3d.geometry.TriangleMesh()

            for vtype, color in zip([0, 1], [[1, 0, 0], [0, 0, 1]]):  # red, blue
                vis = combined_graph.plot_graph(
                    show=False,
                    block=False,
                    edge_filter=(vtypeEdge == vtype),
                    min_radius=5.0,
                    cyl_res=10,
                    radius_scale=1.0
                )
                mesh = vis.cylinders_combined
                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    n_verts = np.asarray(mesh.vertices).shape[0]
                    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (n_verts, 1)))
                    full_mesh += mesh
                vis.destroy_window()

            # Add torus annotations (optic disc, fovea, macula, etc.)
            if eye is not None:
                def create_colored_torus(center, torus_radius, tube_radius, color):
                    torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=torus_radius, tube_radius=tube_radius)
                    torus.paint_uniform_color(color)
                    torus.translate(center)
                    return torus

                full_mesh += create_colored_torus(eye.optic_disc_centre, eye.optic_disc_radius, 20.0, [0, 0, 0])
                full_mesh += create_colored_torus(eye.fovea_centre, eye.fovea_radius, 20.0, [0, 0, 0])
                full_mesh += create_colored_torus(eye.macula_centre, eye.macula_radius, 20.0, [0, 0, 0])

                centre = eye.occular_centre.copy()
                centre[2] = 0.0
                domain_radius = (eye.domain[0, 1] - eye.domain[0, 0]) / 2.0
                full_mesh += create_colored_torus(centre, domain_radius, 20.0, [0, 0, 0])

            out_path = join(opath, "lsystem.ply")
            o3d.io.write_triangle_mesh(out_path, full_mesh)
            print(f"[SUCCESS] Output mesh saved to: {out_path}")

        except Exception as e:
            print(f"[ERROR] Failed to generate output: {e}")

    return combined_graph, mfiles

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

# Updated the function to run in headless mode
def vascular_upper_lower(opath=None, lpath=None, input_graphs=None, convert_to_json=True, join_feeding=True, eye=None,
                         ArteryInPath_lower=None, VeinInPath_lower=None, ArteryOutPath_lower=None, VeinOutPath_lower=None,
                         ArteryInPath_upper=None, VeinInPath_upper=None, ArteryOutPath_upper=None, VeinOutPath_upper=None,
                         macula_fraction=1., FlowFactor=1., Actions="coarse;macula;resolve", FrozenFactor=0,
                         quiet=False, combine=True, plot=True):

    # Use Retina library to generate vessel networks.
    # Display-independent version: exports .ply instead of requiring DISPLAY variable.

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
                print("[INFO] Exporting artery/vein mesh...")
                vtypeEdge = graph.point_scalars_to_edge_scalars(name='VesselType')
                full_mesh = o3d.geometry.TriangleMesh()

                for vtype, color in zip([0, 1], [[1, 0, 0], [0, 0, 1]]):
                    vis = graph.plot_graph(
                        show=False,
                        block=False,
                        edge_filter=(vtypeEdge == vtype),
                        min_radius=5.0,
                        cyl_res=10,
                        radius_scale=1.0
                    )
                    mesh = vis.cylinders_combined
                    if isinstance(mesh, o3d.geometry.TriangleMesh):
                        n_verts = np.asarray(mesh.vertices).shape[0]
                        mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (n_verts, 1)))
                        full_mesh += mesh
                    vis.destroy_window()

                if eye is not None:
                    def create_colored_torus(center, torus_radius, tube_radius, color):
                        torus = o3d.geometry.TriangleMesh.create_torus(
                            torus_radius=torus_radius, tube_radius=tube_radius)
                        torus.paint_uniform_color(color)
                        torus.translate(center)
                        return torus

                    full_mesh += create_colored_torus(eye.optic_disc_centre, eye.optic_disc_radius, 20.0, [0, 0, 0])
                    full_mesh += create_colored_torus(eye.fovea_centre, eye.fovea_radius, 20.0, [0, 0, 0])
                    full_mesh += create_colored_torus(eye.macula_centre, eye.macula_radius, 20.0, [0, 0, 0])
                    centre = eye.occular_centre.copy()
                    centre[2] = 0.0
                    domain_radius = (eye.domain[0, 1] - eye.domain[0, 0]) / 2.0
                    full_mesh += create_colored_torus(centre, domain_radius, 20.0, [0, 0, 0])

                out_path = join(opath, "artery_vein_overlay_cco.ply")
                o3d.io.write_triangle_mesh(out_path, full_mesh)
                print(f"[SUCCESS] Colored mesh saved to: {out_path}")

            except Exception as e:
                print(f"[ERROR] Could not export colored mesh: {e}")

        return True, amfiles, ofile, graph

    return True, amfiles, None, None

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
    print("[INFO] Creating Directory Structure...")
    lpath, cco_path, dataPath, surfacePath, embedPath, concPath = create_directories(args.path,args.name,overwrite_existing=args.overwrite_existing)

    # Load or create geometry
    geometry_file = join(dataPath,"retina_geometry.p")
    eye = get_eye(geometry_file,create_new_geometry=args.create_new_geometry)

    ### Create L-system seed ###
    print("[INFO] Creating Lsystem...")
    if args.create_lsystem:
        ofile = 'retina_lsystem.am' #join(dataPath,'retina_lsystem.am')
        combined_graph,mfiles = generate_lsystem(opath=dataPath,lpath=lpath,gfile=ofile,eye=eye)

    ### Run Retina sims ###
    print("[INFO] Running Retina Simulations...")
    cco_ofile = 'retina_cco.am' # default value
    if args.run_vascular:
        res, amfiles, ofile, graph = vascular_upper_lower(lpath=lpath,convert_to_json=args.convert_to_json,opath=cco_path,join_feeding=True,eye=eye,quiet=args.quiet,macula_fraction=args.macula_fraction)
    else:
        graph = spatialgraph.SpatialGraph()
        graph.read(join(cco_path,cco_ofile))

    ### Create capillary bed or directly join arteries to veins ###
    print("[INFO] Creating capillary bed...")
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

    # Updated the if block to run in headless mode
    print("[INFO] Sinusoidal fluctuations...")
    if args.add_wriggle:
   
        graph = sine_interpolate.apply_to_graph('', graph=graph, ofile=None, interp=True)
        cap_file_r_c = cap_file_r.replace('.am', '_crawl.am')
        graph.write(cap_file_r_c)

        # Get graph data
        edgeconn = graph.get_field('EdgeConnectivity')['data']
        edgepoints = graph.get_field('EdgePointCoordinates')['data']
        nedgepoints = graph.get_field('NumEdgePoints')['data'].flatten()
        category = graph.get_field('VesselType')['data'] if 'VesselType' in graph.fieldNames else np.zeros(edgeconn.shape[0], dtype=int)

        print(f"Total edges: {len(edgeconn)}")
        print(f"Total edgepoints: {len(edgepoints)}")
        print(f"Unique vessel types: {np.unique(category)}")

        # Build full edge point list
        vessel_meshes = []
        idx = 0
        for i, npts in enumerate(nedgepoints):
            pts = edgepoints[idx:idx + npts]
            idx += npts

            if len(pts) < 2:
                continue  # Skip short segments

            vessel_type = category[i] if i < len(category) else 2
            if vessel_type == 0:
                color = [0, 0, 1]  # Vein = blue
            elif vessel_type == 1:
                color = [1, 0, 0]  # Artery = red
            else:
                color = [0.5, 0.5, 0.5]

            # Create small cylinders between each pair of points
            for j in range(len(pts) - 1):
                start, end = pts[j], pts[j + 1]
                direction = end - start
                height = np.linalg.norm(direction)
                if height < 1e-3:
                    continue

                cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=height, resolution=10)
                cyl.paint_uniform_color(color)
                cyl.compute_vertex_normals()

                # Rotate to align with direction
                z_axis = np.array([0, 0, 1])
                direction_norm = direction / height
                axis = np.cross(z_axis, direction_norm)
                angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0))
                if np.linalg.norm(axis) > 1e-6:
                    axis = axis / np.linalg.norm(axis)
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    cyl.rotate(R, center=np.array([0, 0, 0]))

                cyl.translate(start)
                vessel_meshes.append(cyl)

        # Merge all
        full_mesh = o3d.geometry.TriangleMesh()
        for cyl in vessel_meshes:
            full_mesh += cyl
        full_mesh.compute_vertex_normals()

        # Save mesh
        output_ply = os.path.join(cco_path, 'artery_vein_cco_sine.ply')
        o3d.io.write_triangle_mesh(output_ply, full_mesh, write_ascii=False)
        print(f"Vascular structure saved to: {output_ply}")
        print(f"Total vessels drawn: {len(vessel_meshes)}")

        # Optional: re-run REANIMATE
        if args.direct_conn:
            graph = reanimate_sim(
                graph,
                opath=cco_path,
                ofile=os.path.basename(cap_file_r_c),
                a_pressure=args.arterial_pressure,
                v_pressure=args.venous_pressure
            )
    else:
        cap_file_r_c = cap_file_r

    # Updated block to run in headless mode
    print("[INFO] Entered capillary post-processing block")
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

    vis = graph.plot_graph(
        scalar_color_name='Flow',
        cmap='jet',
        log_color=True,
        show=False,
        block=False,
        win_width=args.win_width,
        win_height=args.win_height,
        domain=None,
        bgcolor=[1., 1., 1.]
    )


    cco_path = os.path.join(args.name, 'cco')  # Output path

    # 2. Export FLOW
    flow_data = graph.get_data('Flow')
    if flow_data is not None and np.any(np.isfinite(flow_data)):
        # Determine color range
        min_flow, max_flow = np.nanmin(flow_data), np.nanmax(flow_data)
        if min_flow == max_flow:
            max_flow += 1.0

        # Set color and export .ply
        vis.set_cylinder_colors(
            scalar_color_name='Flow',
            cmap='jet',
            log_color=True,
            cmap_range=[min_flow, max_flow],
            update=True
        )
        vis.export_colored_ply(os.path.join(cco_path, 'log_flow.ply'))
        print("[INFO] Created log_flow.ply")
    else:
        print("[WARNING] Flow data is missing or invalid.")

    # 3. Export PRESSURE
    pressure_data = graph.get_data('Pressure')
    if pressure_data is not None and np.any(np.isfinite(pressure_data)):
        min_p = args.venous_pressure
        max_p = args.arterial_pressure

        vis.set_cylinder_colors(
            scalar_color_name='Pressure',
            cmap='jet',
            log_color=False,
            cmap_range=[min_p, max_p],
            update=True
        )
        vis.export_colored_ply(os.path.join(cco_path, 'pressure.ply'))
        print("[INFO] Created pressure.ply")
    else:
        print("[WARNING] Pressure data is missing or invalid.")

    # 4. Clean up
    vis.destroy_window()
    print("[INFO] log flow and pressure file generation complete.")


# Updated the section to run in headless mode
    # Create enface image
    print("[INFO] Creating Enface Image...")
    if args.create_enface_image:
        xdim = 500*3
        ydim = 500*3
        view_size = arr([6000.,6000.])*3
        create_enface(opath=cco_path,graph=graph,xdim=xdim,ydim=ydim,view_size=view_size,eye=eye,plot=False,figsize=[7,7],exclude_capillaries=True)
            
    ### Create retina layer surfaces ###
    print("[INFO] Creating Retina Layer Surfaces...")
    if args.create_surfaces:
        ofile = join(surfacePath,'retina_surface.ply')
        plot_file = ofile.replace('.ply','_profile.png')
        path = join(cco_path,'graph')
        create_surface(path=path,ofile=ofile,plot=True,plot_file=plot_file,add_simplex_noise=True,eye=eye,project=args.project_surface_to_sphere)
    
    # Project vessels onto surface
    print("[INFO] Projecting vessel onto surface...")
    if args.project:
        vfile = join(surfacePath,'retina_surface_vessels.ply')
        ofile = cap_file_r_c.replace('.am','_projected.am')
        project_to_surface(graph=graph,eye=eye,vfile=vfile,plot=False,interpolate=False,ofile=ofile,write_mesh=True,iterp_resolution=10.,filter=False)
    
    ### Embed into volume ###
    print("[INFO] Embed into Volume...")
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
    print("[INFO] Simulate injection...")
    if args.simulate_injection and args.direct_conn:
        graph,conc,t = inject.crawl(graph,proj_path=cco_path,calculate_conc=True,image_dir=concPath)
        recon_times = None
        # Updated the function call
        #inject.conc_movie(graph,data=conc,time=t,recon_times=recon_times,odir=concPath,win_width=args.win_width,win_height=args.win_width,eye=eye,movie_naming=False)
        inject.conc_movie(graph=graph, data=conc, time=t, recon_times=recon_times, odir=concPath,win_width=args.win_width, win_height=args.win_width, eye=None, movie_naming=False, output_mesh=False)

# MPI Implementation in the function
def batch(args):
    # MPI Setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Per-rank Logging
    log_path = f"/scratch/vamshis/LOG/log_rank_{rank}.txt" # Add here path as per your system path
    log_file = open(log_path, "w")
    sys.stdout = sys.stderr = log_file

    print(f"[Rank {rank}] Starting simulation")

    # Step 1: Create output directory (only by rank 0)
    if rank == 0 and not os.path.exists(args.path):
        os.makedirs(args.path)
        print(f"[Rank {rank}] Created directory: {args.path}")
    comm.Barrier()

    # Step 2: Run assigned simulations
    for i in range(args.batch_offset, args.nbatch):
        print(f"[Rank {rank}] Checking if simulation {i} belongs to this rank")
        if i % size != rank:
            print(f"[Rank {rank}] Skipping simulation {i}")
            continue

        cname = os.path.join(args.path, f'sim{str(i).zfill(8)}')
        args.name = cname
        print(f"[Rank {rank}] Starting simulation {i} at {datetime.now()} as {cname}")

        try:
            main(args)
            print(f"[Rank {rank}] Finished simulation {i} at {datetime.now()}")
        except Exception as e:
            print(f"[Rank {rank}] ERROR during simulation {i}: {e}")

    # Step 3: Final synchronization
    print(f"[Rank {rank}] BEFORE final barrier")
    comm.Barrier()
    print(f"[Rank {rank}] AFTER final barrier")

    # Step 4: Finish
    if rank == 0:
        print(f"[SUCCESS] [Rank 0] All simulations finished. Job will exit at {datetime.now()}")

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
