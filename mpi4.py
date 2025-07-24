from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"Hello from rank {comm.Get_rank()} of {comm.Get_size()}")

"""

   ### Screen grabs ###
    ### Screen grabs and colored PLY exports ###
    domain = None

    # === FLOW ===
    flow = graph.get_data('Flow')
    if flow is None or not np.any(np.isfinite(flow)):
        print("[WARNING] Flow data missing or invalid — using fallback.")
        flow = np.ones(graph.nedge)

    min_flow, max_flow = np.nanmin(flow), np.nanmax(flow)
    if min_flow == max_flow or not np.isfinite(min_flow) or not np.isfinite(max_flow):
        print("[WARNING] Invalid Flow values — using default range [0, 1]")
        min_flow, max_flow = 0.0, 1.0

    # Register Flow scalar if not already
    register_scalar_to_graph(graph, 'Flow')

    # Plot with Flow
    vis = graph.plot_graph(
        scalar_color_name='Flow',
        cmap='jet',
        log_color=True,
        show=False,
        block=False,
        win_width=args.win_width,
        win_height=args.win_height,
        domain=domain,
        bgcolor=[1., 1., 1.]
    )

    # Set colors
    vis.set_cylinder_colors(
        scalar_color_name='Flow',
        cmap='jet',
        log_color=True,
        cmap_range=[min_flow, max_flow]
    )

    # Export PLY for Flow
    vis.export_colored_ply(os.path.join(cco_path, 'log_flow.ply'))

    # === PRESSURE ===
    pressure = graph.get_data('Pressure')
    if pressure is None or not np.any(np.isfinite(pressure)):
        print("[WARNING] Pressure data missing or invalid — using fallback.")
        pressure = np.ones(graph.nedge)

    min_p, max_p = np.nanmin(pressure), np.nanmax(pressure)
    if min_p == max_p or not np.isfinite(min_p) or not np.isfinite(max_p):
        print("[WARNING] Invalid Pressure values — using default range [0, 1]")
        min_p, max_p = 0.0, 1.0

    # Register Pressure scalar
    register_scalar_to_graph(graph, 'Pressure')

    # Set pressure color and export PLY
    vis.set_cylinder_colors(
        scalar_color_name='Pressure',
        cmap='jet',
        log_color=False,
        cmap_range=[min_p, max_p]
    )
    vis.export_colored_ply(os.path.join(cco_path, 'pressure.ply'))

    vis.destroy_window() """



"""
    # --- Setup display-independent graph rendering ---
    domain = None

    # Step 1: Create visualization object (headless mode, no show)
    vis = graph.plot_graph(
        show=False,
        block=False,
        win_width=args.win_width,
        win_height=args.win_height,
        domain=domain,
        bgcolor=[1.0, 1.0, 1.0]
    )

    # ===  Export log_flow.ply (in RED) ===
    print("[INFO] Exporting log_flow.ply in red color...")
    red_color = np.tile(np.array([[1.0, 0.0, 0.0]]), (graph.nedgepoint, 1))
    vis.set_cylinder_colors(edge_color=red_color)
    vis.export_colored_ply(os.path.join(cco_path, 'log_flow.ply'))

    # === Export pressure.ply (in BLUE) ===
    print("[INFO] Exporting pressure.ply in blue color...")
    blue_color = np.tile(np.array([[0.0, 0.0, 1.0]]), (graph.nedgepoint, 1))
    vis.set_cylinder_colors(edge_color=blue_color)
    vis.export_colored_ply(os.path.join(cco_path, 'pressure.ply'))

    # Clean up rendering session
    vis.destroy_window()"""



"""
    # --- Setup display-independent graph rendering ---
    domain = None

    # Step 1: Create visualization object (headless mode, no show)
    vis = graph.plot_graph(
        show=False,
        block=False,
        win_width=args.win_width,
        win_height=args.win_height,
        domain=domain,
        bgcolor=[1.0, 1.0, 1.0]
    )

    # === Export log_flow.ply (RED vessels) ===
    print("[INFO] Exporting log_flow.ply in red color...")
    red_color = np.ones((graph.nedgepoint, 3))
    red_color[:, 0] = 1.0  # Red
    red_color[:, 1] = 0.0
    red_color[:, 2] = 0.0
    vis.color = red_color               # <--- This ensures the class uses it
    vis.set_cylinder_colors(edge_color=red_color, update=True)
    vis.export_colored_ply(os.path.join(cco_path, 'log_flow.ply'))

    # === Export pressure.ply (BLUE vessels) ===
    print("[INFO] Exporting pressure.ply in blue color...")
    blue_color = np.ones((graph.nedgepoint, 3))
    blue_color[:, 0] = 0.0
    blue_color[:, 1] = 0.0
    blue_color[:, 2] = 1.0  # Blue
    vis.color = blue_color              # <--- Also ensure used before setting
    vis.set_cylinder_colors(edge_color=blue_color, update=True)
    vis.export_colored_ply(os.path.join(cco_path, 'pressure.ply'))

    # Clean up rendering session
    vis.destroy_window()
"""
