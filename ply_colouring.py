import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

# === USER: Define your input/output file paths here ===
ply_path = "/scratch/vamshis/OUTPUT/COLOUR/sim21/cco/log_flow.ply"                 # Example: "log_flow.ply"
scalar_path = "/scratch/vamshis/OUTPUT/COLOUR/sim21/cco/flow.npy"      # Example: "flow.npy"
output_path = "/scratch/vamshis/OUTPUT/COLOUR/sim21/cco/output_colored.ply"     # Example: "log_flow_colored.ply"

cmap = 'jet'          # Colormap name: 'jet', 'viridis', etc.
log_color = True      # True if you want log scale
cmap_min = None       # Optional: set to float like 0.1
cmap_max = None       # Optional: set to float like 10.0

# =======================================================

def color_ply_by_scalar(ply_path, scalar_path, output_path, cmap='jet', log_color=False, cmap_min=None, cmap_max=None):
    print(f"[INFO] Loading PLY: {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    print(f"[INFO] Loading scalar values: {scalar_path}")
    scalars = np.load(scalar_path)

    if log_color:
        scalars = np.abs(scalars)
        scalars[scalars <= 0] = 1e-12
        scalars = np.log(scalars)

    if len(scalars) != len(mesh.vertices):
        raise ValueError(f"Mismatch: {len(scalars)} scalars vs {len(mesh.vertices)} vertices in mesh.")

    vmin = cmap_min if cmap_min is not None else np.nanmin(scalars)
    vmax = cmap_max if cmap_max is not None else np.nanmax(scalars)
    if vmin == vmax:
        print("[WARNING] cmap min == max. Adjusting...")
        vmax += 1.0

    norm_scalars = np.clip((scalars - vmin) / (vmax - vmin), 0.0, 1.0)

    cmap_func = plt.cm.get_cmap(cmap)
    colors = cmap_func(norm_scalars)[:, :3]  # Drop alpha

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    print(f"[INFO] Saving colored PLY: {output_path}")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print("[DONE] Colored PLY generated successfully.")


# === Call the function ===
color_ply_by_scalar(
    ply_path=ply_path,
    scalar_path=scalar_path,
    output_path=output_path,
    cmap=cmap,
    log_color=log_color,
    cmap_min=cmap_min,
    cmap_max=cmap_max
)

