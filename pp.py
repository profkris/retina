import open3d as o3d
import numpy as np
import pyvista as pv
import os

# Step 1: Create a simple colored cylinder and save it as a .ply file
def create_colored_ply(filename="test_cylinder.ply"):
    # Create cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=2.0)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color([0.2, 0.7, 0.3])  # greenish color
    o3d.io.write_triangle_mesh(filename, cylinder)
    print(f"✅ PLY file written: {filename}")

# Step 2: Load the .ply file and render to PNG using PyVista
def render_ply_to_png(ply_file="test_cylinder.ply", png_file="test_cylinder.png"):
    mesh = pv.read(ply_file)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh)
    plotter.set_background("white")
    plotter.show(screenshot=png_file)
    print(f"✅ PNG snapshot saved: {png_file}")

# Run the test
if __name__ == "__main__":
    create_colored_ply()
    render_ply_to_png()

