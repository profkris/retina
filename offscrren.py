import open3d as o3d
import numpy as np
from PIL import Image

# Create a red sphere
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([1, 0, 0])  # Red

# Create an OffscreenRenderer
width, height = 640, 480
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

# Set up scene
scene = renderer.scene
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"

scene.add_geometry("sphere", mesh, mat)

# Set up camera
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
scene.camera.look_at(center, center + [0, 0, 5], [0, 1, 0])

# Render to image
img = renderer.render_to_image()

# Save image using PIL
img_np = np.asarray(img)
Image.fromarray(img_np).save("offscreen_test.png")
print("✅ Rendered offscreen image saved as 'offscreen_test.png'")

