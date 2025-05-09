import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
vis.add_geometry(mesh)
vis.poll_events()
vis.update_renderer()
img = vis.capture_screen_float_buffer(False)
vis.destroy_window()

import numpy as np
import cv2
img_np = (np.asarray(img) * 255).astype(np.uint8)
cv2.imwrite("test.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
