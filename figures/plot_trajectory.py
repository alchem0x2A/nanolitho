# This function requires blender's python environment to work
# 3 

import bpy
import math
import numpy as np

def create_hemisphere_framework(R, num_longitudinal, num_latitudinal, minor_radius=0.005, major_seg=128, minor_seg=12):
    # Ensure we're in object mode
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    toruses = []
    
    # Helper function to create a torus
    def create_torus(major_radius, minor_radius, location, rotation):
        bpy.ops.mesh.primitive_torus_add(
            major_radius=major_radius - minor_radius,
            minor_radius=minor_radius, 
            major_segments=major_seg,
            minor_segments=minor_seg,
            location=location,
            rotation=rotation
        )
        torus = bpy.context.active_object
        toruses.append(torus)
        return torus

    # Add the sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=R, location=(0, 0, 0), segments=128,  ring_count=128)
#    bpy.ops.object.metaball_add(type="BALL", radius=2 R)
    sphere = bpy.context.active_object
    
    # Create longitudinal circles (vertical ones around the sphere)
    for i in range(num_longitudinal):
        angle = 2 * math.pi * i / num_longitudinal
        rotation = (math.pi / 2, 0, angle)
        toruses.append(create_torus(R, minor_radius, (0, 0, 0), rotation))

    # Create latitudinal circles (horizontal ones slicing the sphere)
    for i in range(1, num_latitudinal):  # We skip creating a circle at the top-most and bottom-most points
        angle = math.pi * i / num_latitudinal - math.pi / 2  # Subtracting pi/2 to start from the bottom
        z = R * math.sin(angle)
        r = R * math.cos(angle)
        toruses.append(create_torus(r, minor_radius, (0, 0, z), (0, 0, 0)))

    # Truncate/halve objects above hemisphere
#    bpy.ops.object.select_all(action='DESELECT')
#    bpy.ops.object.select_pattern(pattern="Torus*")
#    sphere.select_set(True)
#    bpy.context.view_layer.objects.active = sphere
#    bpy.ops.object.mode_set(mode='EDIT')
#    bpy.ops.mesh.select_all(action='DESELECT')
#    bpy.ops.mesh.bisect(plane_co=(0, 0, 0), plane_no=(0, 0, 1), clear_outer=True)
#    bpy.ops.object.mode_set(mode='OBJECT')

    # Join all toruses into a single object (if needed)
    for torus in toruses:
        torus.select_set(True)
    bpy.context.view_layer.objects.active = toruses[0]
    bpy.ops.object.join()
    grid = bpy.context.active_object
    return sphere, grid

def polar_to_cartesian(R, psi, theta):
    """Convert polar coordinates to cartesian coordinates."""
    x = R * np.sin(psi) * np.cos(theta)
    y = R * np.sin(psi) * np.sin(theta)
    z = R * np.cos(psi)
    return (x, y, z)

def plot_trajectory(R, trajectory, radius=0.005):
    """Plot points on the surface of a sphere given a list of polar coordinates."""
    psi, theta = trajectory
    x, y, z = polar_to_cartesian(R, psi, theta)
    spheres = []
    for x_, y_, z_, psi_, theta_ in zip(x, y, z, psi, theta):
        bpy.ops.mesh.primitive_circle_add(radius=radius, location=(x_, y_, z_), rotation=(0, psi_, theta_), vertices=6, calc_uvs=False, fill_type="NGON")
        sp = bpy.context.active_object
        spheres.append(sp)
    for sp in spheres:
        sp.select_set(True)
    bpy.context.view_layer.objects.active = spheres[0]
    bpy.ops.object.join()
    traj_obj = bpy.context.active_object
    bpy.ops.object.modifier_add(type='SUBSURF')
    return traj_obj

def main(trajectories, colors, n_long, n_lat)
# Clear any existing mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Create the spherical circles
create_hemisphere_framework(1, 18, 18, 0.002)

# Create a 270 circle at 10 degrees
psis = np.deg2rad(np.ones(270) * 5)
thetas = np.deg2rad(np.linspace(0, 270, 270))
plot_trajectory(1, (psis, thetas), radius=0.01)


# A single spot
psis = np.deg2rad([5])
thetas = np.deg2rad([315])
plot_trajectory(1, (psis, thetas), radius=0.01)

# Save the Blender file
# bpy.ops.wm.save_as_mainfile(filepath="path/to/your/directory/generated_sphere.blend")
