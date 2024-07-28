import time
import trimesh
import trimesh.sample
import numpy as np
from copy import deepcopy
from components.arrangement import Arrangement

def optimize(arrangement: Arrangement, 
             resolve_collisions: bool = True,
             collision_move_step: float = 0.005, 
             collision_max_iters: int = 1000,
             make_tight: bool = True,
             make_tight_iters: int = 10, 
             approximate_gravity: bool = True) -> Arrangement:
    '''
    Optimize the spatial positions and orientations of the objects in the arrangement 
    such that they are physically possible.

    Args:
        arrangement: Arrangement, the arrangement to optimize
        resolve_collisions: bool, whether to resolve collisions between objects
        collision_move_step: float, the distance per step to move the objects when resolving collisions
        collision_max_iters: int, the maximum number of iterations to resolve collisions
        make_tight: bool, whether to make the objects fit tightly together
        make_tight_iters: int, the number of iterations to make the objects fit tightly together
        approximate_gravity: bool, whether to approximate gravity

    Returns:
        optimized_arrangement: Arrangement, the optimized arrangement
    '''

    obj_with_mesh = [obj for obj in arrangement.objs if obj.has_mesh]
    all_meshes = [obj.mesh for obj in obj_with_mesh]
    all_meshes = deepcopy(all_meshes)

    collision_manager = trimesh.collision.CollisionManager()
    scene = trimesh.Scene()

    print("\nSpatial optimization started...")
    overall_start_time = time.time()

    # Initialize the applied transformations for each object to the identity matrix
    # All transformations used during optimization are stored in this array, which is then used to update the arrangement at the end
    applied_transformations = np.dstack([np.eye(4)] * len(obj_with_mesh)).transpose(2, 0, 1)
    
    for i, obj in enumerate(obj_with_mesh):
        object_start_time = time.time()

        # Get the mesh of the object and place it at the bounding box centroid
        mesh = all_meshes[i]
        mesh.apply_transform(obj.bounding_box.no_scale_matrix)
        
        # Skip the first mesh as there is nothing to compare against
        if i >= 1:
            
            # ------------------------------------------------------------------------------ Push objects apart if in collision
            
            # Resolve collisions by moving the object away from the other objects
            if resolve_collisions:
                collision_iter = 0
                while collision_iter < collision_max_iters:
                    in_collision, contacts = collision_manager.in_collision_single(mesh, return_data=True)
                    if not in_collision:
                        break
                    
                    # Find the direction to separate the objects
                    contact_pts = np.array([contact.point for contact in contacts])
                    separate_direction = np.mean(mesh.centroid - contact_pts, axis=0)

                    # Weight the direction based on the size of the object
                    # The collision would be resolved faster when moving in the direction of the smallest dimension
                    weights = 1 / mesh.bounding_box_oriented.extents
                    weighted_direction = separate_direction * weights
                    weighted_direction /= np.linalg.norm(weighted_direction)

                    # Move the object in the direction of the weighted direction
                    translation = weighted_direction * collision_move_step
                    translation_matrix = trimesh.transformations.translation_matrix(translation)
                    mesh.apply_transform(translation_matrix)
                    applied_transformations[i] = np.dot(translation_matrix, applied_transformations[i])

                    collision_iter += 1
        
            # ------------------------------------------------------------------------------ Pull objects together for a tight fit

            # Make the object fit tightly with the previous objects by moving it towards the centroid of the previous object
            if make_tight:
                for tight_iter in range(make_tight_iters):
                    
                    # Find the direction to move the object towards the centroid of the previous object
                    centroid_direction = all_meshes[i - 1].centroid - mesh.centroid
                    centroid_direction /= np.linalg.norm(centroid_direction)

                    # Get the visible points of the object facing the centroid direction
                    surface_pts, face_idxs = trimesh.sample.sample_surface_even(mesh, 2048)
                    normals = mesh.face_normals[face_idxs]
                    visible_pts = surface_pts[np.dot(normals, centroid_direction) > 0.0]

                    # Find the intersection points of the rays from the visible points towards the centroid direction
                    combined_static_mesh: trimesh.Trimesh = trimesh.util.concatenate(all_meshes[:i])
                    ray_origins = visible_pts
                    ray_directions = np.tile(centroid_direction, (len(visible_pts), 1))
                    ray_intersections_pts, ray_idxs, _ = combined_static_mesh.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
                    corresponding_ray_origins = ray_origins[ray_idxs]

                    # Move the object towards the centroid direction until it touches the other objects
                    if len(ray_intersections_pts) > 0:

                        # Find the minimum distance to move the object
                        distances = np.linalg.norm(corresponding_ray_origins - ray_intersections_pts, axis=1)

                        # Move the object by the minimum distance, weighted by the iteration
                        move_distance = np.min(distances) * (0.5 + 0.5 * (tight_iter+1) / make_tight_iters)

                        # Move the object
                        translation = centroid_direction * move_distance
                        translation_matrix = trimesh.transformations.translation_matrix(translation)
                        mesh.apply_transform(translation_matrix)
                        applied_transformations[i] = np.dot(translation_matrix, applied_transformations[i])
        
        # Add the optimized mesh to the collision manager for the next iteration
        collision_manager.add_object(f"mesh_{i}", mesh)
        scene.add_geometry(mesh)
        print(f"Optimized mesh {i} in {(time.time() - object_start_time):.3f} s")
    
    # ------------------------------------------------------------------------------ Approximate gravity

    if approximate_gravity:
        combined_static_mesh: trimesh.Trimesh = trimesh.util.concatenate(all_meshes)
        global_min_y = np.min(combined_static_mesh.bounds[:, 1])

        # Add a floor plane at the minimum y value
        floor_plane: trimesh.Trimesh = trimesh.creation.box(extents=[10, 0.01, 10], transform=trimesh.transformations.translation_matrix([0, global_min_y - 0.005, 0]))
        scene.add_geometry(floor_plane, node_name="floor_plane")

        # Simulate gravity for each object
        for i, mesh in enumerate(all_meshes):

            # Get samples of the ground facing points of the mesh
            surface_pts, face_idxs = trimesh.sample.sample_surface_even(mesh, 2048)
            normals = mesh.face_normals[face_idxs]
            ground_facing_pts = surface_pts[np.dot(normals, [0, -1, 0]) > 0.0]

            # Prepare ray origins and directions for the ground facing points
            ground_ray_origins = ground_facing_pts
            ground_ray_directions = np.tile([0, -1, 0], (len(ground_facing_pts), 1))

            # First check if a majority of rays intersect with the static meshes, if yes, no need to approximate gravity
            other_static_mesh: trimesh.Trimesh = trimesh.util.concatenate(all_meshes[:i] + all_meshes[i+1:])
            intersection_pts, _, _ = other_static_mesh.ray.intersects_location(ground_ray_origins, ground_ray_directions, multiple_hits=False)

            if len(intersection_pts) / len(ground_ray_origins) < 0.15:
                # Simulate gravity
                ground_intersections_pts, ray_idxs, _ = floor_plane.ray.intersects_location(ground_ray_origins, ground_ray_directions, multiple_hits=False)
                if len(ground_intersections_pts) > 0:
                    corresponding_ray_origins = ground_ray_origins[ray_idxs]
                    min_ground_distance = np.min(np.linalg.norm(ground_intersections_pts - corresponding_ray_origins, axis=1))
                    gravity_translation = [0, -min_ground_distance, 0]
                    translation_matrix = trimesh.transformations.translation_matrix(gravity_translation)
                    mesh.apply_transform(translation_matrix)
                    applied_transformations[i] = np.dot(translation_matrix, applied_transformations[i])

        # Remove the floor plane from the scene
        scene_graph_geometry_nodes_dict = scene.graph.geometry_nodes
        node_keys = list(scene_graph_geometry_nodes_dict.keys())
        node_values = list(scene_graph_geometry_nodes_dict.values())
        scene.delete_geometry(node_keys[node_values.index(["floor_plane"])])

    # ------------------------------------------------------------------------------ Update the arrangement with the applied transformations
    optimized_arrangement = deepcopy(arrangement)
    for i, obj in enumerate(optimized_arrangement.objs):
        if obj.has_mesh:
            applied_transformation = applied_transformations[i]
            obj.bounding_box.centroid = np.dot(applied_transformation, np.append(obj.bounding_box.centroid, 1))[:3]
    
    print(f"Optimized all objects in {(time.time() - overall_start_time):.3f} s\n")
    return optimized_arrangement
