import struct
import trimesh
import numpy as np
import pygltflib as gltf
from components.obj import Obj
from components.bounding_box import BoundingBox

def _get_unpack_format(component_type: int) -> tuple[str, int]:
    '''
    Get the unpack format and the size of the component type.

    Args:
        component_type: int, the component type of the accessor (glTF 2.0)

    Returns:
        type_char: string, the unpack format of the component type
        unit_size: int, the size of the component type
    '''

    match component_type:
        case 5120: # BYTE
            return "b", 1
        case 5121: # UNSIGNED_BYTE
            return "B", 1
        case 5122: # SHORT
            return "h", 2
        case 5123: # UNSIGNED_SHORT
            return "H", 2
        case 5125: # UNSIGNED_INT
            return "I", 4
        case 5126: # FLOAT
            return "f", 4
        case _:
            raise ValueError(f"Unknown component type: {component_type}")

def _get_num_components(type: str) -> int:
    '''
    Get the number of components of the accessor type.

    Args:
        type: string, the type of the accessor (glTF 2.0)

    Returns:
        num_components: int, the number of components of the accessor type
    '''

    match type:
        case "SCALAR":
            return 1
        case "VEC2":
            return 2
        case "VEC3":
            return 3
        case "VEC4":
            return 4
        case "MAT2":
            return 4
        case "MAT3":
            return 9
        case "MAT4":
            return 16
        case _:
            raise ValueError(f"Unknown type: {type}")

def _read_buffer(glb_file: gltf.GLTF2, accessor_idx: int) -> list:
    '''
    Read the data buffer pointed by the accessor.

    Args:
        glb_file: GLTF2, the glTF 2.0 file
        accessor_idx: int, the index of the accessor
    
    Returns:
        results: list, the data pointed by the accessor
    '''

    # Get data via accessor, buffer view, and buffer
    accessor = glb_file.accessors[accessor_idx]
    buffer_view = glb_file.bufferViews[accessor.bufferView]
    buffer = glb_file.buffers[buffer_view.buffer]
    data = glb_file.get_data_from_buffer_uri(buffer.uri)

    # Find out how to unpack the data
    type_char, unit_size = _get_unpack_format(accessor.componentType)
    num_components = _get_num_components(accessor.type)
    unpack_format = f"<{type_char * num_components}"
    data_size = unit_size * num_components

    # Read the data
    results = []
    for i in range(accessor.count):
        idx = buffer_view.byteOffset + accessor.byteOffset + i * data_size
        binary_data = data[idx:idx+data_size]
        result = struct.unpack(unpack_format, binary_data)
        results.append(result)

    return results

def _load_mesh(glb_file: gltf.GLTF2, mesh: gltf.Mesh) -> tuple[list, list, list]:
    '''
    Load the mesh data from a glTF 2.0 mesh.

    Args:
        glb_file: GLTF2, the glTF 2.0 file
        mesh: Mesh, the glTF 2.0 mesh
    
    Returns:
        vertices: list, the vertices of the mesh
        normals: list, the normals of the mesh
        faces: list, the faces of the mesh
    '''

    vertices = []
    normals = []
    faces = []

    for primitive in mesh.primitives:
        # Read vertex locations
        if primitive.attributes.POSITION is not None:
            loaded_vertices = _read_buffer(glb_file, primitive.attributes.POSITION)
            vertices.extend(loaded_vertices)
        else:
            raise ValueError(f"No vertex positions found in the glTF file")

        # Read vertex normals
        if primitive.attributes.NORMAL is not None:
            loaded_normals = _read_buffer(glb_file, primitive.attributes.NORMAL)
            normals.extend(loaded_normals)
        else:
            raise ValueError(f"No vertex normals found in the glTF file")

        # Read faces
        if primitive.indices is not None:
            loaded_faces = _read_buffer(glb_file, primitive.indices)
            faces.extend(loaded_faces)
        else:
            raise ValueError(f"No faces found in the glTF file")
    
    return vertices, normals, faces

def _load_all_meshes(glb_file: gltf.GLTF2, main_node: gltf.Node) -> trimesh.Trimesh:
    '''
    Given a main node, load all the meshes in the node and its children.

    Args:
        glb_file: GLTF2, the glTF 2.0 file
        node: Node, the glTF 2.0 node
    
    Returns:
        full_mesh: Trimesh, the full mesh of the node and its children
    '''

    full_mesh = trimesh.Trimesh(vertices=[], faces=[])
    all_nodes = glb_file.nodes

    # Load the mesh in the main node if it exists
    if main_node.mesh is not None:
        mesh = glb_file.meshes[main_node.mesh]
        loaded_vertices, loaded_normals, loaded_faces = _load_mesh(glb_file, mesh)
        part_mesh = trimesh.Trimesh(vertices=loaded_vertices, faces=np.array(loaded_faces).reshape(-1, 3), vertex_normals=loaded_normals)
        full_mesh += part_mesh
    
    # Load the meshes in the children of the main node if they exist
    for child_node_id in main_node.children:
        child_node = all_nodes[child_node_id]
        part_mesh = _load_all_meshes(glb_file, child_node)
        full_mesh += part_mesh
    
    return full_mesh

def load_glb(file_path: str) -> tuple[gltf.GLTF2, list[Obj]]:
    '''
    Load a glb file.

    Args:
        file_path: string, the path to the glb file
    
    Returns:
        glb_file: GLTF2, the glTF 2.0 file
        objs_in_file: list, the objects in the file
    '''

    # TODO: Make less assumptions about the structure of the glTF file

    glb_file = gltf.GLTF2().load(file_path)
    objs_in_file: list[Obj] = []
    
    # Load the main scene and the scene's world nodes
    all_nodes = glb_file.nodes
    main_scene = glb_file.scenes[glb_file.scene]
    world_node_ids = main_scene.nodes
    
    # Load the objects in the world nodes
    for world_node_id in world_node_ids:
        main_node_ids = all_nodes[world_node_id].children

        # Load the objects in the main nodes
        # Each main node is a separate object
        for main_node_id in main_node_ids:
            main_node = all_nodes[main_node_id]

            # ----- Matrix -----
            # Note: This logic is not general!
            # Assumes that the mesh_root_node has a matrix for the mesh
            # And the main_node has a matrix for placing the mesh in the world
            if main_node.matrix:
                main_node_matrix = np.array(main_node.matrix).reshape(4, 4, order="F")
            else:
                translation = main_node.translation if main_node.translation else [0, 0, 0]
                rotation = main_node.rotation if main_node.rotation else [1, 0, 0, 0]
                scale = main_node.scale if main_node.scale else [1, 1, 1]
                translation_matrix = trimesh.transformations.translation_matrix(translation)
                rotation_matrix = trimesh.transformations.quaternion_matrix(rotation)
                scale_matrix = np.diag(scale + [1])
                main_node_matrix = translation_matrix @ rotation_matrix @ scale_matrix

            if len(main_node.children) > 0:
                mesh_root_node_matrix = all_nodes[main_node.children[0]].matrix
                if mesh_root_node_matrix is not None:
                    mesh_root_node_matrix = np.array(mesh_root_node_matrix).reshape(4, 4, order="F")
            else:
                mesh_root_node_matrix = None

            # ----- Mesh -----
            main_node_mesh = _load_all_meshes(glb_file, main_node)
            
            # Apply the mesh_root_node_matrix if it exists
            if mesh_root_node_matrix is not None:
                main_node_mesh.apply_transform(mesh_root_node_matrix)

            # ----- Bounding box -----
            # Compute the oriented bounding box of the mesh
            centroid = main_node_mesh.bounding_box_oriented.centroid + main_node_matrix[:3, 3]
            half_size = main_node_mesh.bounding_box_oriented.extents / 2
            main_node_bounding_box = BoundingBox(centroid, half_size, main_node_matrix[:3, :3])
            
            # ----- Obj -----
            # TODO: Allow the label to be provided separately
            label = main_node.extras["semantics"]["label"] if "semantics" in main_node.extras else "unknown"

            obj = Obj(label, main_node_bounding_box, main_node_mesh, main_node_matrix)
            objs_in_file.append(obj)

    return glb_file, objs_in_file
