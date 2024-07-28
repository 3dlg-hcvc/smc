import os
import csv
import json
import yaml
import random
import trimesh
import numpy as np
import systems.gpt as gpt
from copy import deepcopy
from components.obj import Obj
from components.motif import Motif
from components.arrangement import Arrangement

# Preload the blocked asset IDs
block_ids_path = os.path.join(os.path.dirname(__file__), "retrieval_block_ids.yaml")
if os.path.exists(block_ids_path):
    with open(block_ids_path, "r") as f:
        block_ids = yaml.safe_load(f)
else:
    block_ids = []

hssd_wnsynsetkey_index = None

def _get_hssd_wnsynsetkey_index() -> dict[str, list[dict]]:
    '''
    Get the HSSD wordnet synset key index.

    Returns:
        hssd_wnsynsetkey_index: dictionary, the HSSD wordnet synset key index
    '''

    global hssd_wnsynsetkey_index
    if hssd_wnsynsetkey_index is None:
        hssd_wnsynsetkey_index = _load_hssd_wnsynsetkey_index()
    return hssd_wnsynsetkey_index

def _load_hssd_wnsynsetkey_index(hssd_path: str = "./hssd-models/",
                                 objects_csv_path: str = "semantics_objects.csv",
                                 index_json: str = "wnsynsetkey_index.json",
                                 force_recreate: bool = False) -> dict[str, list[dict]]:
    '''
    Load the HSSD wordnet synset key index. Create the index if it does not exist.
    The index is a dictionary indexed by WordNet synset keys that each contains a list of corresponding asset records.
    
    Args:
        hssd_path: string, the path to the HSSD dataset
        objects_csv_path: string, the path to the CSV file containing the summary of the assets
        index_json: string, the name of the HSSD wordnet synset key index JSON file
        force_recreate: bool, whether to force recreate the dictionary

    Returns:
        index: dictionary, the HSSD data index
    '''

    index_path = os.path.join(hssd_path, index_json)
    
    # Load the index if it exists
    if os.path.exists(index_path) and not force_recreate:
        with open(index_path, "r") as file:
            index = json.load(file)
        print("Loaded HSSD data index")

    # Create the index if it does not exist
    else:
        print("Creating HSSD data index...", end="")
        index: dict[str, list[dict]] = {}

        # Process the CSV file containing the summary of the assets
        with open(os.path.join(hssd_path, objects_csv_path), "r") as file:
            csv_content = csv.DictReader(file, delimiter=",", quotechar='"')

            # Group rows by the WordNet synset key
            for row_idx, row in enumerate(csv_content):
                wnsynsetkey = row["wnsynsetkey"]
                
                # Skip rows without a WordNet synset key or with multiple objects
                if wnsynsetkey == "" or row["hasMultipleObjects"].lower() == "true":
                    continue

                if wnsynsetkey not in index:
                    index[wnsynsetkey] = []

                index[wnsynsetkey].append({
                    "csv_row_idx": row_idx,
                    "id": row["id"],
                    "name": row["name"],
                    "up": row["up"],
                    "front": row["front"]
                })
        
        with open(index_path, "w") as file:
            json.dump(index, file, indent=4)
        
        print("Done")

    return index

def _get_wnsynsetkeys_for_labels(labels: list[str]) -> list[str]:
    '''
    Get the WordNet synset keys for the labels.

    Args:
        labels: list[string], the labels to get the WordNet synset keys for
    
    Returns:
        wnsynsetkeys: list[string], the WordNet synset keys for the labels
    '''

    num_objs = len(labels)
    all_wnsynsetkeys = list(_load_hssd_wnsynsetkey_index().keys())

    # ----- Validation function for this task -----
    def wnsynsetkeys_validation(response: str) -> tuple[bool, str, int]:
        try:
            response_json: dict[str, list[str]] = json.loads(gpt.extract_json(response))
        except json.JSONDecodeError as e:
            return False, f"Failed to decode the json response: {e}", -1

        valid = True
        error_message = ""

        if "wnsynsetkeys" not in response_json or len(response_json["wnsynsetkeys"]) != num_objs:
            valid = False
            error_message = f"Expected {num_objs} wnsynsetkeys, got {len(response_json['wnsynsetkeys'])}"
        else:
            for wnsynsetkey in response_json["wnsynsetkeys"]:
                if wnsynsetkey not in all_wnsynsetkeys:
                    valid = False
                    error_message = f"The WordNet synset key '{wnsynsetkey}' is invalid. Valid WordNet synset keys are: {all_wnsynsetkeys}"
                    break

        return valid, error_message, -1
    # ----- End of validation function -----
    
    # Extract the WordNet synset keys via LLM
    wnsynsetkeys_session = gpt.Session()
    wnsynsetkeys_response = wnsynsetkeys_session.send_with_validation("wnsynsetkeys", 
                                                                      {"wnsynsetkeys": ",".join(all_wnsynsetkeys),
                                                                       "object_labels": ",".join(labels)}, 
                                                                       wnsynsetkeys_validation)
    wnsynsetkeys_response: dict[str, list[str]] = json.loads(gpt.extract_json(wnsynsetkeys_response))
    wnsynsetkeys = wnsynsetkeys_response["wnsynsetkeys"]

    return wnsynsetkeys

def _check_side_rotations(motif_description: str, object_labels: list[str], threshold: float = 0.4) -> dict[str, bool]:
    '''
    Check which object labels need side rotations after mesh retrieval during orientation optimization.

    Args:
        motif_description: string, the description of the motif
        object_labels: list[string], the labels of the objects in the motif
        threshold: float, the threshold for determining whether an object needs side rotations

    Returns:
        need_side_rotations: dictionary, the object labels paired with whether they need side rotations
    '''

    # ----- Validation function for this task -----
    def side_rotations_validation(response: str) -> tuple[bool, str, int]:
        try:
            response_json: dict[str, float] = json.loads(gpt.extract_json(response))
        except json.JSONDecodeError as e:
            return False, f"Failed to decode the json response: {e}", -1

        valid = True
        error_message = ""

        if len(response_json.keys()) != len(object_labels):
            valid = False
            error_message += (
                f"Number of object labels in the json does not match the number of object labels in the input. "
                f"Expected: {len(object_labels)}, Got: {len(response_json.keys())}"
            )

        for response_label in response_json.keys():
            if not all(key in response_json[response_label] for key in ["correct", "incorrect"]):
                valid = False
                error_message += (
                    f"Object label '{response_label}' does not have the required keys in the json. "
                    f"Expected: ['correct', 'incorrect'], Got: {response_json[response_label].keys()}"
                )
                break
        
        return valid, error_message, -1
    # ----- End of validation function -----
    
    side_rotations_session = gpt.Session()
    side_rotations_response = side_rotations_session.send_with_validation("retrieval_mesh_rotations", 
                                                                          {"description": motif_description, 
                                                                           "object_labels": ", ".join(object_labels)}, 
                                                                           side_rotations_validation)
    side_rotations_response: dict[str, dict[str, float]] = json.loads(gpt.extract_json(side_rotations_response))

    # Determine which object labels need side rotations
    need_side_rotations = {label: False for label in object_labels}
    for label in need_side_rotations.keys():
        chances = side_rotations_response[label]
        need_side_rotations[label] = chances["incorrect"] >= threshold

    print(f"For motif '{motif_description}', the following object labels need side rotations:\n{need_side_rotations}\n")
    
    return need_side_rotations

def _optimize_mesh_rotation(obj: Obj, mesh: trimesh.Trimesh, try_side_rotations: bool = False) -> trimesh.Trimesh:
    '''
    Optimize the rotation of the mesh for the object based on bounding box fit.

    Args:
        obj: Obj, the object
        mesh: trimesh.Trimesh, the mesh to optimize
        try_side_rotations: bool, whether to try side rotations
    
    Returns:
        optimized_mesh: trimesh.Trimesh, the optimized mesh
    '''

    RADIAN_90 = np.radians(90)
    
    side_rotations = [
        (0, [1, 0, 0]) # No rotation
    ]
    if try_side_rotations:
        side_rotations.extend([
            (RADIAN_90, [1, 0, 0]), # Rotate 90 degrees around x-axis
            (RADIAN_90, [0, 0, 1])  # Rotate 90 degrees around z-axis
        ])
    
    target_dimensions = obj.bounding_box.full_size

    # Try rotating the mesh to fit the bounding box
    best_score = np.inf
    best_mesh = None
    for side_rotation in side_rotations:
        side_rotated_mesh = deepcopy(mesh)
        side_rotated_mesh.apply_transform(trimesh.transformations.rotation_matrix(*side_rotation))
        
        # Try rotating the mesh around the y-axis in 90 degree steps
        # The original orientation is tried the last such that if there are ties, the original orientation is chosen
        for y_rotation_step in reversed(range(4)):
            y_rotated_mesh = deepcopy(side_rotated_mesh)
            y_rotation_matrix = trimesh.transformations.rotation_matrix(y_rotation_step * RADIAN_90, [0, 1, 0])
            y_rotated_mesh.apply_transform(y_rotation_matrix)

            # Calculate the score based on the difference in dimensions
            y_rotated_mesh_extents = y_rotated_mesh.extents
            score = np.sum(np.abs(target_dimensions - y_rotated_mesh_extents))
            if score < best_score:
                best_score = score
                best_mesh = y_rotated_mesh

    return best_mesh

def _get_mesh_for_obj(obj: Obj, 
                      wnsynsetkey: str, 
                      try_side_rotations: bool = False, 
                      randomize: bool = False, 
                      use_top_k: int = 5, 
                      force_k: int = -1, 
                      avoid_used: bool = False, 
                      used_mesh_paths: list[str] = [], 
                      hssd_path: str = "./hssd-models/") -> tuple[trimesh.Trimesh, str]:
    '''
    Get a mesh for the object.

    Args:
        obj: Obj, the object to get the mesh for
        wnsynsetkey: string, the WordNet synset key for the object's label
        try_side_rotations: bool, whether to try side rotations during orientation optimization
        randomize: bool, whether to randomize the selection of the meshes
        use_top_k: int, the number of top candidates to consider
        force_k: int, the index of the mesh to force use, overrides other options
        avoid_used: bool, whether to avoid using meshes that have been used before
        used_mesh_paths: list[string], the list of paths to meshes that have been used before
        hssd_path: string, the path to the HSSD dataset
    
    Returns:
        mesh: trimesh.Trimesh, the mesh retrieved
        mesh_path: string, the path to the mesh file
    '''

    # Use the key to get a list of candidate objects
    hssd_index = _get_hssd_wnsynsetkey_index()
    obj_records: list[dict] = hssd_index[wnsynsetkey]

    # Find a best that best matches the object's bounding box
    top_scores = []
    top_meshes = []
    top_paths = []
    for record in obj_records:

        # Skip if the object is in the block list
        if record["id"] in block_ids:
            continue
        
        # Prepare the mesh path
        mesh_path = os.path.join(hssd_path, "objects", record["id"][0], f"{record['id']}.glb")
        
        # Skip if the mesh does not exist or has been used before (if avoid_used is True)
        if not os.path.exists(mesh_path) or (avoid_used and mesh_path in used_mesh_paths):
            continue

        mesh = trimesh.load(mesh_path, force="mesh")
        optimized_mesh = _optimize_mesh_rotation(obj, mesh, try_side_rotations) if try_side_rotations else mesh

        # Calculate the score based on the difference in dimensions
        target_dimensions = obj.bounding_box.full_size
        mesh_extents = optimized_mesh.extents
        score = np.sum(np.abs(target_dimensions - mesh_extents))
        top_scores.append(score)
        top_meshes.append(optimized_mesh)
        top_paths.append(mesh_path)

        # Keep the top k candidates
        if len(top_scores) > use_top_k:
            top_scores, top_meshes, top_paths = (list(t) for t in zip(*sorted(zip(top_scores, top_meshes, top_paths), key=lambda x: x[0])))
            top_scores.pop()
            top_meshes.pop()
            top_paths.pop()
        
    # Pick the mesh to use
    if force_k == -1:
        # Randomly choose from the top k candidates if randomize is True
        choice = random.randint(0, use_top_k - 1) if randomize else 0
    else:
        # Use the forced choice
        choice = force_k
    choice = min(choice, len(top_meshes) - 1)
    return top_meshes[choice], top_paths[choice]

def retrieve(description: str, 
             objs: list[Obj], 
             same_per_label: bool = False,
             randomize: bool = False,
             use_top_k: int = 5,
             force_k: int = -1, 
             avoid_used: bool = False) -> None:
    '''
    Retrieve the meshes for the objects.
    
    Args:
        description: string, the description of the motif
        objs: list[Obj], the objects to retrieve the meshes for
        same_per_label: bool, whether to use the same mesh for objects with the same label
        randomize: bool, whether to randomize the selection of the meshes
        use_top_k: int, the number of top candidates to consider
        force_k: int, the index of the mesh to force use, overrides other options
        avoid_used: bool, whether to avoid using meshes that have already been used

    Returns:
        None
    '''

    # Get the WordNet synset key for each label
    labels = list(set([obj.label for obj in objs]))
    label_to_wnsynsetkey = {}
    wnsynsetkeys = _get_wnsynsetkeys_for_labels(labels)
    for i, label in enumerate(labels):
        label_to_wnsynsetkey[label] = wnsynsetkeys[i]
    
    # See which object labels need side rotations
    need_side_rotations = _check_side_rotations(description, labels)

    # Retrieve the meshes for the objects
    used_mesh_paths = []
    mesh_dict = {}
    for obj in objs:

        if same_per_label and obj.label in mesh_dict:
            existing_mesh = deepcopy(mesh_dict[obj.label])
            obj.mesh = _optimize_mesh_rotation(obj, existing_mesh, need_side_rotations[obj.label])
            continue

        obj.mesh, mesh_path = _get_mesh_for_obj(obj, 
                                                label_to_wnsynsetkey[obj.label], 
                                                try_side_rotations=need_side_rotations[obj.label],
                                                randomize=randomize, 
                                                use_top_k=use_top_k, 
                                                force_k=force_k, 
                                                avoid_used=avoid_used, 
                                                used_mesh_paths=used_mesh_paths)
        if same_per_label:
            mesh_dict[obj.label] = obj.mesh
        
        used_mesh_paths.append(mesh_path)
    
    paths_str = "\n".join(used_mesh_paths)
    print(f"Retrieved meshes from:\n{paths_str}\n")

def motif_to_arrangement(motif: Motif, 
                         same_per_label: bool = False,
                         randomize: bool = False,
                         use_top_k: int = 5,
                         force_k: int = -1, 
                         avoid_used: bool = False) -> Arrangement:
    '''
    Convert a motif to an arrangement by retrieving meshes for the objects.
    
    Args:
        motif: Motif, the motif to convert to an arrangement by retrieving meshes
        same_per_label: bool, whether to use the same mesh for objects with the same label
        randomize: bool, whether to randomize the selection of the meshes during retrieval
        use_top_k: int, the number of top candidates to consider during retrieval
        force_k: int, the index of the mesh to force use, overrides other options
        avoid_used: bool, whether to avoid using meshes that have already been used
    
    Returns:
        arrangement: Arrangement, the arrangement created from the motif
    '''
    
    objs = deepcopy(motif.objs)
    retrieve(motif.description, objs, same_per_label, randomize, use_top_k, force_k, avoid_used)
    return Arrangement(objs, motif.description)
