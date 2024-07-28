import json
import numpy as np
import systems.gpt as gpt
from copy import deepcopy
from components.bounding_box import BoundingBox
from components.obj import Obj
from components.program import Program
from utils.iou import sampling_iou

def _map_objects(objs1: list[Obj], objs2: list[Obj]) -> tuple[dict[Obj, Obj], np.ndarray]:
    '''
    Map objects in objs1 to objects of the same label in objs2 by closest centroid.

    Args:
        objs1: list[Obj], the first list of objects
        objs2: list[Obj], the second list of objects
    
    Returns:
        obj_mapping: dict[Obj, Obj], the mapping from objects in objs1 to objects in objs2
        distances: np.ndarray, the distances between the centroids of the pairs of objects
    '''

    obj_mapping: dict[Obj, Obj] = {}
    distances = np.ones(len(objs1)) * np.inf
    for i, obj1 in enumerate(objs1):
        for obj2 in objs2:
            if obj1.label != obj2.label:
                continue
            dist = np.linalg.norm(obj1.bounding_box.centroid - obj2.bounding_box.centroid)
            if dist < distances[i]:
                distances[i] = dist
                obj_mapping[obj1] = obj2
        
    return obj_mapping, distances

def validate_syntax(program: Program, require_objs: bool = True) -> tuple[bool, str]:
    '''
    Evaulate whether the syntax of the program is valid.

    Args:
        program: Program, the program to validate
        require_objs: bool, whether the program must have an "objs" variable
        
    Returns:
        valid: bool, whether the syntax of the program is valid
        error_message: str, the error message if the program has a syntax error (used for retrying)
    '''

    try:
        variable_dict = program.execute()
        if require_objs and "objs" not in variable_dict:
            return False, "The program is missing an list named `objs`. You should append all created objects to it.\n"
        else:
            return True, ""
    except Exception as e:
        return False, f"The program has a syntax error: {e}\n"

def validate_naive_listing(program: Program) -> tuple[bool, str]:
    '''
    Evaulate whether the program does not contain a naive listing of object attributes.

    Args:
        program: Program, the program to validate
    
    Returns:
        valid: bool, whether the program does not contain a naive listing of object attributes
        error_message: str, the error message if the program is invalid (used for retrying)
    '''

    # ----- Validation function for this task -----
    def response_validation(response: str) -> tuple[bool, str, int]:
        try:
            response_dict = json.loads(gpt.extract_json(response))
            if not all(key in response_dict for key in ["valid", "variable_names"]):
                return False, "The response does not contain the required keys: 'valid' and 'variable_names'.\n", -1
            return True, "", -1
        except:
            return False, "The response is not a valid JSON string.", -1
    # ----- End of validation function -----

    # Prompt GPT to analyze if the program contains a naive listing of object attributes
    gpt_validator = gpt.Session()
    response = gpt_validator.send_with_validation("validate_naive_listing", {"program": program.code_string}, response_validation)
    
    resposne_dict: dict[str, str | list] = json.loads(gpt.extract_json(response))

    if resposne_dict["valid"].lower() == "yes":
        return True, ""
    else:
        return False, f"These variables in your program are naive listings of object attributes: {resposne_dict['variable_names']}\n"
    
def validate_num_objects(program: Program, ground_truth_objs: list[Obj]) -> tuple[bool, str]:
    '''
    Evaulate whether the number of objects in the program matches the ground truth.

    Args:
        program: Program, the program to validate (assumed to be syntactically valid)
        ground_truth_objs: list[Obj], the ground truth objects to compare the program's objects to
    
    Returns:
        valid: bool, whether the number of objects in the program is valid
        error_message: str, the error message if the number of objects is invalid (used for retrying)
    '''

    valid = True
    error_message = ""

    # Execute the program and get the objects
    variable_dict = program.execute()
    program_objs: list[Obj] = variable_dict["objs"]

    # Group the objects by label
    program_obj_dict = {}
    for obj in program_objs:
        program_obj_dict.setdefault(obj.label, []).append(obj)

    ground_truth_obj_dict = {}
    for ground_truth_obj in ground_truth_objs:
        ground_truth_obj_dict.setdefault(ground_truth_obj.label, []).append(ground_truth_obj)
    
    # Check if the number of objects is the same for each object label
    for ground_truth_label, ground_truth_objs in ground_truth_obj_dict.items():
        if ground_truth_label not in program_obj_dict:
            error_message += f"The original program has {len(ground_truth_objs)} {ground_truth_label}, but there are none in your program.\n"
            valid = False
        elif len(ground_truth_objs) != len(program_obj_dict[ground_truth_label]):
            error_message += f"The original program has {len(ground_truth_objs)} {ground_truth_label}, but your program created {len(program_obj_dict[ground_truth_label])} {ground_truth_label}.\n"
            valid = False

            # Identify if there are duplicated objects
            if len(program_obj_dict[ground_truth_label]) > len(ground_truth_objs):
                for ground_truth_obj in ground_truth_objs:
                    match_count = 0
                    for program_obj in program_obj_dict[ground_truth_label]:
                        if np.all(np.isclose(ground_truth_obj.bounding_box.centroid, program_obj.bounding_box.centroid)):
                            match_count += 1
                    if match_count > 1:
                        error_message += f"The ground truth {ground_truth_label} at {ground_truth_obj.bounding_box.centroid.tolist()} is duplicated in your program.\n"                

    return valid, error_message

def validate_centroids(program: Program, ground_truth_objs: list[Obj], centroid_threshold: float = 0.01) -> tuple[bool, str]:
    '''
    Evaluate whether the centroids of the objects in the program match the ground truth objects.

    Args:
        program: Program, the program to validate (assumed to be syntactically valid)
        ground_truth_objs: list[Obj], the ground truth objects to compare the program's objects to
        centroid_threshold: float, the maximum distance for the centroids to be considered valid
    
    Returns:
        valid: bool, whether the centroids of the objects in the program are valid
        error_message: str, the error message if the centroids are invalid (used for retrying)
    '''

    valid = True
    error_data: dict[str, dict[str, list]] = {
        "gt_centroids": dict[str, list[np.ndarray]](),
        "program_centroids": dict[str, list[np.ndarray]]()
    }
    error_message = ""

    # Execute the program and get the objects
    variable_dict = program.execute()
    program_objs: list[Obj] = variable_dict["objs"]

    # Map the ground truth objects to the program objects by closest centroid
    obj_mapping, distances = _map_objects(ground_truth_objs, program_objs)

    # If two objects are mapped to the same ground truth object, the program is invalid
    if len(obj_mapping) != len(set(obj_mapping.values())):
        valid = False
        error_message += "Not all ground truth objects are covered by the program objects.\n"

    # Check the distance between the centroids of the pairs of objects
    for i, dist in enumerate(distances):
        if dist > centroid_threshold:
            valid = False
            mapped_obj = obj_mapping[ground_truth_objs[i]]
            error_data["gt_centroids"].setdefault(ground_truth_objs[i].label, []).append(ground_truth_objs[i].bounding_box.centroid)
            error_data["program_centroids"].setdefault(mapped_obj.label, []).append(mapped_obj.bounding_box.centroid)
    
    # Construct the error message
    for label, ground_truth_centroids in error_data["gt_centroids"].items():
        error_message += (
            f"{len(ground_truth_centroids)} {label}(s) should have centroids at {[centroid.tolist() for centroid in ground_truth_centroids]}, "
            f"but in your program the cloest {label}(s) are at {[centroid.tolist() for centroid in error_data['program_centroids'][label]]}.\n"
        )
    return valid, error_message

def validate_bounding_boxes(program: Program, 
                            ground_truth_objs: list[Obj], 
                            iou_threshold: float = 0.9, 
                            snap_centroids: bool = True) -> tuple[bool, str]:
    '''
    Evaulate whether the bounding boxes from executing the program match the ground truth objects.

    Args:
        program: Program, the program to validate (assumed to be syntactically valid)
        ground_truth_objs: list[Obj], the ground truth objects to compare the program's objects to
        iou_threshold: float, the minimum IoU for the bounding boxes to be considered valid
        snap_centroids: bool, whether to snap the centroids of the bounding boxes to the mapped ground truth objects
    
    Returns:
        valid: bool, whether the bounding box of the objects in the program is valid
        error_message: str, the error message if the bounding boxes are invalid (used for retrying)
    '''

    valid = True
    error_data: dict[str, dict[str, list]] = {
        "gt_bboxes": dict[str, list[BoundingBox]](),
        "program_bboxes": dict[str, list[BoundingBox]](),
        "ious": dict[str, list[float]]()
    }
    error_message = ""

    # Execute the program and get the objects
    variable_dict = program.execute()
    program_objs: list[Obj] = variable_dict["objs"]
    
    # Map the ground truth objects to the program objects by closest centroid
    obj_mapping, _ = _map_objects(ground_truth_objs, program_objs)

    # Snap the centroids of the bounding boxes to the mapped ground truth objects
    # This ensures the IoUs are not affected by slight differences in the centroids
    if snap_centroids:
        for ground_truth_obj, program_obj in obj_mapping.items():
            program_obj.bounding_box.centroid = ground_truth_obj.bounding_box.centroid
    
    # Check the IoU between the pairs of objects
    for ground_truth_obj, program_obj in obj_mapping.items():
        iou = sampling_iou(program_obj.bounding_box, ground_truth_obj.bounding_box)
        if iou < iou_threshold:
            valid = False
            error_data["gt_bboxes"].setdefault(ground_truth_obj.label, []).append(ground_truth_obj.bounding_box)
            error_data["program_bboxes"].setdefault(ground_truth_obj.label, []).append(program_obj.bounding_box)
            error_data["ious"].setdefault(ground_truth_obj.label, []).append(iou)
    
    # Construct the error message
    for label, ground_truth_bboxes in error_data["gt_bboxes"].items():
        error_message += (
            f"{len(ground_truth_bboxes)} {label}(s) at {[bbox.centroid.tolist() for bbox in ground_truth_bboxes]} "
            f"should have bounding boxes of half_size {[bbox.half_size.tolist() for bbox in ground_truth_bboxes]}, "
            f"but in your program they have bounding boxes of size "
            f"{[bbox.half_size.tolist() for bbox in error_data['program_bboxes'][label]]}."
            f"Through evaluating the 3D IoU between the bounding boxes, the IoU values are "
            f"{error_data['ious'][label]}, which are below the threshold of {iou_threshold}.\n"
        )

    return valid, error_message

def validate_relative_directions(program: Program, ground_truth_objs: list[Obj], angle_threshold: float = 5.0) -> tuple[bool, str]:
    '''
    Evaluate whether the relative directions of the objects in the program match the ground truth objects.

    Args:
        program: Program, the program to validate (assumed to be syntactically valid)
        ground_truth_objs: list[Obj], the ground truth objects to compare the program's objects to
        angle_threshold: float, the angle in degrees for the relative directions to be considered valid
    
    Returns:
        valid: bool, whether the relative directions of the objects in the program are valid
        error_message: str, the error message if the relative directions are invalid (used for retrying)
    '''

    valid = True
    flagged_ground_truth_objs = set()
    error_message = ""

    # Execute the program and get the objects
    variable_dict = program.execute()
    program_objs: list[Obj] = variable_dict["objs"]

    # Map the ground truth objects to the program objects by closest centroid
    obj_mapping, _ = _map_objects(ground_truth_objs, program_objs)

    # Check the relative directions of the objects
    for ground_truth_obj, program_obj in obj_mapping.items():
        
        error_count = 0
        for other_ground_truth_obj in ground_truth_objs:

            # Skip checking the relative direction of the same object
            if other_ground_truth_obj == ground_truth_obj:
                continue
                
            # Get the relative direction between the two ground truth objects
            gt_relative_direction = other_ground_truth_obj.bounding_box.centroid - ground_truth_obj.bounding_box.centroid
            gt_relative_direction /= (np.linalg.norm(gt_relative_direction) + 1e-12)

            # Get the relative direction between the mapped program objects
            other_program_obj = obj_mapping[other_ground_truth_obj]
            program_relative_direction = other_program_obj.bounding_box.centroid - program_obj.bounding_box.centroid
            program_relative_direction /= (np.linalg.norm(program_relative_direction) + 1e-12)

            # Check if the relative directions are within the angle threshold
            angle = np.arccos(np.dot(gt_relative_direction, program_relative_direction))
            if angle > np.radians(angle_threshold):
                error_count += 1
        
        # Flag the ground truth object if the relative directions are not accurate for all other objects
        # This indicates that the object is not in the correct position in the program
        if error_count == len(ground_truth_objs) - 1:
            valid = False
            flagged_ground_truth_objs.add(ground_truth_obj)
    
    error_message += (
        f"{len(flagged_ground_truth_objs)} objects in the ground truth are not in the correct positions in your program.\n"
        f"This is detected by comparing the relative directions between pair of objects in the ground truth and pair of objects in your program.\n"
        f"The objects that are not accurately recreated are:\n"
    )
    for flagged_ground_truth_obj in flagged_ground_truth_objs:
        error_message += f"{flagged_ground_truth_obj.label} at {flagged_ground_truth_obj.bounding_box.centroid.tolist()}\n"
    
    return valid, error_message

def validate_meta_program(meta_program: Program, function_calls: list[str], reference_programs: list[Program]) -> tuple[bool, str]:
    '''
    Evaulate whether the meta-program executed with the function call recreates the program's motif.

    Args:
        meta_program: Program, the meta-program to validate (assumed to be syntactically valid)
        function_calls: list[str], the function calls to execute the meta-program with
        reference_programs: list[Program], the reference programs to compare the meta-program to
    
    Returns:
        overall_valid: bool, whether the meta-program is valid
        overall_error_message: str, the error message if the meta-program is invalid (used for retrying)
    '''

    if len(function_calls) != len(reference_programs):
        raise ValueError("The number of function calls must match the number of reference programs")
    
    # Define the validations to run and the arguments to pass to them
    validations = [
        validate_syntax,
        validate_num_objects,
        validate_relative_directions,
    ]

    valids = [True] * len(reference_programs)
    error_messages = ["" for _ in range(len(reference_programs))]
    
    # Validate the meta-program with the function call for each reference program
    for reference_idx, reference_program in enumerate(reference_programs):

        # Execute the reference program to get the objects for comparison
        reference_program_variable_dict = reference_program.execute()
        reference_program_objs = reference_program_variable_dict["objs"]

        # Append the function call to the meta-program
        meta_program_with_call = deepcopy(meta_program)
        meta_program_with_call.append_code(f"objs = {function_calls[reference_idx]}")

        # Define the arguments to pass to the validation functions
        arguments = [
            [meta_program_with_call],
            [meta_program_with_call, reference_program_objs],
            [meta_program_with_call, reference_program_objs]
        ]

        # Validate the meta-program with the function call
        for i, (validation, argument) in enumerate(zip(validations, arguments)):
            valid, error_message = validation(*argument)
            if not valid:
                valids[reference_idx] = False
                error_messages[reference_idx] += error_message
                break

    # Construct the overall error message if not all programs are valid
    overall_valid = all(valids)
    overall_error_message = ""
    if not overall_valid:
        for i, (valid, message) in enumerate(zip(valids, error_messages)):
            if not valid:
                overall_error_message += f"Program {i + 1}:\n{message}\n"

    return overall_valid, overall_error_message
