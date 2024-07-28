import os
import argparse
import time
import json
import yaml
import libraries.library as library
import systems.gpt as gpt
import systems.validator as validator
import systems.retriever as retriever
import systems.spatial_optimizer as spatial_optimizer
from copy import deepcopy
from components.motif import Motif
from components.arrangement import Arrangement
from components.program import Program

def classify_motif_type(llm_session: gpt.Session, description: str) -> str:
    '''
    Prompt the LLM to classify the motif type of the description.

    Args:
        llm_session: Session, the LLM session
        description: string, the description of the arrangement
    
    Returns:
        motif_type: string, the classified motif type
    '''

    with open("motif_types.yaml", "r") as f:
        motif_types = yaml.safe_load(f)["types"].keys()

    # ----- Validation function for this task -----
    def classify_validation(response: str) -> tuple[bool, str, int]:
        motif_type = response.strip().lower()
        valid = motif_type in motif_types or (motif_type.startswith("letter_") and motif_type[-1] in "abcdefghijklmnopqrstuvwxyz")
        error_message = f"The motif type '{motif_type}' is invalid. Valid motif types are: {motif_types}" if not valid else ""
        return valid, error_message, -1
    # ----- End of validation function -----

    motif_type = llm_session.send_with_validation("classify", {"description": description}, classify_validation)
    return motif_type

def write_function_call(llm_session: gpt.Session, motif_type: str, description: str, meta_program: Program) -> Program:
    '''
    Write a function call to execute the meta-program given the motif type and description.

    Args:
        llm_session: Session, the LLM session
        motif_type: string, the motif type of the description
        description: string, the description of the arrangement
        meta_program: Program, the meta-program to execute
    
    Returns:
        meta_program_with_call: Program, the meta-program with the function call appended
    '''

    # ----- Validation function for this task -----
    def inference_validation(response: str) -> tuple[bool, str, int]:
        function_call = gpt.extract_code(response)
        meta_program_with_call = deepcopy(meta_program)
        meta_program_with_call.append_code(f"objs = {function_call}")
        valid, error_message = validator.validate_syntax(meta_program_with_call)
        return valid, error_message, 0 if not valid else -1
    # ----- End of validation function -----

    function_call = llm_session.send_with_validation("inference",
                                                     {"motif_type": motif_type,
                                                     "description": description,
                                                     "meta_program": meta_program.code_string},
                                                     inference_validation)
    function_call = gpt.extract_code(function_call)
    meta_program_with_call = deepcopy(meta_program)
    meta_program_with_call.append_code(f"objs = {function_call}")

    return meta_program_with_call

def run_spatial_optimization(llm_session: gpt.Session, new_arrangement: Motif, description: str, no_gravity: bool) -> Arrangement:
    '''
    Run spatial optimization on the new arrangement.

    Args:
        llm_session: Session, the LLM session
        new_arrangement: Motif, the new arrangement to optimize
        no_gravity: bool, whether to turn off gravity approximation
    
    Returns:
        optimized_arrangement: Arrangement, the optimized arrangement
    '''

    # Check whether the objects in the arrangement need to be in close contact
    # ----- Validation function for this task -----
    def spatial_optimization_touch_validation(response: str) -> tuple[bool, str, int]:
        try:
            response_json: dict[str, float] = json.loads(gpt.extract_json(response))
        except json.JSONDecodeError as e:
            return False, f"Failed to decode the json response: {e}", -1
        
        valid = all(key in response_json for key in ["touch", "no_touch"])
        error_message = "The json must have the keys 'touch' and 'no_touch'." if not valid else ""
        return valid, error_message, -1
    # ----- End of validation function -----
    
    need_touch = llm_session.send_with_validation("spatial_optimization_touch",
                                                  {"description": description},
                                                  spatial_optimization_touch_validation)
    need_touch: dict[str, float] = json.loads(gpt.extract_json(need_touch))
    need_touch = need_touch["touch"] > need_touch["no_touch"]

    # Optimize the arrangement
    optimized_arrangement = spatial_optimizer.optimize(new_arrangement, make_tight=need_touch, approximate_gravity=not no_gravity)

    return optimized_arrangement

def save_results(optimized_arrangement: Arrangement, meta_program_with_call: Program, description: str, out_dir: str) -> None:
    '''
    Save the optimized arrangement and the program.

    Args:
        optimized_arrangement: Arrangement, the optimized arrangement
        meta_program_with_call: Program, the meta-program with the function call appended
        description: string, the description of the arrangement
        out_dir: string, the directory to save the output arrangement and program
    
    Returns:
        None
    '''

    # Create a directory to save the results
    save_name = description.replace(" ", "_")
    save_dir = os.path.join(out_dir, f"{save_name}@{time.strftime('%y%m%d-%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    # Save the arrangement
    optimized_arrangement.save(os.path.join(save_dir, f"{save_name}.glb"))
    
    # Save the program
    program_json = {
        "description": description,
        "code_string": meta_program_with_call.code_string
    }
    with open(os.path.join(save_dir, f"program.json"), "w") as f:
        json.dump(program_json, f, indent=4)

def inference(args: argparse.Namespace) -> None:
    '''
    Generate an arrangement given the description.

    Args:
        args: argparse.Namespace, the arguments for inference
    
    Returns:
        None
    '''

    description = args.desc
    out_dir = args.out_dir
    retrieval_same_per_label = args.same_per_label
    retrieval_no_reuse = args.no_reuse
    retrieval_no_randomize = args.no_randomize
    retrieval_use_top_k = args.use_top_k
    retrieval_force_k = args.force_k
    spatial_optimization_no_gravity = args.no_gravity

    inference_session = gpt.Session()

    print("Classifying motif type...")
    motif_type = classify_motif_type(inference_session, args.desc)
    print(f"Classified motif type: {motif_type}\n")

    meta_program = library.load(motif_type, is_meta=True)[0]
    print(f"Loaded meta-program:\n{meta_program.code_string}\n")

    print("Getting function call for meta-program...")
    meta_program_with_call = write_function_call(inference_session, motif_type, description, meta_program)

    print("Executing meta-program with function call...\n")
    execute_result = meta_program_with_call.execute()
    objs = execute_result["objs"]
    motif = Motif(objs, description)

    print("Retrieving meshes for the motif to create an arrangement...")
    new_arrangement = retriever.motif_to_arrangement(motif,
                                                     same_per_label=retrieval_same_per_label,
                                                     randomize=not retrieval_no_randomize,
                                                     use_top_k=retrieval_use_top_k,
                                                     force_k=retrieval_force_k,
                                                     avoid_used=retrieval_no_reuse)
    
    print("Optimizing the arrangement spatially...")
    optimized_arrangement = run_spatial_optimization(inference_session, new_arrangement, description, spatial_optimization_no_gravity)

    print("Saving the optimized arrangement and the program...")
    save_results(optimized_arrangement, meta_program_with_call, description, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn a meta-program given an input arrangement")
    parser.add_argument("--desc", type=str, help="The description of the arrangement to generate")
    parser.add_argument("--out_dir", type=str, default="outputs", help="The directory to save the output arrangement and program")
    parser.add_argument("--same_per_label", action="store_true", help="Whether to use the same mesh per label in retrieval")
    parser.add_argument("--no_randomize", action="store_true", help="Turn off randomization in retrieval")
    parser.add_argument("--use_top_k", type=int, default=5, help="The number of top meshes to use when randomizing in retrieval")
    parser.add_argument("--force_k", type=int, default=-1, help="Force to use the kth mesh in retrieval, overrides other options")
    parser.add_argument("--no_reuse", action="store_true", help="Whether to avoid reusing the same mesh in retrieval")
    parser.add_argument("--no_gravity", action="store_true", help="Turn off gravity approximation in spatial optimization")

    args, extra = parser.parse_known_args()

    inference(args)
