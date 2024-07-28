import argparse
import json
import yaml
import libraries.library as library
import systems.gpt as gpt
import systems.validator as validator
from systems.glb_loader import load_glb
from components.arrangement import Arrangement
from components.program import Program

def load_example(file_path: str, description: str) -> Arrangement:
    '''
    Load an example arrangement from a GLB file.

    Args:
        file_path: string, the path to the GLB file
        description: string, the description of the arrangement
    
    Returns:
        example_arrangement: Arrangement, the loaded example arrangement
    '''

    _, input_objs = load_glb(file_path)
    example_arrangement = Arrangement(input_objs, description)
    example_arrangement.normalize()
    return example_arrangement

def make_high_level_observations(llm_session: gpt.Session, naive_program: Program) -> None:
    '''
    Prompt the LLM to make high level observations of the naive program.

    Args:
        llm_session: Session, the LLM session
        naive_program: Program, the naive program to observe
    
    Returns:
        None
    '''

    llm_session.send("optimize_highlevel_count", {"program": naive_program.code_string, "description": naive_program.description})
    llm_session.send("optimize_highlevel_general_pattern", {"description": naive_program.description})
    llm_session.send("optimize_highlevel_xyz_pattern")
    llm_session.send("optimize_highlevel_xyz_displacements")

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

def optimize_motif_program(llm_session: gpt.Session, naive_program: Program, example_arrangement: Arrangement) -> Program:
    '''
    Prompt the LLM to optimize the naive program to create a motif program.

    Args:
        llm_session: Session, the LLM session
        naive_program: Program, the naive program to optimize
        example_arrangement: Arrangement, the example arrangement
    
    Returns:
        optimized_program: Program, the optimized motif program
    '''
    
    # ----- Validation function for this task -----
    def optimize_validation(response: str) -> tuple[bool, str, int]:
        program = gpt.extract_program(response, naive_program.description)

        validations = [
            validator.validate_syntax,
            validator.validate_naive_listing,
            validator.validate_num_objects,
            validator.validate_centroids,
            validator.validate_bounding_boxes,
        ]

        arguments = [
            [program],
            [program],
            [program, example_arrangement.objs],
            [program, example_arrangement.objs],
            [program, example_arrangement.objs],
        ]

        for i, (validation, argument) in enumerate(zip(validations, arguments)):
            valid, error_message = validation(*argument)
            if not valid:
                return valid, error_message, i
        
        return True, "", -1
    # ----- End of validation function -----

    optimize_response = llm_session.send_with_validation("optimize_lowlevel", None, optimize_validation)
    optimized_program = gpt.extract_program(optimize_response, naive_program.description)
    return optimized_program

def observe_commonalities_and_differences(llm_session: gpt.Session, motif_type: str) -> None:
    '''
    Prompt the LLM to observe the high level commonalities and differences in programs of the same motif type.

    Args:
        llm_session: Session, the LLM session
        motif_type: string, the motif type
    
    Returns:
        None
    '''

    # Get all programs of the same motif type
    all_programs = ""
    if library.length(motif_type) > 0:
        loaded_programs = library.load(motif_type)
        for i, program in enumerate(loaded_programs):
            all_programs += f"Program {i + 1}. '{program.description}':\n{program.code_string}\n\n"
        print(f"Loaded {len(loaded_programs)} programs of motif type: {motif_type}\n")
    else:
        raise RuntimeError(f"No programs in library for motif type: {motif_type}")

    llm_session.send("generalize_high_level_commonalities", {"num_programs": str(len(loaded_programs)),
                                                             "motif_type": motif_type,
                                                             "all_programs": all_programs})
    llm_session.send("generalize_high_level_differences")
    llm_session.send("generalize_high_level_motif_reason", {"motif_type": motif_type})

def prepare_meta_program_info(llm_session: gpt.Session, motif_type: str) -> None:
    '''
    Prompt the LLM to prepare information for writing the meta-program.

    Args:
        llm_session: Session, the LLM session
        motif_type: string, the motif type
    
    Returns:
        None
    '''

    llm_session.send("generalize_low_level_arguments", {"motif_type": motif_type})
    llm_session.send("generalize_low_level_structure", {"motif_type": motif_type})

def write_meta_program(llm_session: gpt.Session, motif_type: str, refine_comments: bool = True) -> Program:
    '''
    Prompt the LLM to write the meta-program.

    Args:
        llm_session: Session, the LLM session
        motif_type: string, the motif type
        refine_comments: bool, whether to refine the comments of the meta-program
    
    Returns:
        meta_program: program, the written meta-program
    '''

    # Get all programs of the same motif type
    if library.length(motif_type) > 0:
        loaded_programs = library.load(motif_type)
    else:
        raise RuntimeError(f"No programs in library for motif type: {motif_type}")

    # Get the previous meta-program if available
    if library.length(motif_type, is_meta=True) > 0:
        past_meta_program = library.load(motif_type, is_meta=True)[0].code_string
    else:
        past_meta_program = "# NO PAST META-PROGRAM AVAILABLE"

    # ----- Validation function for this task -----
    def generalize_validation(response: str) -> tuple[bool, str, int]:
        meta_program = gpt.extract_program(response, motif_type)

        batch_recreate_response = llm_session.send("generalize_low_level_batch_recreate")
        try:
            recreate_calls: dict[str, str] = json.loads(gpt.extract_json(batch_recreate_response))
            recreate_calls = list(recreate_calls.values())
        except json.JSONDecodeError as e:
            return False, f"Failed to decode the json response: {e}", -1

        valid, error_message = validator.validate_meta_program(meta_program, recreate_calls, loaded_programs)

        return valid, error_message, 0 if not valid else -1
    # ----- End of validation function -----

    generalize_response = llm_session.send_with_validation("generalize_low_level", 
                                                           {"motif_type": motif_type, 
                                                            "past_meta_program": past_meta_program}, 
                                                            generalize_validation)
    meta_program = gpt.extract_program(generalize_response, motif_type)

    # Refine the documentation of the meta-program
    if refine_comments:
        # ----- Validation function for this task -----
        def refine_comments_validation(response: str) -> tuple[bool, str, int]:
            meta_program = gpt.extract_program(response, motif_type)
            valid, error_message = validator.validate_syntax(meta_program, require_objs=False)
            return valid, error_message, -1
        # ----- End of validation function -----

        refine_response = llm_session.send_with_validation("generalize_refine_comments", 
                                                        {"motif_type": motif_type}, 
                                                        refine_comments_validation)
        meta_program = gpt.extract_program(refine_response, motif_type)
    
    return meta_program

def learn(example_file_path: str, description: str) -> None:
    '''
    Learn a meta-program given an input arrangement.

    Args:
        example_file_path: string, the path to the example arrangement file
        description: string, the description of the arrangement
    
    Returns:
        None
    '''

    learn_session = gpt.Session()

    print("Loading example arrangement...")
    example_arrangement = load_example(example_file_path, description)
    print(f"Loaded arrangement with {len(example_arrangement.objs)} objects\n")

    naive_program = Program.from_arrangement(example_arrangement)
    print(f"Naive program:\n{naive_program.code_string}\n")

    print("Making high level observations of the naive program...")
    make_high_level_observations(learn_session, naive_program)
    print()

    print("Classifying motif type...")
    motif_type = classify_motif_type(learn_session, description)
    print(f"Classified motif type: {motif_type}\n")

    print("Optimizing naive program...")
    motif_program = optimize_motif_program(learn_session, naive_program, example_arrangement)
    print(f"Motif program:\n{motif_program.code_string}\n")

    library.store(motif_program, motif_type)
    print(f"Stored motif program in library under motif type: {motif_type}\n")

    print("Observing high level commonalities and differences of programs of the same motif type...")
    observe_commonalities_and_differences(learn_session, motif_type)
    print()

    print("Preparing meta-program information...")
    prepare_meta_program_info(learn_session, motif_type)
    print()

    print("Writing meta-program...")
    meta_program = write_meta_program(learn_session, motif_type)
    print(f"Final meta-program:\n{meta_program.code_string}\n")

    library.store(meta_program, motif_type, is_meta=True)
    print(f"Stored meta-program in library under motif type: {motif_type}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn a meta-program given an input arrangement")
    parser.add_argument("--file", type=str, help="The path to the example arrangement file")
    parser.add_argument("--desc", type=str, help="The description of the arrangement")
    args, extra = parser.parse_known_args()

    learn(args.file, args.desc)
