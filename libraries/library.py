import os
import json
from components.program import Program

# Define and create the directories for storing programs
base_dir = os.path.dirname(os.path.abspath(__file__))
lib_programs_dir = os.path.join(base_dir, "lib_programs")
lib_meta_programs_dir = os.path.join(base_dir, "lib_meta_programs")
os.makedirs(lib_programs_dir, exist_ok=True)
os.makedirs(lib_meta_programs_dir, exist_ok=True)

def store(program: Program, motif_type: str, is_meta: bool = False) -> None:
    '''
    Store the program in the program library.

    Args:
        program: Program, the program to store
        motif_type: string, the motif type of the program
        is_meta: bool, whether the program is a meta-program
    
    Returns:
        None
    '''

    if not is_meta:
        motif_program_dir = os.path.join(lib_programs_dir, motif_type)
        os.makedirs(motif_program_dir, exist_ok=True)
    
        program_id = len(os.listdir(motif_program_dir)) + 1
        file_path = os.path.join(motif_program_dir, f"{program_id}.json")

    else:
        program_id = -1
        file_path = os.path.join(lib_meta_programs_dir, f"{motif_type}.json")
    
    program_json = {
        "id": program_id,
        "description": program.description,
        "code_string": program.code_string
    }

    with open(file_path, "w") as file:
        json.dump(program_json, file, indent=4)

def load(motif_type: str, program_id: int | None = None, is_meta: bool = False) -> list[Program]:
    '''
    Load one or more programs from the program library.

    Args:
        motif_type: string, the motif type of the programs
        program_id: int, the ID of a specific program to load (if None, load all programs)
        is_meta: bool, whether to load a meta-program
    
    Returns:
        programs: list[Program], the loaded programs
    '''

    programs = []

    if not is_meta:
        motif_program_dir = os.path.join(lib_programs_dir, motif_type)

        if program_id is not None:
            file_paths = [os.path.join(motif_program_dir, f"{program_id}.json")]
        else:
            file_paths = [os.path.join(motif_program_dir, file_name) for file_name in os.listdir(motif_program_dir)]

    else:
        file_paths = [os.path.join(lib_meta_programs_dir, f"{motif_type}.json")]
    
    for file_path in file_paths:
        with open(file_path, "r") as file:
            program_json: dict = json.load(file)
            description: str = program_json["description"]
            code_string: str = program_json["code_string"]
            program = Program(code_string.split("\n"), description)
            programs.append(program)
    
    return programs

def length(motif_type: str, is_meta: bool = False) -> int:
    '''
    Return the number of programs in the program library.

    Args:
        motif_type: string, the motif type of the programs
        is_meta: bool, whether to return the number of meta-programs
    
    Returns:
        length: int, the number of programs in the program library
    '''

    if not is_meta:
        if os.path.exists(os.path.join(lib_programs_dir, motif_type)):
            return len(os.listdir(f"{lib_programs_dir}/{motif_type}"))
        else:
            return 0
    else:
        return int(os.path.exists(os.path.join(lib_meta_programs_dir, f"{motif_type}.json")))
