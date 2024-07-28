import os
import yaml
from typing import Callable
from dotenv import load_dotenv
from openai import OpenAI
from components.program import Program

# Create a .env file in the project root directory and add your OpenAI API key as OPENAI_API_KEY=<your_key>
load_dotenv()
client = OpenAI()

# Load the predefined prompts for the LLM
with open(os.path.join(os.path.dirname(__file__), "prompts.yaml")) as file:
    predefined_prompts: dict[str, str] = yaml.safe_load(file)

# Load the motif type information
with open("motif_types.yaml", "r") as f:
    motif_type_info = yaml.safe_load(f)["types"]

class Session:
    def __init__(self) -> None:
        '''
        Initialize a Session for interacting with GPT that keeps track of the messages and responses history.

        Returns:
            None
        '''

        self.client = client
        self.past_tasks: list[str] = []
        self.past_messages = [{"role": "system", "content": predefined_prompts["system"]}]
        self.past_responses: list[str] = []
    
    def send(self, task: str, prompt_info: dict[str, str] | None = None) -> str:
        '''
        Send a message of a specific task to the GPT-4 model and return the response.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            
        Returns:
            response: string, the response from the model
        '''

        print(f"$ --- Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info)
        self._send(prompt)
        response = self.past_responses[-1]
        print(f"$ --- Response:\n{response}\n")

        return response
    
    def send_with_validation(self, task: str, 
                             prompt_info: dict[str, str] | None = None, 
                             validation: Callable[[str], tuple[bool, str, int]] | None = None,
                             retry: int = 10) -> str:
        '''
        Send a message of a specific task to the GPT-4 model and return the response after validating it.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            validation: function, the validation function to validate the response for the task
                The function should return a tuple of three values:
                - bool: whether the response is valid
                - string: the error message if the response is invalid
                - integer: the index of the error prompt to use for retry (-1 if not applicable)
            retry: integer, the number of retries for the task
        
        Returns:
            response: string, the response from the model
        '''
        
        response = self.send(task, prompt_info)

        # Validate the response
        count = 0   # Includes the first try above
        while count < retry:
            if validation is not None:
                valid, error_message, error_index = validation(response)

                # Retry starts here
                if not valid:
                    print(f"$ --- Validation failed for task {task} at try {count+1}")
                    print(f"$ --- Error message: {error_message}")
                    
                    count += 1
                    if count < retry:
                        print(f"$ --- Retrying task {task} [try {count+1} / {retry}]\n")
                        
                        # Get the specific retry prompt using the error index
                        # The retry prompts are named as "<task_name>_feedback_<description>"
                        # The error index is used to get the specific retry prompt in the order they are defined in the prompts.yaml file
                        retry_prompt_keys = [key for key in predefined_prompts.keys() if task in key and "feedback" in key]
                        
                        if retry_prompt_keys:
                            retry_task_name = retry_prompt_keys[error_index]
                        else:
                            # If there is no specific retry prompt, use the generic one
                            retry_task_name = "invalid_response"
                        response = self.send(retry_task_name, {"feedback": error_message})
                else:
                    print(f"$ --- Validation passed for task {task} at try {count+1}\n")
                    break

        if count == retry:
            raise RuntimeError(f"$ --- Validation failed for task {task} after {retry} retries")

        return response
    
    def _make_prompt(self, task: str, prompt_info: dict[str, str] | None) -> str:
        '''
        Make a prompt for the LLM model.

        Args:
            task: string, the task of the prompt
            prompt_info: dictionary, the extra information for making the prompt for the task

        Returns:
            prompt: string, the prompt for the LLM model
        '''

        # Get the predefined prompt for the task
        prompt = predefined_prompts[task]

        # Check for task-specific required information ()
        # All tasks that require extra information should have a case here
        valid = True
        match task:
            case "optimize_highlevel_count":
                valid = all(key in prompt_info for key in ["program", "description"])

            case "optimize_highlevel_general_pattern" | \
                 "classify" | \
                 "spatial_optimization_touch":
                valid = "description" in prompt_info

            case "optimize_lowlevel_feedback_syntax" | \
                 "optimize_lowlevel_feedback_naive_listing" | \
                 "optimize_lowlevel_feedback_num_objs" | \
                 "optimize_lowlevel_feedback_centroids"| \
                 "optimize_lowlevel_feedback_bounding_boxes"| \
                 "invalid_response":
                valid = "feedback" in prompt_info

            case "validate_naive_listing":
                valid = "program" in prompt_info
            
            case "generalize_high_level_commonalities":
                valid = all(key in prompt_info for key in ["num_programs", "motif_type", "all_programs"])
            
            case "generalize_high_level_motif_reason" | \
                 "generalize_low_level_arguments" | \
                 "generalize_low_level_structure" | \
                 "generalize_refine_comments":
                valid = "motif_type" in prompt_info
            
            case "generalize_low_level":
                valid = all(key in prompt_info for key in ["motif_type", "past_meta_program"])
            
            case "inference":
                valid = all(key in prompt_info for key in ["motif_type", "description", "meta_program"])
            
            case "wnsynsetkeys":
                valid = all(key in prompt_info for key in ["wnsynsetkeys", "object_labels"])
            
            case "retrieval_mesh_rotations":
                valid = all(key in prompt_info for key in ["description", "object_labels"])

        if not valid:
            raise ValueError(f"Extra information is required for the task: {task}")
        
        # Add the motif type information as extra information for specific tasks
        if task in ("classify", "generalize_high_level_motif_reason"):
            motif_info_str = "\n".join([f"{i+1}. {motif_type} - {info}" for i, (motif_type, info) in enumerate(motif_type_info.items())])
            prompt_info["motif_info"] = motif_info_str

        # Replace the placeholders in the prompt with the information
        if prompt_info is not None:
            for key in prompt_info:
                prompt = prompt.replace(f"<{key.upper()}>", prompt_info[key])

        return prompt
    
    def _send(self, new_message: str) -> None:
        '''
        Send a message to GPT along with the past messages and store the response.

        Args:
            new_message: string, the new message to be sent to the model
        
        Returns:
            None
        '''

        self.past_messages.append({"role": "user", "content": new_message})
        
        completion = self.client.chat.completions.create(
            # model="gpt-4-turbo",
            model="gpt-4o",
            messages=self.past_messages
        )
        
        response = completion.choices[0].message.content
        self.past_messages.append({"role": "assistant", "content": response})
        self.past_responses.append(response)

def extract_program(response: str, description: str) -> Program:
    '''
    Extract the program from the response of the LLM.

    Args:
        response: string, the response from the LLM
        description: string, the description of the program

    Returns:
        program: Program, the program extracted from the response
    '''

    if "```python" in response:
        response = response.split("```python\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    code = response.split("\n")
    program = Program(code, description)

    return program

def extract_code(response: str) -> str:
    '''
    Extract the code from the response of the LLM.

    Args:
        response: string, the response from the LLM
    
    Returns:
        code: string, the code extracted from the response
    '''

    if "```python" in response:
        response = response.split("```python\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response

def extract_json(response: str) -> dict:
    '''
    Extract the JSON object from the response of the LLM.

    Args:
        response: string, the response from the LLM
    '''

    if "```json" in response:
        response = response.split("```json\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response
