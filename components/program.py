from __future__ import annotations

import numpy as np
from components.arrangement import Arrangement
from components.language import *

class Program():
    def __init__(self, code: list[str], description: str) -> None:
        '''
        Initialize a program.

        Args:
            code: list of strings, the code of the program
            description: string, the description of the program
        
        Returns:
            None
        '''

        self.code = code
        self.description = description

    @property
    def code_string(self) -> str:
        '''
        Return the code of the program in a single string.

        Returns:
            code_string: string, the code of the program in a single string
        '''

        return "\n".join(self.code)
    
    def append_code(self, code: str) -> None:
        '''
        Append code to the program.

        Args:
            code: string, the code to append
        
        Returns:
            None
        '''

        self.code.append(code)
    
    def execute(self) -> dict:
        '''
        Execute the program and return the local variables.
        
        Returns:
            locals: dict, the local variables after executing the program
        '''

        locals = {}
        exec(self.code_string, globals(), locals)
        return locals
    
    @classmethod
    def from_arrangement(cls, arrangement: Arrangement) -> Program:
        '''
        Create a program from an arrangement.

        Args:
            arrangement: Arrangement, an arrangement of objects

        Returns:
            program: Program, the program of the arrangement
        '''

        code = []
        code.append("objs = []")

        for i, obj in enumerate(arrangement.objs):
            label = obj.label
            centroid = obj.bounding_box.centroid.tolist()
            half_size = obj.bounding_box.half_size.tolist()
            coord_axes = obj.bounding_box.coord_axes

            # The object is initially aligned with the canonical coordinate system, so we need to rotate it to the correct orientation
            # For user convenience, the rotation operation is decomposed into three rotations around the x, y, and z axes
            # Here, we need to extract the three rotation angles needed to rotate the object to the correct orientation
            # In particular, we expect most objects only has a single rotation around the up axis (y axis), so we choose a rotation order of YXZ
            rotation_matrix = coord_axes
            y_angle = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
            x_angle = np.arcsin(-rotation_matrix[1, 2])
            z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[1, 1])
            
            code.append(f"obj_{i+1}_half_size = {half_size}")
            code.append(f"obj_{i+1}_centroid = {centroid}")
            code.append(f"obj_{i+1} = create('{label}', obj_{i+1}_half_size)")
            code.append(f"move(obj_{i+1}, obj_{i+1}_centroid[0], obj_{i+1}_centroid[1], obj_{i+1}_centroid[2])")

            for axis, angle in [("y", y_angle), ("x", x_angle), ("z", z_angle)]:
                if abs(angle) > 1e-5:
                    code.append(f"rotate(obj_{i+1}, '{axis}', {np.rad2deg(angle).round(1)})")
            
            code.append(f"objs.append(obj_{i+1})")
        
        program = Program(code, arrangement.description)
        return program
