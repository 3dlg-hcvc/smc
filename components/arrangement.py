import trimesh
import numpy as np
from copy import deepcopy
from components.obj import Obj

class Arrangement:
    def __init__(self, objs: list[Obj], description: str) -> None:
        '''
        Initialize an arrangement.

        Args:
            objs: list of Obj, the objects in the motif
            description: string, the description of the motif
        
        Returns:
            None
        '''

        self.objs = objs
        self.description = description
    
    def __str__(self) -> str:
        return self.description
    
    def normalize(self) -> None:
        '''
        Normalize the arrangement's objects so that one of the objects is at the origin and the rest are relative to it.

        Args:
            None

        Returns:
            None
        '''
        
        # The first object will be the one at the origin
        origin_obj = self.objs[0]

        # Move all the other objects relative to the origin object
        for obj in self.objs[1:]:
            obj.bounding_box.centroid -= origin_obj.bounding_box.centroid
            obj.bounding_box.centroid = np.round(obj.bounding_box.centroid, 5)
        
        # Move the origin object to the origin
        origin_obj.bounding_box.centroid = np.array([0, 0, 0])
    
    def save(self, file_path: str = "saved_arrangement.glb", minus_mesh_bbox_centroid: bool = False) -> None:
        '''
        Save the arrangement to a .glb file.

        Args:
            file_path: string, the file path for saving the arrangement
            minus_mesh_bbox_centroid: bool, whether to subtract the mesh bounding box centroid
                                      (this fixes the issue of the arrangement being offseted from the original arrangement
                                       when saving an arrangement that was loaded from a glb file using the glb_loader;
                                       this is not needed when the arrangement is created through the inference process)

        Returns:
            None
        '''
        
        scene = trimesh.Scene()
        for obj in self.objs:
            mesh = deepcopy(obj.mesh)
            mesh.apply_transform(obj.bounding_box.no_scale_matrix)
            
            if minus_mesh_bbox_centroid:
                mesh.apply_translation(-obj.mesh.bounding_box.centroid)
            
            scene.add_geometry(mesh)
        
        scene.export(file_path)
        print(f"Arrangement saved to {file_path}")
