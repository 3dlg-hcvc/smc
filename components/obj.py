import trimesh
import numpy as np
import numpy.typing as npt
from components.bounding_box import BoundingBox

class Obj:
    def __init__(self, 
                 label: str, 
                 bounding_box: BoundingBox,
                 mesh: trimesh.Trimesh | None,
                 transform_matrix: npt.ArrayLike | None,
                 matrix_order: str = "F") -> None:
        '''
        Initialize an object.

        Args:
            label: string, the label of the object
            bounding_box: BoundingBox, the bounding box of the object
            mesh: Trimesh, the mesh of the object
            transform_matrix: 4x4 matrix, the transform matrix of the object
            matrix_order: string, the order for reshaping the transform_matrix matrix into a 4x4 matrix
        
        Returns:
            None
        '''

        self.label = label
        self.bounding_box = bounding_box
        self.mesh = mesh
        if transform_matrix is not None:
            self.transform_matrix = np.array(transform_matrix).astype(float).reshape(4, 4, order=matrix_order)
        else:
            self.transform_matrix = None

    @property
    def has_mesh(self) -> bool:
        '''
        Check if the object has a mesh.
        
        Returns:
            bool, whether the object has a mesh
        '''

        return self.mesh is not None
    