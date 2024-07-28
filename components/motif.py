from components.obj import Obj

class Motif:
    def __init__(self, objs: list[Obj], description: str) -> None:
        '''
        Initialize a motif.

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
    