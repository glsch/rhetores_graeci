from typing import List


from lightning.pytorch import LightningDataModule
from
class RhetoresDataModule(LightningDataModule):
    def __init__(self,
                 epithets: List[str]
                 ):
        super().__init__()