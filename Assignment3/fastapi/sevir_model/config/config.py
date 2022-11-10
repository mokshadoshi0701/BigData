from pathlib import Path
from pydantic import BaseModel

# import sevir_model

class ModelConfig(BaseModel): 
    """
    All configuration relevant to model 
    training and feature engineering
    """
    features: list
    column_names: list
    target: str
    buying: str 
    maint: str 
    doors: str
    persons: str
    lug_boot: str
    safety: str
    buying_and_maint: list
    buying_and_maint_mappings: dict
    doors_mappings: dict
    persons_mappings: dict
    lug_boot_mappings: dict
    safety_mappings: dict
    class_mappings: dict
    random_state: int
    test_size: float
