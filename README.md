# Fetal MRI with Deep Learning
Long description. PyTorch. Matlab.

# Table of Contents

-  [Creating New Projects](#creating-new-projects)
    -  [Define a Model](#define-a-model)
-  [Folder Structure](#folder-structure)

# Creating New Projects

Follow these steps to create a new PyTorch project.
Click [here](#folder-structure) for an explanation of the structure of the repository.

## Define a Model

-  Define a new class and inherit from BaseModel.
BaseModel already implements methods .save() and .load().

```python
from flow.base.model import BaseModel

class UNet(BaseModel):
    def __init__(self):
        super(UNet, self).__init__()
        # Define PyTorch layers

    def forward(self, x):
        # Define forward pass
```

# Folder Structure

```
├───base
│   ├───model.py            # Base class of the model
│   └───trainer.py          # Base class of the trainer
│
├───data                    # Data that can be shared between experiments
│   ├───DatasetName
│   │   ├───external        # Data from third party sources
│   │   ├───interim         # Convenient form of the raw data
│   │   ├───processed       # Data ready for training
│   │   └───raw             # Original data
│   │
│   └───datasetname.py      # Class to read data
│
├───mains                   # Create experiment folders in here
│   └───experimentname
│       ├───train.json      # json file same name as main file
│       └───train.py
│
├───models                  # Models that can be shared between experiments
│   └───neuralnet.py
│
├───saved_models            # Pre-trained or trained models
│   └───trained_model.pth
│
└───utils
    ├───config.py           # Class to read json configurations file
    └───logger.py           # Python logging
```
