# Fetal MRI with Deep Learning
Long description. PyTorch. Matlab.

# Install

Add '*yourdirectory*/flow/flow' to PYTHONPATH environment variable.

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

The flow folder should contain codes and only codes.

```
flow
├───flow                        # All codes and only codes here
│   ├───base
│   │   ├───model.py            # Base class of the model
│   │   └───trainer.py          # Base class of the trainer
│   │
│   ├───data                    # No actual data, just codes
│   │   └───datasetname
│   │       ├───process.py      # Process data
│   │       └───dataset.py      # Class to read data
│   │
│   ├───mains                   # Create experiment folders in here
│   │   └───experimentname
│   │       ├───train.json      # json file should be same name as main file
│   │       └───train.py        # Main file for training
│   │
│   ├───models
│   │   └───neuralnet.py
│   │
│   └───utils
│       ├───config.py           # Class to read json config file
│       └───logger.py           # Python logging
│
│
├───data
│   └───datasetname
│       ├───external            # Data from third party sources
│       ├───interim             # Convenient intermediate form of the raw data
│       ├───processed           # Data ready for training
│       └───raw                 # Original data
│
│
├───experimentname              # Results and saved models separated from codes
│   ├───results
│   └───saved_models
│
│
└───saved_models                # Trained models that can be shared b/w experiments
    └───trained_model           # Folder
```
