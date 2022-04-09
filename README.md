# Sentiment Classifier

This sentiment classifier was built by Simone Beckmann and Julia Evans for the class Team Lab at the IMS, University of Stuttgart.

## Packages

This classifier requires the following packages:
- re
- csv
- math

## Usage

The classifier can be accessed through the basic_model.py file.

The training data file is specified towards the bottom of the basic_model.py file in the initialization of the MCP (multi-class perceptron).


```python
# initialize MCP with training data  
MCP_Basic = MCP("isear-train.csv") 
```

After creating the model, test data for generating predictions is specified.


```python
our_eval_data = get_predictions("isear-test.csv")
```

