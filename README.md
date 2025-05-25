# YOLO detection based kitchen violation 
There are two models that run together. 
First model checks for all the violations, second model checks if the person based violations are within bounds of a person. 
Currently only RKNN models are used. 

# Installation
*   Create and activate venv
``` 
python -m  venv venv
source venv/bin/activate
```
*   Install requirements
```
pip install -r requirements.txt
```
*   Run
```
python kitchen_safety.py
```
