#!/bin/bash
python --version
pip install --upgrade azure-cli
pip install --upgrade azureml-sdk azure-mgmt-resource==18.0.0
pip install -r requirements.txt
