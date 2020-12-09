import pickle
import json
import numpy as np
import pandas as pd 
import time

from sklearn.linear_model import Ridge
from joblib import load

from azureml.core.model import Model
#from azureml.monitoring import ModelDataCollector

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

def init():
    global model
    
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))
    
    model_path = Model.get_model_path(model_name = 'diabetes_model')
    model = load(model_path)

input_sample = NumpyParameterType(np.array([[1,2,3,4,54,6,7,8,88,10], [10,9,8,37,36,45,4,33,2,1]]))
output_sample = PandasParameterType(pd.DataFrame({"result": [27791.59951581891, 10958.615160340678] }))

@input_schema('data', input_sample)
@output_schema(output_sample)
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return json.dumps({"error": error})
