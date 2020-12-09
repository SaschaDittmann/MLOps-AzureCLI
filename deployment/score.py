import pickle
import json
import numpy 
import time

from sklearn.linear_model import Ridge
from joblib import load

from azureml.core.model import Model
#from azureml.monitoring import ModelDataCollector

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

def init():
    global model

    print ("model initialized" + time.strftime("%H:%M:%S"))
    model_path = Model.get_model_path(model_name = 'diabetes_model')
    model = load(model_path)
    
input_sample = { "data": [ [1,2,3,4,54,6,7,8,88,10], [10,9,8,37,36,45,4,33,2,1] ] }
output_sample = { "result": [27791.59951581891, 10958.615160340678] }

@input_schema('raw_data', StandardPythonParameterType(input_sample))
@output_schema(StandardPythonParameterType(output_sample))
def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
