# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Declaring our FastAPI instance
app = FastAPI()

class request_body(BaseModel):
    description: str
    
@app.post('/predict-job')
def predict(data : request_body):
    return {'class' : None}
