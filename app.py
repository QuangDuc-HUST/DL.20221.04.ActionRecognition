# Importing Necessary modules
from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile
from pydantic import BaseModel
from deployment.model import *
# from colabcode import ColabCode
import os
import traceback



# Declaring our FastAPI instance
app = FastAPI()

class request_body(BaseModel):
    description: str

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}
    
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile, model_name: str):

    temp = NamedTemporaryFile(delete=False)
    try:
        print('Reading video..')
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception as e:
            print(traceback.format_exc())
            print("There was an error uploading the file")
        finally:
            file.file.close()

        res = predict(temp.name, file.filename, model_name)
    except Exception as e:
        print(traceback.format_exc())
        print("There was an error processing the file")
    finally:
        print("Reading completed")
        os.remove(temp.name)

    return {"filename": res}

# cc = ColabCode(port=12000, code=False)
# cc.run_app(app=app)
