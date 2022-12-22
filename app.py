# Importing Necessary modules
from tempfile import NamedTemporaryFile
from deployment.utils import *
from deployment.request_body import *
# from colabcode import ColabCode
import os
import traceback

# Declaring our FastAPI instance
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Go to /docs'}
    
@app.post("/predict/")
async def predict_action(
    model_name: str = Form(default='lrcn'),
    file: UploadFile= File()
    ):

    res, sftm = None, None
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
        print("Reading completed")

        print('Predicting..')
        if model_name == 'lrcn':
            res, sftm = predict(temp.name, LRCN_ARGS)
        else:
            res, sftm = predict(temp.name, C3D_ARGS)
        print('Predicted')
    except Exception as e:
        print(traceback.format_exc())
        print("There was an error processing the file")
    finally:
        os.remove(temp.name)
    if res is not None:
        return {"Prediction": ', '.join([str(i) for i in res.numpy()]),
                "Softmax": ', '.join([str(i) for i in sftm.detach().numpy()])}
    else:
        return {"Prediction": "None"}


# cc = ColabCode(port=12000, code=False)
# cc.run_app(app=app)
