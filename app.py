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
    
@app.post("/lrcn/")
async def lrcn_upload_file(
    model_name: str = Form(default='lrcn'),
    file: UploadFile= File()
    ):

    res = None
    agrs = {
        'model_name': model_name,
        'latent_dim': 512,
        'hidden_size': 256,
        'lstm_layers': 2,
        'bidirectional': True,
    }
    print(agrs)

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
        res = predict(temp.name, file.filename, agrs)
        print('Predicted')
    except Exception as e:
        print(traceback.format_exc())
        print("There was an error processing the file")
    finally:
        os.remove(temp.name)
    
    if res is not None:
        return {"Prediction": ', '.join([str(i) for i in res.numpy()])}
    else:
        return {"Prediction": "None"}

@app.post("/c3d/")
async def c3d_upload_file(
    model_name: str = Form(default='c3d'),
    file: UploadFile= File()
    ):

    res, sftm = None, None
    agrs = {
        'model_name': model_name,
        'drop_out': .5,
        'pretrain': False,
        'weight_path': 'c3d.pickle',
    }
    print(agrs)

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
        res, sftm = predict(temp.name, file.filename, agrs)
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
