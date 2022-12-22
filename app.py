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
    latent_dim: int = Form(default=512),
    hidden_size: int = Form(default=256),
    lstm_layers: int = Form(default=2),
    bidirectional: bool = Form(default=True),
    file: UploadFile= File()
    ):

    res = None
    agrs = {
        'model_name': model_name,
        'latent_dim': latent_dim,
        'hidden_size': hidden_size,
        'lstm_layers': lstm_layers,
        'bidirectional': bidirectional,
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

        res = predict(temp.name, file.filename, agrs)
    except Exception as e:
        print(traceback.format_exc())
        print("There was an error processing the file")
    finally:
        print("Reading completed")
        os.remove(temp.name)
    
    if res is not None:
        return {"Prediction": ', '.join([str(i) for i in res.numpy()])}
    else:
        return {"Prediction": "None"}

@app.post("/c3d/")
async def c3d_upload_file(
    model_name: str = Form(default='c3d'),
    drop_out: float = Form(default=0.5),
    pretrain: bool = Form(default=False),
    weight_path: str = Form(default='c3d.pickle'),
    file: UploadFile= File()
    ):

    res = None
    agrs = {
        'model_name': model_name,
        'drop_out': drop_out,
        'pretrain': pretrain,
        'weight_path': weight_path,
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
        
        print('Predicting..')
        res = predict(temp.name, file.filename, agrs)
        print('Predicted')
    except Exception as e:
        print(traceback.format_exc())
        print("There was an error processing the file")
    finally:
        print("Reading completed")
        os.remove(temp.name)

    return {"filename": res}

# cc = ColabCode(port=12000, code=False)
# cc.run_app(app=app)
