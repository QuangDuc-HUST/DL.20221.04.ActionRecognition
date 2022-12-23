# Importing Necessary modules
from tempfile import NamedTemporaryFile
from deployment.utils import *
from deployment.request_body import *
# from colabcode import ColabCode
import os
import traceback
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request

# Declaring our FastAPI instance
app = FastAPI()

templates = Jinja2Templates(directory="deployment/templates")
app.mount("/static", StaticFiles(directory="deployment/templates/static"), name="static")

# @app.on_event('startup')
# async def download_model_wandb():
#     download_model()

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('./deployment/templates/static/img/favicon.ico')

@app.get('/', response_class=HTMLResponse)
# @app.get('/')
def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
    
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
                f.write(contents)
        except Exception as e:
            print(traceback.format_exc())
            print("There was an error uploading the file")
        finally:
            file.file.close()
        print("Reading completed")
        convert_avi_to_mp4(temp.name)
        print('Predicting..')
        if model_name == 'lrcn':
            res, sftm = predict(temp.name, argparse.Namespace(**LRCN_ARGS))
        else:
            res, sftm = predict(temp.name, argparse.Namespace(**C3D_ARGS))
        print('Predicted')
    except Exception as e:
        print(traceback.format_exc())
        print("There was an error processing the file")
    finally:
        os.remove(temp.name)
    if res is not None:
        return {"Prediction": ', '.join([str(i) for i in res.detach().numpy()]),
                "Softmax": ', '.join([str(i) for i in sftm.detach().numpy()])}
    else:
        return {"Prediction": "None"}


# cc = ColabCode(port=12000, code=False)
# cc.run_app(app=app)
