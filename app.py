# Importing Necessary modules
from tempfile import NamedTemporaryFile
from deployment.utils import *
from deployment.request_body import *
# from colabcode import ColabCode
import os
import traceback
import argparse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request, status

# Declaring our FastAPI instance
app = FastAPI()

templates = Jinja2Templates(directory="deployment/templates")
app.mount("/static", StaticFiles(directory="deployment/templates/static"), name="static")
app.mount("/staging", StaticFiles(directory="deployment/staging"), name="staging")

predict_label = None
softmax_res = None
ground_true = None

# @app.on_event('startup')
# async def download_model_wandb():
#     download_model()


@app.get("/")
async def main(request: Request):
    if predict_label is not None:
        return templates.TemplateResponse("predict_home.html", {"request": request, "res": predict_label[['label', 'softmax']].values.tolist(), "heading": ['Label', 'Prediction'], "ground_true": ground_true})
    else:
        return templates.TemplateResponse("predict_home.html", {"request": request, "res": [[None, 100]], "heading": ['Label', 'Prediction'], "ground_true": 'None'})


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('./deployment/templates/static/img/favicon.ico')


@app.post("/predict/")
async def predict_action(model_name: str = Form(default='lrcn'), file: UploadFile = File()):
    global predict_label, softmax_res, ground_true

    ground_true = file.filename.split('_')[1]
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

    predict_label = res.detach().numpy().flatten().tolist()
    predict_label = LABEL[LABEL.label_id.isin(predict_label)][['label']]
    predict_label.index = pd.RangeIndex(start=1, stop=11, step=1)

    softmax_res = [round(i * 100, 3) for i in sftm.detach().numpy().flatten().tolist()]
    predict_label['softmax'] = softmax_res

    # write_result_to_video(f"Predict: {predict_label.loc[1]['label']}")
    return RedirectResponse('/', status_code=status.HTTP_303_SEE_OTHER)


# cc = ColabCode(port=12000, code=False)
# cc.run_app(app=app)
