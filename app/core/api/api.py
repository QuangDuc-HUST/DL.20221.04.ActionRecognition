from fastapi import APIRouter
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse
from fastapi import Request, status, UploadFile, Form, File
from tempfile import NamedTemporaryFile
from app.core.utils.utils import *
from app.core.constants.constants import *
import traceback
import argparse
import os
import glob


router = APIRouter()

templates = Jinja2Templates(directory="app/templates")

predict_label = None
softmax_res = None


@router.get("/")
async def main(request: Request):
    if predict_label is not None:
        return templates.TemplateResponse("predict_home.html", {"request": request, "res": predict_label[['label', 'softmax']].values.tolist(), "heading": ['Label', 'Prediction']})
    else:
        return templates.TemplateResponse("predict_home.html", {"request": request, "res": [[None, 100]], "heading": ['Label', 'Prediction']})


@router.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('./app/templates/static/img/favicon.ico')


@router.post("/predict/")
async def predict_action(model_name: str = Form(default='lrcn'), file: UploadFile = File()):
    global predict_label, softmax_res

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

        elif model_name == 'c3d':
            res, sftm = predict(temp.name, argparse.Namespace(**C3D_ARGS))

        elif model_name == 'i3d':
            res, sftm = predict(temp.name, argparse.Namespace(**I3D_ARGS))

        elif model_name == 'non_local':
            res, sftm = predict(temp.name, argparse.Namespace(**NON_LOCAL_ARGS))

        else:
            res, sftm = predict(temp.name, argparse.Namespace(**LATE_FUSION_ARGS))

        print('Predicted')
    except Exception as e:
        print(traceback.format_exc())
        print("There was an error processing the file")
    finally:
        os.remove(temp.name)

    print(res)

    predict_label = res.detach().numpy().flatten().tolist()
    predict_label = LABEL.merge(pd.DataFrame({'label_id': predict_label, 'softmax': [round(i * 100, 3) for i in sftm.detach().numpy().flatten().tolist()]}), on='label_id', how='inner')
    predict_label = predict_label.sort_values(by="softmax", ascending=False)
    predict_label.index = pd.RangeIndex(start=1, stop=11, step=1)

    print(predict_label)

    return RedirectResponse('/', status_code=status.HTTP_303_SEE_OTHER)
