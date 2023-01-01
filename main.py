# Importing Necessary modules
from app.core.utils.utils import *
# from colabcode import ColabCode
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from app.core.api import api


# Declaring our FastAPI instance
app = FastAPI()


app.mount("/static", StaticFiles(directory="app/templates/static"), name="static")
app.mount("/staging", StaticFiles(directory="app/staging"), name="staging")


# @app.on_event('startup')
# async def download_model_wandb():
#     download_model()

@app.on_event("shutdown")
def shutdown_event():
    files = glob.glob('./app/staging/video/*.mp4')
    for file in files:
        print(file)
        os.remove(file)


app.include_router(api.router)

# cc = ColabCode(port=12000, code=False)
# cc.run_app(app=app)
