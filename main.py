

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
#from starlette.responses import FileResponse

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()

#conect to css file

app.mount("/static", StaticFiles(directory="static"), name="static")

#get to templare
templates = Jinja2Templates(directory="templates")


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("Model.h5")

CLASS_NAMES = [ "Normal", "Paition"]

@app.get("/ping")
async def ping():
    #return FileResponse('./templates/index.html')
    return templates.TemplateResponse('index.html')

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    #img_batch = np.expand_dims(image, 0)

    #img=cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(gray, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    test_img=cv2.resize(thresh,(224,224))
    test_img=test_img/255
    test_img=test_img.reshape(1,224,224,1)   


    predictions = MODEL.predict(test_img)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return templates.TemplateResponse("predict.html", {"request": request, "predicted_class": predicted_class, "confidence":confidence})

    '''
    {"request": request, "disease": predicted_class, "predict_value":confidence}

    web=FileResponse('./templates/predict.html')
    return (web , {
    'class': predicted_class,
    'confidence': float(confidence)
    })
    return {
    'class': predicted_class,
    'confidence': float(confidence)
    }'''

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
