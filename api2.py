from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import io
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/files")
async def process_files(files: list[UploadFile] = File(...)):
    list_img = []

    for file in files:
        content = await file.read()
        stream = io.BytesIO(content)
        nparray = np.asarray(bytearray(stream.read()), dtype="uint8")
        img = cv.cvtColor(cv.imdecode(nparray, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        list_img.append(img)
        cv.imwrite("static/123.png", img)


    content = """
    <body>
        <form action="/files" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
        </form>
        <img src="static/123.png">
    </body>
    """

    return HTMLResponse(content)



@app.get("/")
async def main():
    content = """
    <body>
        <form action="/files" enctype="multipart/form-data" method="post">
        <input name="files", type="file" multiple>
        <input type="submit">
    <body/>
    """
    return HTMLResponse(content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, host="localhost")