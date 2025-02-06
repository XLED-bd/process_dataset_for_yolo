from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import io
import numpy as np
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

import cv2 as cv

app = FastAPI()

@app.post("/file")
async def create_files(file: bytes = File(...)):
    stream = io.BytesIO(file)
    
    nparr = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    
    cv.imwrite("123.png", image)
    
    output = FileResponse(path="123.png")

    return output



@app.post("/files")
async def create_files(files: list[UploadFile] = File(...)):
    processed_images = []

    for file in files:
        contents = await file.read()
        stream = io.BytesIO(contents)
        nparr = np.asarray(bytearray(stream.read()), dtype="uint8")
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)

        # Преобразования с изображением (например, конвертация в grayscale)
        # Здесь можно добавить любые преобразования, которые вам нужны
        processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        processed_images.append(processed_image)

    # Создание фигуры для отображения изображений
    fig, axes = plt.subplots(1, len(processed_images), figsize=(15, 5))
    if len(processed_images) == 1:
        axes = [axes]  # Чтобы можно было итерироваться даже по одному изображению

    for ax, img in zip(axes, processed_images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # Скрыть оси

    # Сохранение фигуры в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Возвращение изображения в ответе
    return StreamingResponse(buf, media_type="image/png")

@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</body>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)