"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""

import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np

import torch
from flask import Flask, render_template, request, redirect
from flask_images import resized_img_src
from functions import main

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return 
        # read in image file
        img_bytes = file.read()
        # opem image file
        image = Image.open(io.BytesIO(img_bytes))
        # calculate bounding boxes
        results = model(image, size=640)
        # calculate image with bounding boxes and confidence level
        # updates results.imgs with boxes and labels
        results.render()
        # save the image
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")           
        
        # label image with bounding boxes and identify missing teeth
        
        # convert PIL image to cv2 and make a copy
        cv_image = np.array(image) 
        cv_image = cv_image[:, :, ::-1].copy()
        
        bounding_boxes = results.pandas().xyxy[0]
        bounding_boxes.drop(columns=['confidence', 'name'], inplace=True)
        # run missing tooth function
        main(bounding_boxes, cv_image)       
        return render_template("image.html", 
                               user_image1 = "static/image0.jpg",
                               user_image2 = "static/image1.jpg")

    return render_template("index.html")

@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'custom', 
                           path='weights/initial_tooth_segmentation_weight.pt', 
                           force_reload=True).autoshape()
    model.eval()
    app.run(host="localhost", port=8013, debug=True)  # debug=True causes Restarting with stat