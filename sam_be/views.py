from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torch


class SamView(APIView):
    def get(self, request):
        return Response("GET: Hello World!")

    def post(self, request):
        print(1)
        blob = request.FILES.get('file')
        print(2)
        img_data = blob.read()
        print(3)
        np_arr = np.frombuffer(img_data, np.uint8)
        print(4)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print(5)
        # Load predictor
        checkpoint = "sam_vit_h_4b8939.pth"
        print(6)
        model_type = "vit_h"
        print(7)
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        print(8)
        sam.to(device='cuda')
        print(9)
        predictor = SamPredictor(sam)
        print(10)
        predictor.set_image(img)
        print(11)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        print(12)
        return Response(image_embedding)
