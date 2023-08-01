from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor


class SamView(APIView):
    def get(self, request):
        return Response("GET: Hello World!")

    def post(self, request):
        blob = request.FILES.get('file')
        img_data = blob.read()
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Load predictor
        checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device='cuda')
        predictor = SamPredictor(sam)
        predictor.set_image(img)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        return Response(image_embedding)
