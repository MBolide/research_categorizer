from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import status
from .serializers import AbstractSerializer
from .utils import predict_categories
from rest_framework.decorators import api_view
from django.http import HttpResponseNotFound


    
@api_view(['GET','POST'])
def predict_category(request):
    if request.method == "POST":
        if isinstance(request.data, str):
            data = {"abstract": request.data}
        else:
            data = request.data

        # Use the serializer to validate the input
        serializer = AbstractSerializer(data=data)

        if serializer.is_valid():
            abstract = serializer.validated_data['abstract']
            # Call to the
            predicted_categories = predict_categories(abstract)
            return Response(
                {"abstract": abstract, "predicted_categories": predicted_categories}, 
                status=status.HTTP_200_OK
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return HttpResponseNotFound('This page does not exist.')
    
def index_page(request):
    return render(request, 'index.html')
