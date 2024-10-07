from django.urls import path
from .views import index_page, predict_category

urlpatterns = [
    # URL for rendering the HTML form
    path('', index_page, name='index_page'),

    # URL for the API to handle predictions
    path('predict/', predict_category, name='predict_category'),
]
