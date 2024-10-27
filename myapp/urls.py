from django.urls import path
# from .views import chatbotResponse
from .views import chatbotResponse
urlpatterns = [
    path('req/',chatbotResponse,name='req'),
]