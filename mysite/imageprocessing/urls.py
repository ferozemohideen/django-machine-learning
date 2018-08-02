from django.urls import path

from . import views

app_name = 'imageprocessing'
urlpatterns = [
    path('', views.classification, name='glasses'),
    path('Frederick/', views.fred, name='fred'),
    path('Johannes/', views.johannes, name='johannes'),
    path('Jimothy/', views.jimothy, name='jimothy'),
    path('Lana/', views.lana, name='lana'),
    ]