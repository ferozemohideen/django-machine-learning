from django.urls import path

from . import views

app_name = 'imageprocessing'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    path('glasses/', views.classification, name='glasses'),
    path('glasses/Frederick/', views.fred, name='fred'),
    path('glasses/Johannes/', views.johannes, name='johannes'),
    ]