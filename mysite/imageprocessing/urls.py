from django.urls import path

from . import views

app_name = 'imageprocessing'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
    path('glasses/', views.classification, name='glasses'),
    path('glasses/Frederick/', views.fred, name='fred'),
    ]