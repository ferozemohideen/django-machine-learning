from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    # /music/
    path('', views.index, name='index'),

    # /music/71/
    path('<int:album_id>/', views.detail, name='detail'),

    # /music/<album_id>/favorite
    path('<int:album_id>/favorite/', views.favorite, name='favorite'),
]
