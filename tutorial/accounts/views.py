from django.http import Http404
from .models import Album
from django.shortcuts import render

def index(request):
    all_albums = Album.objects.all()
    return render(request, 'accounts/index.html', {'all_albums': all_albums})

def detail(request, album_id):
    try:
        album = Album.objects.get(id=album_id)
    except Album.DoesNotExist:
        raise Http404("This album doesn't exist!")
    return render(request, 'accounts/detail.html', {'album': album})


