from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic

app_name = 'homepage'

def home(request):
    return render(request, 'homepage/homepage.html')

