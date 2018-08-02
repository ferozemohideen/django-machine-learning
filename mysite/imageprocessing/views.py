from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.conf import settings
base_dir = settings.BASE_DIR

from .models import Choice, Question
from .testKNN import knn


class IndexView(generic.ListView):
    template_name = 'imageprocessing/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.order_by('-pub_date')[:5]


class DetailView(generic.DetailView):
    model = Question
    template_name = 'imageprocessing/Fred.html'


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'imageprocessing/results.html'


def classification(request):
    return render(request, 'imageprocessing/glasses/indexdetail.html')

def fred(request):
    context = {}
    if request.GET:
        numneighbors = request.GET['numneighbors']
        # get return values
        context = knn(name=1, neighbors=numneighbors, dir=base_dir)
        print(context)
    return render(request, 'imageprocessing/glasses/Fred.html', {'context': context})

def johannes(request):
    context = {}
    if request.GET:
        numneighbors = request.GET['numneighbors']
        # get return values
        context = knn(name=2, neighbors=numneighbors, dir=base_dir)
        print(context)
    return render(request, 'imageprocessing/glasses/Johannes.html', {'context': context})

def jimothy(request):
    context = {}
    if request.GET:
        numneighbors = request.GET['numneighbors']
        # get return values
        context = knn(name=3, neighbors=numneighbors, dir=base_dir)
        print(context)
    return render(request, 'imageprocessing/glasses/Jimothy.html', {'context': context})

def lana(request):
    context = {}
    if request.GET:
        numneighbors = request.GET['numneighbors']
        # get return values
        context = knn(name=4, neighbors=numneighbors, dir=base_dir)
        print(context)
    return render(request, 'imageprocessing/glasses/Lana.html', {'context': context})


