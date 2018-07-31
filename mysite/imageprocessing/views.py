from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic

from .models import Choice, Question
# from .testKNN import knn


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

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/Fred.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('imageprocessing:results', args=(question.id,)))

def classification(request):
    return render(request, 'imageprocessing/glasses/indexdetail.html')

def fred(request):
    context = {}
    if request.GET:
        numneighbors = request.GET['numneighbors']
        # get return values
    return render(request, 'imageprocessing/glasses/Fred.html', context)


