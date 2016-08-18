from django.views.generic import CreateView, DetailView, ListView

from .models import Photo


class PhotoCreate(CreateView):

    model = Photo
    fields = ['image', 'title', 'style']


class PhotoDetail(DetailView):

    model = Photo
    context_object_name = 'photo'


class PhotoList(ListView):

    model = Photo
    context_object_name = 'photos'


