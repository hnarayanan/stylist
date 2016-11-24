from django.views.generic import CreateView, DetailView, ListView

from .models import Photo, Style


class PhotoCreate(CreateView):

    model = Photo
    fields = ['image', 'style']

    def get_context_data(self, **kwargs):
        context = super(PhotoCreate, self).get_context_data(**kwargs)
        context['styles'] = Style.objects.all()
        return context


class PhotoDetail(DetailView):

    model = Photo
    context_object_name = 'photo'


class PhotoList(ListView):

    model = Photo
    context_object_name = 'photos'
    queryset = Photo.objects.filter(is_highlighted=True)
