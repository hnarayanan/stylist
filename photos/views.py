from django.views.generic import DetailView, ListView
from django.views.generic.edit import FormView

from .models import Photo
from .forms import PhotoUploadForm


class PhotoList(ListView):

    model = Photo
    context_object_name = 'photos'


class PhotoDetail(DetailView):

    model = Photo
    context_object_name = 'photo'


class PhotoUpload(FormView):

    template_name = 'photos/photo_upload.html'
    form_class = PhotoUploadForm
    success_url = '/admin/'

    def form_valid(self, form):
        # TODO: Do some stuff with the valid form data
        # that's been posted.
        photo = Photo(
            image=self.get_form_kwargs().get('files')['image'],
            style=self.get_form_kwargs().get('data')['style'],
        )
        return super(PhotoUploadView, self).form_valid(self)
