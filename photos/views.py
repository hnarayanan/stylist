from django.views.generic import DetailView, ListView
from django.views.generic.edit import FormView
from django.core.urlresolvers import reverse

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

    def form_valid(self, form):
        photo = Photo(
            image=self.get_form_kwargs().get('files')['image'],
            title=self.get_form_kwargs().get('data')['title'],
            style=self.get_form_kwargs().get('data')['style'],
        )
        photo.save()
        self.photo_id = photo.id
        return super(PhotoUpload, self).form_valid(self)

    def get_success_url(self):
        return reverse('photo:detail', kwargs={'pk': self.photo_id})
