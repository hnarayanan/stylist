from django.forms import ModelForm

from .models import Photo


class PhotoUploadForm(ModelForm):

    class Meta:
        model = Photo
        fields = ['image', 'style']
