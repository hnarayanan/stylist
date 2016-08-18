from __future__ import unicode_literals

from django.db import models
from django.urls import reverse


class Photo(models.Model):

    STYLES = (
        ('ML',  'Mona Lisa'),
        ('TSN', 'The Starry Night'),
        ('TS',  'The Scream'),
        ('GPE', 'Girl with a Pearl Earring'),
    )

    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    # TODO: Perhaps needs two fields, one for the original
    # and one for the transformed image.
    title = models.CharField(max_length=100, blank=True)
    style = models.CharField(max_length=5, choices=STYLES)

    class Meta:
        ordering = ['-id']

    def __unicode__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('photo:detail', kwargs={'pk': self.id})
