from __future__ import unicode_literals

from django.db import models
from django.urls import reverse

class Style(models.Model):

    image = models.ImageField(upload_to='styles/%Y/%m/%d/')
    title = models.CharField(max_length=100)

    def __unicode__(self):
        return self.title


class Photo(models.Model):

    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    title = models.CharField(max_length=100, null=True, blank=True)
    style = models.ForeignKey(Style, on_delete=models.PROTECT)
    is_highlighted = models.BooleanField(default=False)

    class Meta:
        ordering = ['-id']

    def __unicode__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('photo:detail', kwargs={'pk': self.id})
