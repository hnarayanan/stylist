from django.contrib import admin

from .models import Photo


class PhotoAdmin(admin.ModelAdmin):

    list_display = ('title', 'style', 'image')
    list_filter = ('style',)
    search_fields = ['title']


admin.site.register(Photo, PhotoAdmin)
