from django.contrib import admin

from .models import Photo, Style


def highlight(modeladmin, request, queryset):
    queryset.update(is_highlighted=True)
highlight.short_description = "Highlight selected photos"


class PhotoAdmin(admin.ModelAdmin):

    list_display = ('title', 'style', 'image', 'is_highlighted')
    list_filter = ('style', 'is_highlighted')
    search_fields = ['title']
    actions = [highlight]


class StyleAdmin(admin.ModelAdmin):

    list_display = ('title', 'image')
    search_fields = ['title']


admin.site.register(Photo, PhotoAdmin)
admin.site.register(Style, StyleAdmin)
