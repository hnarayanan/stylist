from django.conf.urls import include, url

from .views import PhotoCreate, PhotoDetail


urlpatterns = [
    url(r'^upload/$', PhotoCreate.as_view(), name='upload'),
    url(r'^(?P<pk>[0-9]+)/$', PhotoDetail.as_view(), name='detail'),
]
