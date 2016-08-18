from django.conf.urls import include, url

from .views import PhotoList, PhotoDetail, PhotoUpload


urlpatterns = [
    url(r'^$', PhotoList.as_view(), name='list'),
    url(r'^(?P<pk>[0-9]+)/$', PhotoDetail.as_view(), name='detail'),
    url(r'^upload/$', PhotoUpload.as_view(), name='upload'),
]
