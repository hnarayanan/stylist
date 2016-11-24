from django.conf import settings
from django.conf.urls import include, url
from django.conf.urls.static import static
from django.contrib import admin

from photos.views import PhotoList


urlpatterns = [
    url(r'^$', PhotoList.as_view(), name='home'),
    url(r'^photos/', include('photos.urls', namespace='photo')),
    url(r'^admin/', admin.site.urls),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
