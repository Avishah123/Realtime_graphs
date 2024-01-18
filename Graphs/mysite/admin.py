from django.contrib import admin
from .models import *
from import_export.admin import ImportExportModelAdmin

# Register your models here.
class BlogAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    pass

admin.site.register(OptionGreeks, BlogAdmin) 

# admin.site.register(OptionGreeks)
