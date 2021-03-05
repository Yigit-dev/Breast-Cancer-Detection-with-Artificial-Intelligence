from django.contrib import admin

# Register your models here.
from .models import Cancer
class CancerAdmin(admin.ModelAdmin): #categoriadmin de yapacağım ayarlar
  list_display = ['uniformity_cell_size', 
                  'uniformity_cell_shape',
                  'marginal_adhesion',
                  'single_epithelial_cell_size',
                  'bare_nuclei',
                  'bland_chromatin',
                  'normal_nucleoli',
                  'mitoses',
                  'result'] # hangi alanları görmek istiyorum ?
  list_filter = ['result']

admin.site.register(Cancer,CancerAdmin) # İlişkilendir