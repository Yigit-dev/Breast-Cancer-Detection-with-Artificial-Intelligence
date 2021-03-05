from django.db import models

# Create your models here.
class Cancer(models.Model):
    # id vermeye gerek yok django bunu yapıyor
    STATUS = (
        ('True', 'Evet'),
        ('False', 'Hayır'),
    ) # status combobox tan beslenecek
    # Hasta Bilgileri
    tc        = models.CharField(max_length=12, verbose_name= "TC Kimliği")
    firstName = models.CharField(max_length=55, verbose_name= "Ad")
    lastName  = models.CharField(max_length=55, verbose_name= "Soyad")
    length    = models.CharField(max_length=3, verbose_name= "Boy")
    age       = models.CharField(max_length=3, verbose_name= "Yaş")
    sex       = models.CharField(max_length=20, verbose_name= "Cinsiyet")
    city      = models.CharField(max_length=50, verbose_name="Şehir")
    country   = models.CharField(max_length=12, verbose_name="Ülke")
    # ****Hasta Sonuçları
    # Hücre Boyutunun Düzgünlüğü
    uniformity_cell_size = models.IntegerField()
    # Hücre Şeklinin Düzgünlüğü
    uniformity_cell_shape = models.IntegerField()
    # Marjinal Yapışma
    marginal_adhesion = models.IntegerField()
    # Tek Epitel Hücre Boyutu
    single_epithelial_cell_size = models.IntegerField()
    # Çıplak Çekirdekler
    bare_nuclei = models.IntegerField()
    # Uyumlu Kromatin
    bland_chromatin = models.IntegerField()
    # Normal Nikloeller
    normal_nucleoli = models.IntegerField()
    # Mitoz
    mitoses = models.IntegerField()
    # 2: iyi huylu , 4: kötü huylu
    result = models.IntegerField()
    # Açıklamalar
    status = models.CharField(max_length=10, choices=STATUS)
    create_at = models.DateTimeField(auto_now_add=True)
    update_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.firstName