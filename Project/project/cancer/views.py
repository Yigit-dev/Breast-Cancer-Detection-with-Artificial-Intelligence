from django.shortcuts import render,HttpResponse,redirect,get_object_or_404
from .models import Cancer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# Create your views here.

def index(request):
  return render(request, "index.html")

def features(request):
  return render(request, "features.html")

def price(request):
  return render(request, "pricing.html")

def blog(request):
  return render(request, "blog.html")

def contact(request):
  return render(request, "contact.html")

def cancer(request):
  cancers = Cancer.objects.all()

  return render(request, "test.html",{"cancers":cancers})

def addCancer(request):

  if request.method == "GET":
    return redirect("/cancer")
  else:
    # Hasta Kişisel Bilgileri
    tc = request.POST.get("tc")
    firstName = request.POST.get("firstName")
    lastName = request.POST.get("lastName")
    length = request.POST.get("length")
    age = request.POST.get("age")
    sex = request.POST.get("sex")
    city = request.POST.get("city")
    country = request.POST.get("country")
    
    # ************Yapay Zeka ****************
    veri = pd.read_csv("datasets/breast-cancer.data")
    veri.replace('?', -99999, inplace=True)
    veriyeni = veri.drop(['1000025'], axis=1)
    imp = SimpleImputer(missing_values=-99999, strategy="mean", fill_value=None, verbose=0, copy=True)
    veriyeni = imp.fit_transform(veriyeni) # sklearn

    # 8 adet özelliğe bağlı bir giriş katmanımız var 
    # (Hücre Boyutunun Düzgünlüğü, Hücre Şeklinin Düzgünlüğü, Marjinal Yapışma, Tek Epitel Hücre Boyutu,
    # Çıplak Çekirdekler, Uyumlu Kromatin, Normal Nikloeller, Normal Nikloeller, Mitoz)

    giris = veriyeni[:, 0:8] 
    cikis = veriyeni[:, 9]  

    # ** VERİ SETİ İŞLEMLERİ TAMAM ŞİMDİ MODELİMİZİ OLUŞTURALIM

    model = Sequential() # Yapay sinir ağları algılayıcıların ardışık olmasına bağlıdır

    model.add(Dense(10, input_dim=8))
    # NEDEN RELU ?
    # Matrislerde sürekli y = mx + b işlemi çalışacağı için çok yüksek değerler elde ediyoruz biz bunu belli bir değer arasına sokmamız lazım
    # bunun için aktivasyon fonksiyonu kullanıyoruz verilerimizi 0 ile 1 arasına sokuyoruz 

    model.add(Activation('relu'))  # model.add(Activation('tanh')) daha hızlı sonuç verdi
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))  
    
    optimizer = keras.optimizers.SGD(lr=0.01)
    # metrics = accuracy yapıyoruz 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 2: iyi huylu tümör, 4: kötü huylu tümör
  
    model.fit(giris, cikis, epochs=10, batch_size=32, validation_split=0.20)

    # inputs
    uniformity_cell_size = request.POST.get("uniformity_cell_size")
    uniformity_cell_shape = request.POST.get("uniformity_cell_shape")
    marginal_adhesion = request.POST.get("marginal_adhesion")
    single_epithelial_cell_size = request.POST.get("single_epithelial_cell_size")
    bare_nuclei = request.POST.get("bare_nuclei")
    bland_chromatin = request.POST.get("bland_chromatin")
    normal_nucleoli = request.POST.get("normal_nucleoli")
    mitoses = request.POST.get("mitoses")

    tahmin = np.array(
      [
        uniformity_cell_size,
        uniformity_cell_shape,
        marginal_adhesion,
        single_epithelial_cell_size,
        bare_nuclei,
        bland_chromatin,
        normal_nucleoli,
        mitoses
      ]
    ).reshape(1, 8)
    print(model.predict_classes(tahmin))

    result = model.predict_classes(tahmin)

    newCancer = Cancer(
      tc=tc,
      firstName=firstName,
      lastName=lastName,
      length=length,
      age=age,
      sex=sex,
      city=city,
      country=country,
      uniformity_cell_size=uniformity_cell_size,
      uniformity_cell_shape=uniformity_cell_shape,
      marginal_adhesion=marginal_adhesion,
      single_epithelial_cell_size=single_epithelial_cell_size,
      bare_nuclei=bare_nuclei,
      bland_chromatin=bland_chromatin,
      normal_nucleoli=normal_nucleoli,
      mitoses=mitoses,
      result = result
    )

    newCancer.save()

    return redirect("/cancer")

# Kayıt Silme
def deleteResult(request,id):
  cancer= get_object_or_404(Cancer, id = id)
  cancer.delete()
  return redirect("/cancer")