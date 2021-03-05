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
# from .myform import MyForm
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
  # f = MyForm()
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
    # ************Yapay Zeka Alanı****************
    # verilerimizi okuyup değişkenimizin içine atıyorum
    veri = pd.read_csv("datasets/breast-cancer.data")
    # verisetimizde bulunan "?" yani bilinmeyen kısımları hesaplanmaması için -99999 gibi bir değer veriyorum
    veri.replace('?', -99999, inplace=True)
    veriyeni = veri.drop(['1000025'], axis=1)
    imp = SimpleImputer(missing_values=-99999, strategy="mean", fill_value=None, verbose=0, copy=True)
    veriyeni = imp.fit_transform(veriyeni) # sklearn
    # 8 adet özelliğe bağlı bir giriş katmanımız var 
    # (Hücre Boyutunun Düzgünlüğü, Hücre Şeklinin Düzgünlüğü, Marjinal Yapışma, Tek Epitel Hücre Boyutu,
    # Çıplak Çekirdekler, Uyumlu Kromatin, Normal Nikloeller, Normal Nikloeller, Mitoz)
    # bu 8 katmana göre 1 tane veriyi tahmin etmeye çalışıyoruz
    giris = veriyeni[:, 0:8]  # giris.shape => (698,8) 8 özelliğe bağlı giriş katmanımız
    cikis = veriyeni[:, 9]   # cikis.shape => 8 özellikten tahmin ettiğimiz çıkış katmanımız
    # *** VERİ SETİ İŞLEMLERİ TAMAM ŞİMDİ MODELİMİZİ OLUŞTURALIM
    # bu altta vereceğim fotoğraf sequential le alakalı sunumda felan kullanıp 63 ve 64.satırları sil
    # https://bilimfili.com/wp-content/uploads/2015/12/yapay-sinir-aglari1-bilimfilicom.jpg
    model = Sequential() # Yapay sinir ağları algılayıcıların ardışık olmasına bağlı 
    # Dense: yapay sinir ağında görülen her ağ kendinden sonraki noktalara bağlı
    # input dimension ' a kaç tane özelliğimiz olduğunu yazıyoruz biz 8 özellikten output u tahmin edeceğiz
    model.add(Dense(10, input_dim=8))
    # Aktivasyon fonksiyonuna sokalım
    # Step Fonksiyonu: Bir eşik değeri alarak ikili bir sınıflandırma çıktısı (0 yada 1) üretir.
    # Sigmoid Fonksiyonu: En yaygın kullanılan aktivasyon fonksiyonlarından birisidir, [0,1] aralığında çıktı üretir.
    # Tanh Fonksiyonu: [-1,1] aralığında çıktı üreten doğrusal olmayan bir fonksiyondur.
    # ReLU Fonksiyonu: Doğrusal olmayan bir fonksiyondur. ReLU fonksiyonu negatif girdiler için 0 değerini alırken, x pozitif girdiler için x değerini almaktadır.
    # Softmax Fonksiyonu: Çoklu sınıflandırma problemleri için kullanılan bu fonksiyon, verilen her bir girdinin bir sınıfa ait olma olasılığını gösteren [0,1] arası çıktılar üretmektedir.
    # Softplus Fonksiyonu: Sigmoid ve Tanh gibi geleneksel aktivasyon fonksiyonlarına alternatif olarak sunulan bu fonksiyon (0, +∞) aralığında türevlenebilir bir çıktı üretmektedir.
    # ELU Fonksiyonu: Üstel lineer birim, negatif girdiler hariç ReLU ile benzerdir. Negatif girdilerde ise genellikle 1.0 alınan alfa parametresi almaktadır.
    # PReLU Fonksiyonu: Parametrik ReLU olarak geçen bu aktivasyon fonksiyonu da negatif girdiler için extra alfa sabiti ile verilen girdinin çarpım sonucunu çıktı olarak üretmektedir.
    # Swish Fonksiyonu:  Google araştırmacıları tarafından yeni keşfedilen bu fonksiyon girdiler ile sigmoid fonksiyonunun çarpımını çıktı olarak üretmektedir.
    # İlk başta hidden layers kısmını yazdık şimdi aktivasyon fonksiyonuyla verilerimizi normalize ettik yani 0-1 arasına yerleştirdik
    # NEDEN RELU yu kullandık?
    # Matrislerde sürekli y = mx + b işlemi çalışacağı için çok yüksek değerler elde ediyoruz biz bunu belli bir değer arasına sokmamız lazım
    # bunun için aktivasyon fonksiyonu kullanıyoruz verilerimizi 0 ile 1 arasına sokuyoruz 
    model.add(Activation('relu'))  # model.add(Activation('tanh')) daha hızlı sonuç verdi
    # Katmandaki node ların yarısını o tekrar içine sokmuyor eğer 0.2 yazılırsa 5'te 1 ini o tekrar için işleme sokmaz.
    # farklı node ları işleme sokmamızın nedeni veri seti ezberinin önüne geçmek için yapıyoruz 
    # eğer dropout kullanmazsak tahminimiz %100 olur fakat bu ezberlenmiş bir model demektir bizim için makul değer %90-%95 
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('relu'))  # model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))  # hep en sonda tut
    # Yapay sinir ağımızı oluşturduk 
    # lr: learning rate =>Ne kadar hızlı öğreneceğimizi anlamaya çalışan bir sistem
    # lr ile epoch arasında ters orantı var
    # lr yi düşük alrısak epoch değerini yüksek almamız gerekir 
    optimizer = keras.optimizers.SGD(lr=0.01)
    # gerçek-tahmini karesini alıp türevini 0 a eşitliyoruz 
    # algoritmamızın ne kadar doğru ne kadar yanlış yaptığını anlamak için metrics = accuracy yapıyoruz 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # modelimiz bitti şimdi bu verileri modele yerleştirmek kaldı 
    # 2: iyi huylu tümör, 4: kötü huylu tümör
    # epochs: veri setini ayrı ayrı 10 kere tarayacak
    # batch_size: aynı anda kaç bit lik işlemi hafızaya alsın yazılmazsa kendisi otomatik değer atar.
    # validation_split: (Doğrulama kısmı) bütün verimizin bir kısmını yapay sinir ağımıza yerleştirelim
    # Elimizde kalan sinir ağına sokmadığımız işlenmemiş veriyi veri setine sokarak tahmin etmesini sağlayacağız.
    # eğer tamamıyla işleme sokarsak model verileri ezberler. Bizim amacımız görmeden tahmin etmesi 
    model.fit(giris, cikis, epochs=10, batch_size=32, validation_split=0.20)
    # inputlar
    uniformity_cell_size = request.POST.get("uniformity_cell_size")
    uniformity_cell_shape = request.POST.get("uniformity_cell_shape")
    marginal_adhesion = request.POST.get("marginal_adhesion")
    single_epithelial_cell_size = request.POST.get("single_epithelial_cell_size")
    bare_nuclei = request.POST.get("bare_nuclei")
    bland_chromatin = request.POST.get("bland_chromatin")
    normal_nucleoli = request.POST.get("normal_nucleoli")
    mitoses = request.POST.get("mitoses")
    #a = 5
    #b = 5
    #c = 5
    #d = 8
    #e = 10
    #f = 8
    #g = 7
    #h = 3
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