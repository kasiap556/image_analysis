import numpy as np
import cv2 as cv
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

########### Przygotowanie danych treningowych ###########

datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'datasets',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'datasets',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

print("ktora klasa:", train_generator.class_indices)


########### Model CNN, trenowanie ###########

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)), #rozmiar i 3 kanaly rgb
    layers.Conv2D(32, (3, 3), activation='relu'), #filtry do wykrywania cech, np krawedzie
    layers.MaxPooling2D(2, 2), #redukcja wymiarow
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(), #splaszcza do wektora
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
#Adam dosasowywuje wspolczynniki uczenia, binary_crossentropy - klasyfikacja 2 klasowa, 
#accuracy mierzy dokladnosc podczas treningu

print("trenowanie...")
model.fit(train_generator, validation_data=val_generator, epochs=20)

########### wczytanie zdjęcia ###########

img_path = 'datasets/predict/b16.jpg'
#b2, b3, b4, 
img = cv.imread(img_path)

if img is None:
    raise ValueError("Obraz nie zostal wczytany, pewnie zla ścieżka lub nazwa pliku")

img = cv.resize(img, (600, 900))
img_copy = img.copy()


# ########### niebieski ##########
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) #na hsv

# lower_blue = np.array([90, 10, 150])   # odcień, nasycenie, jasność
# upper_blue = np.array([130, 255, 255])

# blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

# cv.imshow("blue_mask", blue_mask)

########### segmentacja obiektów - Watershed ###########

gauss_blur = cv.GaussianBlur(img, (3,3), 0)
# median_blur = cv.medianBlur(img, 3)

gray = cv.cvtColor(gauss_blur, cv.COLOR_BGR2GRAY)
# gray = cv.bitwise_and(cv.cvtColor(img, cv.COLOR_BGR2GRAY), blue_mask)
# gray = blue_mask

_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  #cv.THRESH_BINARY_INV + 
#otsu wybiera prob jasnosci # + cv.THRESH_OTSU (i zamienic na 0 w drugim)
# do zmieniana na binarny czarno-bialy

cv.imshow("thresh", thresh)

kernel = np.ones((4, 4), np.uint8) #3,3
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=3) #usuwa szumy
background = cv.dilate(opening, kernel, iterations=2) #rozszerza biale obszary?

dist = cv.distanceTransform(opening, cv.DIST_L2, 5) #odleglosc bialych pikseli od czarnych, indentyfikuje srodki obiektow - foreground
_, foreground = cv.threshold(dist, 0.15 * dist.max(), 255, 0) #mniej - wykryje mniej wyrazne
foreground = np.uint8(foreground)
unknown = cv.subtract(background, foreground)

_, markers = cv.connectedComponents(foreground) #marker - etykieta liczba
markers = markers + 1
markers[unknown == 255] = 0

markers = cv.watershed(img, markers) #dzieli nas obszary
img[markers == -1] = [0, 0, 255]  #linie podziału - czerwony, -1 granice miedzy segmentami

########### klasyfikacja segmentów ############

unique_markers = np.unique(markers)
rois = []
boxes = []

for marker in unique_markers:
    if marker <= 1:
        continue

    mask = np.uint8(markers == marker) #maska obszaru
    roi_masked  = cv.bitwise_and(img_copy, img_copy, mask=mask)
    x, y, w, h = cv.boundingRect(mask)
    print(x, y, w, h)

    if w<20 or h < 50:
        continue

    boxes.append((x, y, x+w, y+h))

    # roi = img_copy[y:y + h, x:x + w] #x,y lewo gora, w,h szerokosc wysokosc
    roi = roi_masked[y:y + h, x:x + w]  # przytnij TYLKO tę część maskowanego obrazu

    position = h / w
    if position < 1.2:
        continue  # za niski to skip

    if roi.shape[0] < 50 or roi.shape[1] < 20:
        continue  # za małe to skip

    roi_resized = cv.resize(roi, (224, 224))
    roi_input = roi_resized / 255.0
    # img_input = cv.resize(img, (224, 224)) / 255.0
    roi_input = np.expand_dims(roi_input, axis=0) #na format batcha #reshape((1, 224, 224, 3))
    # img_input = np.expand_dims(img_input, axis=0)

    prediction = model.predict(roi_input, verbose=0)
    label = "Butelka" if prediction[0][0] > 0.4 else "Brak" #mniej to bardziej czuly

    color = (0, 255, 0) if label == "Butelka" else (0, 0, 255)
    cv.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
    cv.putText(img_copy, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


cv.imshow("klasyfikacja", img_copy)
cv.waitKey(0)
cv.destroyAllWindows()
