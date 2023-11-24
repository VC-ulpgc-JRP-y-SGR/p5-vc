## Práctica 5. Reconocimiento de matrículas

## Introducción

Para la realización de esta tarea hemos definido un pipeline del procesado paso a paso necesario para transformar la imagen de entrada a una lectura de matricula fiable.

El esquema sería el siguiente

![image pipeline](./resources/pipeline.png)

Como podemos observar nuestro pipeline esta segmentado en cuatro etapas. Cada una de ellas se ha tratado de forma abstracta con interfaces que en un futuro podrán permitir el cambio de tecnologías y la modularidad del código realizado.

1. PlateDetector: esta sería la etapa dedicada a la detección de las diferentes matrículas.
2. TextProccesor: esta etapa cogería la imagen resultante de la mátricula y la trataria para que el texto sea mas legible para la siguiente etapa.
3. TextExtractor: esta etapa cogería la matricula ya tratada y extraeria el texto a partir de un OCR.
4. TextMatcher: comprobaría que el resultado de la extracción del texto cumple el formato requerido para ser considerado una mátricula.

## Detección de mátriculas

La interfaz que hemos creado para la detección de mátriculas es la siguiente

```python
class PlateDetector(ABC):
    @abstractmethod
    def detect(self, img: array) -> list[array]:
        pass
```

Como sabemos, en python no existen las interfaces por lo tanto lo hemos implementado como una clase abstracta. Esta recibe una imagen y devuelve una lista de imágenes puesto que en una foto podría haber varias matrículas.

La implementación que hemos realizado esta interfaz ha sido con Yolov8. Hemos entrenado el modelo con el siguiente dataset para poder detectar mátriculas.

https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e


```python
from ultralytics import YOLO

class YOLOPlateDetector(PlateDetector):
    def __init__(self, model, category = 0):
        self.model = YOLO(model).cuda()
        self.category = category

    def detect(self, img):
        results = self.model(img)
        plates = []
        for x in results:
            boxes = x.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                category = int(box.cls[0])
                if category == 0:
                    plates.append((x1, y1, x2, y2, img[y1:y2, x1:x2]))
        return plates
```

### Procesado de la matrícula

Esta etapa sería la segunda en nuestro pipeline. Esta encargada de procesar las diferentes mátriculas con filtros usados en la asignatura como Canny, Threshold etc. de forma que al OCR se le facilite la tarea de detección.

Aunque siendo honestos ha funcionado mejor sin ningún tipo de filtros. Pensábamos que al extraer los bordes nada mas con Canny o aplicar ciertos thresholds para eliminar colores diferentes del blanco y negro funcionaría mejor pero no ha sido así. Es más, ha funcionado incluso peor en varios casos.

Esta sería la interfaz de un procesador de matricula.

```python
class ImageProcessor(ABC):
    @abstractmethod
    def process(self, img: array) -> array:
        pass
```

Como podemos observar recibe una imagen y devuelve otra imagen procesada.

Hemos realizado varias implementaciones de esta interfaz. Una para aplicar Canny, otra para Sobel, otra para Threshold y un intento con FindCountours.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CannyImageProcessor(ImageProcessor):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def process(self, img: array) -> array:
        # apply Canny
        canny = cv2.Canny(img, self.x, self.y)
        plt.imshow(canny)
        plt.show()
        return canny

class SobelImageProcessor(ImageProcessor):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def process(self, img: array) -> array:
        # apply Sobel
        sobel = cv2.Sobel(img, cv2.CV_8U, self.x, self.y, ksize=5)
        plt.imshow(sobel)
        plt.show()
        return sobel

class ColorThresholdImageProcessor(ImageProcessor):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    
    def process(self, img: array) -> array:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # extract only black colors
        mask = cv2.inRange(img, self.lower, self.upper)
        img = cv2.bitwise_and(img, img, mask=mask)

        plt.imshow(img)
        plt.show()
        return img

class ContoursImageProcessor(ImageProcessor):
    def process(self, image: array) -> array:
        mask = np.ones(image.shape, dtype=np.uint8) * 255
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilate = thresh

        cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300:
                x,y,w,h = cv2.boundingRect(c)
                mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        
        plt.imshow(mask)
        plt.show()
        return mask
```

Como podemos observar estos se añaden al detector de matrículas en forma de lista para poder mezclar varios filtros.

```python
plate_detector = YOLOPlateDetector(model='plate_recognizer.pt', image_processors=[
        CannyImageProcessor(0, 100),
        ColorThresholdImageProcessor((0, 0, 0), (120, 120, 120))
    ]
)
```

### Extractor del texto

Por último quedaría coger las imágenes procesadas y extraer el correspondiente texto de ellas.

Para esto hemos creado la siguiente interfaz

```python
class TextExtractor(ABC):
    def __init__(self, matcher : PlateMatcher, processor : TextProcessor):
        self.matcher = matcher
        self.processor = processor

    @abstractmethod
    def extract(self, img: array) -> str:
        pass
```

Y la hemos implementado con EasyOCR, TesseractOCR y KerasOCR, siendo la que mejores resultados a dado EasyOCR.

```python
import easyocr
import pytesseract
from keras_ocr.recognition import Recognizer
from keras_ocr.detection import Detector

class TesseractTextExtractor(TextExtractor):
    def extract(self, img):
        pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
        text = pytesseract.image_to_string(img)
        matcher = RegexPlateMatcher()
        if matcher.match(text):
            text = self.processor.process(text)
            return text
        
class KerasOCRTextExtractor(TextExtractor):
    def extract(self, img):
        detector = Detector()
        recognizer = Recognizer()
        pipeline = pipeline.Pipeline(detector=detector, recognizer=recognizer)
        prediction_groups = pipeline.recognize([img])
        for group in prediction_groups:
            for word in group:
                text = word[0]
                text = self.processor.process(text)
                if(self.matcher.match(text)):
                    return text

class EasyOCRTextExtractor(TextExtractor):
    def extract(self, img):
        reader = easyocr.Reader(['es'], detector='dbnet18') 
        result = reader.readtext(img, batch_size=4)
        if(len(result) == 0): return ""
        finaltext = ""
        for x in reversed(result):
            text = x[1]
            text = self.processor.process(text)
            finaltext += str(text)
        if (self.matcher.match(finaltext)):
            return finaltext
```

### Identificadores de texto

Los identificadores de texto son usados para saber si el texto detectado por la etapa anterior corresponde al formato buscado. De esta forma podremos descartar detecciones erróneas.

La interfaz usada para este labor es la siguiente:

```python
class PlateMatcher(ABC):
    @abstractmethod
    def match(self, text : str) -> bool:
        pass
```

Se han realizado dos implementaciones, una con longitud de texto y otra con regex

```python
import re

class LengthPlateMatcher(PlateMatcher):
    def match(self, text : str) -> bool:
        if len(text) == 7: return True
        return False

class RegexPlateMatcher(PlateMatcher):
    def match(self, text : str) -> bool:
        pattern = re.compile("^\d{4}[A-Z]{3}$")
        # pattern = re.compile("^[0-9]{4}([B-D]|[F-H]|[J-N]|[P-T]|[V-Z]){3}$")
        if pattern.match(text):
            return True
        else:
            return False
```

### Mejoras de rendimiento propuestas

Para mejorar el rendimiento se ha probado a implementar multithreading lanzando varios hilos para procesar a la vez multiples iamgenes. El código siguiente ha sido el usado para probar el rendimiento entre una ejecución con normal y otra threads.

Ejecución normal:

```python
text_extractor=EasyOCRTextExtractor(
    processor=BasicTextProcessor(), 
    matcher=LengthPlateMatcher()
)

plate_detector = YOLOPlateDetector(model='plate_recognizer.pt', image_processors=[
    ]
)

detector = CarPlateDetector(
    text_extractor,
    plate_detector
)

def detectImage():
    result = detector.detect('./plates/coches2.jpg')
    print(result)

for x in range(0, 10):
    detectImage()
```

Ejecución con threads:

```python
import threading 

text_extractor=EasyOCRTextExtractor(
    processor=BasicTextProcessor(), 
    matcher=LengthPlateMatcher()
)

plate_detector = YOLOPlateDetector(model='plate_recognizer.pt')

detector = CarPlateDetector(
    text_extractor,
    plate_detector
)

def  detectImage():
    result = detector.detect('./plates/coches2.jpg')

threads = [
]

def add_threads(num_threads=10):
    for x in range(0, num_threads):
        threads.append(threading.Thread(target=detectImage))
    
def join_threads():
    for x in threads:
        x.start()
    for x in threads:
        x.join()

add_threads(10)
join_threads()
```

Tardando la ejecución normal 2m y 21 segundos con 10 imágenes. Y la ejecución con threads 29 segundos como se puede apreciar en el notebook. Por lo tanto la mejora sí que es substancial. Sin embargo, para implementar threads en el video sería necesario código que lanzara un thread por frame. Conociendo la cantidad de frames que tiene un vídeo actual esto sería inviable por limitaciones del procesador. Por lo tanto la implementación en el video se ha realizado sin threads.

### Predicciones en foto

El código usado para una foto ha sido el siguiente:

```python
import cv2
import time

image = cv2.imread('./plates/coches2.jpg')
processor = BasicTextProcessor()
# matcher = LengthPlateMatcher()
matcher = RegexPlateMatcher()
detector = YOLOPlateDetector(model='plate_recognizer.pt')
extractor = EasyOCRPlateExtractor(matcher, processor)
start = time.time()
plates = detector.detect(image)
print("plate detection time:", time.time() - start, "seconds")

# Dibujar rectángulos alrededor de las placas detectadas
for plate in plates:
    if len(plate) == 5:  # Asegurarse de que haya cinco valores en la tupla
        x1, y1, x2, y2, img = plate
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        start = time.time()
        detection = extractor.extract(img)
        print("easyocr reading time:", time.time() - start, "seconds")
        if detection == None or detection == '': detection = "unknown plate"
        cv2.putText(image, str(detection), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con los rectángulos dibujados
cv2.imshow('Plates Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Predicciones en video

El código usado para el video ha sido el siguiente:

```python
import cv2

cap = cv2.VideoCapture('videos/license_plates_fps.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

processor = BasicTextProcessor()
# matcher = LengthPlateMatcher()
matcher = RegexPlateMatcher()
detector = YOLOPlateDetector(model='plate_recognizer.pt')
extractor = EasyOCRPlateExtractor(matcher, processor)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (width, height))
    plates = detector.detect(frame)

    # Dibujar rectángulos alrededor de las placas detectadas
    for plate in plates:
        if len(plate) == 5:  # Asegurarse de que haya cuatro valores en la tupla
            x1, y1, x2, y2, img = plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            detection = extractor.extract(img)
            if detection == None or detection == '': detection = "unknown plate"
            cv2.putText(frame, detection, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Plates', frame)
    if cv2.waitKey(20) == 27: 
        break

cap.release()
cv2.destroyAllWindows()
```

A su vez el vídeo lo hemos grabado nosotros mismos en la calle proporcionando un entorno más realista que los posibles encontrados en internet.