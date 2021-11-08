# image_parser
Скрипт, при помощи которого можно парсить изображения из научных статей.

## Instalation
Устанавливаем layoutparser, PyTorch и Detectron 2
</br>`!pip install layoutparser torch && pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"`
Пакеты, необходимые для OCR
</br>`!pip install "layoutparser[layoutmodels]" # Install DL layout model toolkit`
</br>`!pip install "layoutparser[ocr]" # Install OCR toolkit`
</br>`!git clone https://github.com/Layout-Parser/layout-parser.git`
</br>Устаавливаем Tesseract OCR
</br>`!sudo apt install tesseract-ocr -y`
</br>`!sudo apt install libtesseract-dev -y`
</br>`!sudo apt update && sudo apt install -y poppler-utils`