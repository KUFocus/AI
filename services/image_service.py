from PIL import Image, UnidentifiedImageError
import numpy as np
from easyocr import Reader
import torch

# EasyOCR Reader 초기화
reader = Reader(['ko'], gpu=torch.cuda.is_available())

def process_image_file(file, file_name):
    try:
        # 이미지 파일 로드
        image = Image.open(file.stream)

        # 이미지 크기 확인 후 리사이즈
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size)

        # EasyOCR로 텍스트 감지 및 인식
        results = reader.readtext(np.array(image), detail=0)

        return {'filename': file_name, 'text': ' '.join(results)}

    except UnidentifiedImageError as e:
        raise Exception(f"Invalid image file: {str(e)}")
