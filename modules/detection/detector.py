"""
티켓 검출기 모듈 

OpenCV 기반 티켓 검출 및 추출     
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

image_path = "test/test.jpg"
img = cv2.imread(image_path)

if img is None:
    print("이미지 불러올 수 없습니다")
else:
    start_point = (50, 50)
    end_point = (250,250)
    
    color = (0,255, 0)
    thickness = 3
    
    cv2.rectangle(img, start_point, end_point, color, thickness)
    
    cv2.imshow("Rectangle Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()