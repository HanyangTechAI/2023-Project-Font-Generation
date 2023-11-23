import numpy as np
import matplotlib.pyplot as plt
import cv2
import maxflow
import os

class PostProcessing:
  def imgProcessing(self, path, output_path):
    images = load_output(path)
    # apply cleansing
    cleansing_images = [cleansing(image) for image in images]
    #apply sharpening
    sharpening_images = [sharpening(image) for image in cleansing_images]
    save_images(sharpening_images, output_path)

  def load_output(path):
    files = os.listdir(path)
    images = []
    for file_name in files:
      image_path = os.path.join(path, file_name)
      img = cv.imread(image_path)
      images.append(img)
    return images

  def save_images(images, output_dir):
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)    
    for i, img in enumerate(images): # 생성되는 결과물의 글자를 보고 이후에 파일 이름 변경 필요
      path = os.path.join(output_dir, f'image_{i+1}.png')
      cv2.imwrite(path, img)

  def cleansing(image):
    # cleansing (이미지 상태에 맞춰 수정 필요 - 현재 mxfont 결과물 기준)
    cleansing = cv2.fastNlMeansDenoising(image,None,70,7,21)
    return cleansing

  def sharpening(image):
    # sharpening (이미지 상태에 맞춰 수정 필요 - 현재 mxfont 결과물 기준)
    kernel_sharpening1 = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
    sharpened = cv2.filter2D(cleansing, -1, kernel_sharpening1)
    return sharpend