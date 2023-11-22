import os
import itertools
import numpy as np
import random
import math
from matplotlib import pyplot as plt
import cv2

# Seq: A-Z, a-z, 0-9, SPECIAL_CHARS

ALL_CHARS = ['가', '고', '깊', '꿈', '내', '넓', '다', '란', '람', '마', '며', '밝', '사', '서', '아', '어', '없', '에', '우', '은', '을', '음', '자', '하', '힐']


class SHEETtoPNG:
  """Converter class to convert input sample sheet to character PNGs."""

  def convert(self, sheet, characters_dir, cols=5, rows=5):

    threshold_value = 200;
    if os.path.isdir(sheet):
        raise IsADirectoryError("Sheet parameter should not be a directory.")
    characters = self.detect_characters(
        sheet, threshold_value, cols=cols, rows=rows
    )


    for i in range(len(characters)):
      characters[i][0] = cv2.cvtColor(characters[i][0], cv2.COLOR_BGR2GRAY)
      characters[i][0] = self.m_thresh_trunc(characters[i][0], 225, 255)
      characters[i][0] = self.centering_image(characters[i][0])
      characters[i] = characters[i][0]

    self.save_images(
        characters,
        characters_dir,
    )


  def detect_sheet(self, camera_image, threshold_value):
    image = cv2.imread(camera_image)
    #image, image_norm = self.reduce_shadow(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    image_binary = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel, iterations=10)

    self.plt_imshow(cv2.cvtColor(image_binary, cv2.COLOR_GRAY2BGR))
    #_, image_binary = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, image_binary = cv2.threshold(image_binary, 255, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #image_binary = cv2.adaptiveThreshold(image_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
     #                                    , cv2.THRESH_BINARY,55, -20)

    self.plt_imshow(cv2.cvtColor(image_binary, cv2.COLOR_GRAY2BGR))

    contours, hier = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    image_binary = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2BGR)
    c = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    image_test = cv2.drawContours(image_binary.copy(), contours, -1,c,2, cv2.LINE_8, hier)

    self.plt_imshow(image_test)
    contour_biggest = None
    contour_area_biggest = -1
    for contour in contours:
      area = cv2.contourArea(contour)
      if area > contour_area_biggest:
        contour_biggest = contour
        contour_area_biggest = area

    print(contour_area_biggest)

    epsilon = 0.1 * cv2.arcLength(contour, True)

    contour_biggest_approx = cv2.approxPolyDP(contour_biggest, epsilon ,True)
    print(cv2.arcLength(contour_biggest_approx, True))

    c = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    image_test = cv2.drawContours(image_binary.copy(), contour_biggest_approx, -1,c,2, cv2.LINE_8)

    self.plt_imshow(image_test)
    #c = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    #cv2.drawContours(image_binary, contours, -1,c,2, cv2.LINE_8, hier)
    return image_binary, contour_biggest_approx


  def detect_characters(self, sheet_image, threshold_value, cols=5, rows=5):
    """Detect contours on the input image and filter them to get only characters.

    Uses opencv to threshold the image for better contour detection. After finding all
    contours, they are filtered based on area, cropped and then sorted sequentially based
    on coordinates. Finally returs the cols*rows top candidates for being the character
    containing contours.

    Parameters
    ----------
    sheet_image : str
        Path to the sheet file to be converted.
    threshold_value : int
        Value to adjust thresholding of the image for better contour detection.
    cols : int, default=8
        Number of columns of expected contours. Defaults to 8 based on the default sample.
    rows : int, default=10
        Number of rows of expected contours. Defaults to 10 based on the default sample.

    Returns
    -------
    sorted_characters : list of list
        Final rows*cols contours in form of list of list arranged as:
        sorted_characters[x][y] denotes contour at x, y position in the input grid.
    """
    # TODO Raise errors and suggest where the problem might be
    # Read the image and convert to grayscale
    image = cv2.imread(sheet_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold and filter the image for better contour detection
    _, thresh = cv2.threshold(gray, threshold_value, 255, 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, close_kernel)
    # Search for contours.
    contours, h = cv2.findContours(
        close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Filter contours based on number of sides and then reverse sort by area.
    contours = sorted(
        filter(
            lambda cnt: len(
                cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            )
            == 4,
            contours,
        ),
        key=cv2.contourArea,
        reverse=True,
    )

    # Calculate the bounding of the first contour and approximate the height
    # and width for final cropping.
    x, y, w, h = cv2.boundingRect(contours[0])
    space_h, space_w = 7 * h // 16, 7 * w // 16

    # Since amongst all the contours, the expected case is that the 4 sided contours
    # containing the characters should have the maximum area, so we loop through the first
    # rows*colums contours and add them to final list after cropping.
    characters = []
    for i in range(rows * cols):
        x, y, w, h = cv2.boundingRect(contours[i])
        cx, cy = x + w // 2, y + h // 2

        roi = image[cy - space_h : cy + space_h, cx - space_w : cx + space_w]
        characters.append([roi, cx, cy])

    # Now we have the characters but since they are all mixed up we need to position them.
    # Sort characters based on 'y' coordinate and group them by number of rows at a time. Then
    # sort each group based on the 'x' coordinate.
    characters.sort(key=lambda x: x[2])
    sorted_characters = []
    for k in range(rows):
        sorted_characters.extend(
            sorted(characters[cols * k : cols * (k + 1)], key=lambda x: x[1])
        )


    return sorted_characters


  def save_images(self, characters, characters_dir):
    """Create directory for each character and save as PNG.

    Creates directory and PNG file for each image as following:

        characters_dir/ord(character)/ord(character).png  (SINGLE SHEET INPUT)
        characters_dir/sheet_filename/ord(character)/ord(character).png  (MULTIPLE SHEETS INPUT)

    Parameters
    ----------
    characters : list of list
        Sorted list of character images each inner list representing a row of images.
    characters_dir : str
        Path to directory to save characters in.
    """
    os.makedirs(characters_dir, exist_ok=True)

    # Create directory for each character and save the png for the characters
    # Structure (single sheet): UserProvidedDir/ord(character)/ord(character).png
    # Structure (multiple sheets): UserProvidedDir/sheet_filename/ord(character)/ord(character).png
    for k, images in enumerate(characters) :
      character = os.path.join(characters_dir, str(ALL_CHARS[k]))
      if not os.path.exists(character):
          os.mkdir(character)
      cv2.imwrite(
          os.path.join(character, str(ALL_CHARS[k]) + ".png"),
          images,
        )


  def plt_imshow(self, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

  def reduce_shadow(self, image):
    rgb_planes = cv2.split(image)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
      dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
      bg_img = cv2.medianBlur(dilated_img, 21)
      diff_img = 255 - cv2.absdiff(plane, bg_img)
      norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      result_planes.append(diff_img)
      result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return result, result_norm

  def tight_crop_image(self, img, verbose=False, resize_fix=False):

    self.plt_imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    img_size = img.shape
    full_white = img_size
    print(img.shape)
    print(np.sum(img, axis=0))
    col_sum = np.where(np.sum(img, axis=0) < 255* full_white[0])
    row_sum = np.where(np.sum(img, axis=1) < 255* full_white[1])

    print(col_sum)

    #assert (len(col_sum) == 0)
    #assert (len(row_sum) == 0)

    y1, y2 = row_sum[0][0], row_sum[0][-1]+1
    x1, x2 = col_sum[0][0], col_sum[0][-1]+1

    cropped_image = img[y1:y2, x1:x2]
    cropped_image_size = cropped_image.shape
    self.plt_imshow(img)
    self.plt_imshow(cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR))

    return cropped_image


  def add_padding(self, img, image_size=128, verbose=False, pad_value=None):
    ## To-do list
    ## width, height이 더 클 때 처리
    height, width = img.shape
    print('add_padding before')
    self.plt_imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    # Adding padding of x axis - left, right
    pad_x_width = (image_size - width) // 2
    pad_x = np.full((height, pad_x_width), pad_value, dtype=np.uint8)
    print('pad_x')
    print(pad_x)
    print('img')
    print(img)
    img = np.concatenate((pad_x, img), axis=1)
    img = np.concatenate((img, pad_x), axis=1)

    print('add_padding after x')
    self.plt_imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


    width = img.shape[1]

    # Adding padding of y axis - top, bottom
    pad_y_height = (image_size - height) // 2
    pad_y = np.full((pad_y_height, width), pad_value, dtype=np.uint8)

    img = np.concatenate((pad_y, img), axis=0)
    img = np.concatenate((img, pad_y), axis=0)
    print('add_padding after y')
    self.plt_imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    # Match to original image size
    width = img.shape[1]
    if img.shape[0] % 2:
        pad = np.full((1, width), pad_value, dtype=np.uint8)
        img = np.concatenate((pad, img), axis=0)
    height = img.shape[0]
    if img.shape[1] % 2:
        pad = np.full((height, 1), pad_value, dtype=np.uint8)
        img = np.concatenate((pad, img), axis=1)
    self.plt_imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    return img

  def resize_char(self, image, image_area= 80*80):
    width, height = image.shape
    resize_ratio = math.sqrt(image_area / (width*height))
    new_image_CUBIC = cv2.resize(image,dsize=None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)

    self.plt_imshow(cv2.cvtColor(new_image_CUBIC, cv2.COLOR_GRAY2BGR))

    return new_image_CUBIC

  def sharpening(self, image, strength):
    b = (1 - strength) / 8
    sharpening_kernel = np.array([[b, b, b],
                                  [b, strength, b],
                                  [b, b, b]])
    output = cv2.filter2D(image, -1, sharpening_kernel)
    return output

  def centering_image(self, img, image_size=128, verbose=False, pad_value=255):
    if not pad_value:
        pad_value = img[0][0]
    cropped_image = self.tight_crop_image(img, verbose=verbose)
    resized_image = self.resize_char(cropped_image)
    centered_image = self.add_padding(resized_image, image_size=image_size, verbose=verbose, pad_value=pad_value)


    centered_image = cv2.GaussianBlur(centered_image,(0,0), 2)
    print('blurred center')
    self.plt_imshow(cv2.cvtColor(centered_image, cv2.COLOR_GRAY2BGR))
    centered_image = self.sharpening(centered_image, 57)
    print('sharped')
    self.plt_imshow(cv2.cvtColor(centered_image, cv2.COLOR_GRAY2BGR))
    centered_image = self.m_thresh_trunc(centered_image,220,255)

    centered_image = cv2.medianBlur(centered_image,3)
    self.plt_imshow(cv2.cvtColor(centered_image, cv2.COLOR_GRAY2BGR))
    return centered_image

  def m_thresh_trunc(self, img, thresh, maxVal):
    return np.where(img > thresh, maxVal,img)

