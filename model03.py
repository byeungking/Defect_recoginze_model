import cv2
import numpy as np


def split_image(image_path, grid_size=(5, 5)):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    grid_h, grid_w = h // grid_size[0], w // grid_size[1]

    sub_images = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if i in range(1, 4) and j in range(1, 4):  # (2,2)부터 (4,4)까지의 부분은 건너뜁니다.
                sub_image = image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                sub_images.append(sub_image)
    return image, sub_images, (grid_h, grid_w)

def compare_with_model(sub_image, model, class_labels):
    # 이미지 전처리
    sub_image_resized = cv2.resize(sub_image, (model.input_shape[1], model.input_shape[2]))
    sub_image_array = np.expand_dims(sub_image_resized, axis=0)  # 모델에 맞게 배치 차원 추가
    sub_image_array = sub_image_array / 255.0  # 정규화

    # 모델 예측
    prediction = model.predict(sub_image_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    result = class_labels[class_idx]
    return result

def process_images(image_path, model, class_labels, grid_size=(5, 5)):
    image, sub_images, (grid_h, grid_w) = split_image(image_path, grid_size)
    results = []

    idx = 0  # 결과 리스트의 인덱스
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if i in range(1, 4) and j in range(1, 4):  # (2,2)부터 (4,4)까지의 부분은 건너뜁니다.
                sub_image = sub_images[idx]
                result = compare_with_model(sub_image, model, class_labels)
                results.append(result)

                # 결과 이니셜 추가
                position = (j * grid_w + 10, i * grid_h + 30)
                cv2.putText(image, result, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                idx += 1

    # 그리드 선 그리기
    for i in range(1, grid_size[0]):
        cv2.line(image, (0, i * grid_h), (image.shape[1], i * grid_h), (255, 0, 0), 1)
    for j in range(1, grid_size[1]):
        cv2.line(image, (j * grid_w, 0), (j * grid_w, image.shape[0]), (255, 0, 0), 1)

    return image, results