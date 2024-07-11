from PIL import Image
import os

def make_rotate(folder_path, rotate_number):
    # 이미지가 들어있는 폴더 경로

    # 폴더 내의 모든 파일 가져오기
    files = os.listdir(folder_path)
    angle_step = 360 / rotate_number

    # 각 이미지 파일에 대해 회전 후 저장
    for file_name in files:
        for i in range(rotate_number):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                # 이미지 파일 오픈
                image_path = os.path.join(folder_path, file_name)
                image = Image.open(image_path)

                # 이미지 회전 (원하는 각도 설정)
                rotated_image = image.rotate(angle_step * i)  # 여기서는 90도 회전 예시

                # 회전된 이미지 저장
                rotated_image.save(os.path.join(folder_path, f"rotated_{i+1}" + file_name))

        print("모든 이미지 회전 및 저장이 완료되었습니다.")

if __name__ == "__main__":
    folder_path = input("파일경로")
    rotate_number = 30
    make_rotate(folder_path, rotate_number)