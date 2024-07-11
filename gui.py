import os
from datetime import datetime
import cv2
from tensorflow.keras.models import load_model

# GUI
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# 내부 모듈
import model01
import model02
import model03
from threads import TrainingThread

# UI파일 연결
main_class = uic.loadUiType("GUI/main.ui")[0]

class Communicate(QObject):
    progress_signal = pyqtSignal(int)
    training_finished = pyqtSignal()

class WindowClass(QMainWindow, main_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("PCB기판 불량 체크 프로그램")

        # 탭 관련
        self.tabWidget.setTabText(0, "Predict")
        self.tabWidget.setTabText(1, "Train")
        self.tabWidget.setCurrentIndex(0)

        # 클래스 라벨 설정
        self.class_labels = ['D1', 'D2', 'D3', 'N']
        self.selected_image_path = None

        # UI 관련 함수
        self.ui_tab01()
        self.ui_tab02()



    def ui_tab01(self):
        # 원본 이미지 설정
        self.original = QPixmap("./img02/chip_origin.png")  # 이미지 지정
        self.Label_img_1.setPixmap(QPixmap(self.original).scaled(QSize(300, 300)))  # 브랜드 로고

        # 기본 저장 위치 설정
        default_directory = "기본 디렉토리"
        self.default_directory = os.path.expanduser("save")
        if not os.path.exists(self.default_directory):
            os.makedirs(self.default_directory)
        self.selected_directory = self.default_directory
        self.Label_directory.setText(default_directory)

        # 학습 모델 세팅
        self.model_dir = 'models/chip_check_defect.hdf5'  # 초기 모델 초기화
        if os.path.exists(self.model_dir):
            self.model = load_model(self.model_dir)
        else:
            self.model = None

        self.result_images = {'model02': None, 'model03': None}

        # 각종 버튼
        self.Botton_img_setting.clicked.connect(self.open_image)
        self.Button_directory.clicked.connect(self.directory_save)
        self.Button_open_directory.clicked.connect(self.open_directory)
        self.Button_reset.clicked.connect(self.reset_img)

        # 모델 버튼
        self.Button_run_2.clicked.connect(self.model02_run)
        self.Button_run_3.clicked.connect(self.model03_run)

        # 저장 버튼
        self.Button_save_2.clicked.connect(lambda: self.save_result_image('model02', self.Label_img_2))
        self.Button_save_3.clicked.connect(lambda: self.save_result_image('model03', self.Label_img_3))

        # 모델 콤보 박스
        self.Label_model.setText("선택된 모델: " + "기본")
        self.populateComboBox()
        self.comboBox_model.setCurrentIndex(0)  # 첫 번째 아이템을 디폴트로 선택
        self.comboBox_model.activated[str].connect(self.onActivated)
        self.Button_model_reset.clicked.connect(self.populateComboBox)

    def ui_tab02(self):
        # 각종 패스
        self.train_image_path = None
        self.train_save_path = None

        # 이미지 크기 스핀박스
        self.spinBox_width.setMinimum(1)
        self.spinBox_height.setMinimum(1)
        self.spinBox_width.setValue(60)
        self.spinBox_height.setValue(60)

        # 슬라이더
        self.Label_slider_value.setText("Value: 10")
        self.slider_epoch.setRange(10, 100)
        self.slider_epoch.valueChanged.connect(self.Label_update)
        self.slider_epoch.valueChanged.connect(self.update_progressbar_maximum)

        # 진행 막대기
        self.progressBar.setMaximum(self.slider_epoch.value())
        self.progressBar.setValue(0)  # 프로그레스바 초기화

        # 각종 버튼
        self.Button_image_path.clicked.connect(self.tab2_train_image_path_save)
        self.Button_save_path.clicked.connect(self.tab2_train_save_path_save)
        self.Button_model_creation.clicked.connect(self.start_training)

        # 신호 인스턴스
        self.comm = Communicate()
        self.comm.progress_signal.connect(self.update_progress)
        self.comm.training_finished.connect(self.on_training_finished)

        # 학습 모델 초기화
        self.training_thread = None

    def update_progressbar_maximum(self, value):
        self.progressBar.setMaximum(value)

    @pyqtSlot(int)
    def update_progress(self, epoch):  # 학습 중 출력 메세지
        self.progressBar.setValue(epoch)
        self.label_progress.setText(f'Training Progress: Epoch {epoch}/{self.slider_epoch.value()}')

    @pyqtSlot()
    def on_training_finished(self):  # 학습 종료 후 출력 메세지
        self.label_progress.setText('Training Finished!')
        self.Button_model_creation.setEnabled(True)

    def start_training(self):  # 모델 학습
        self.Button_model_creation.setEnabled(False)

        if self.lineEdit_name.text() != "":
            if self.train_image_path is not None:
                if self.train_save_path is not None:
                    self.progressBar.setValue(0)  # 프로그레스바 초기화
                    self.training_thread = TrainingThread(
                        self.spinBox_width.value(),
                        self.spinBox_height.value(),
                        self.train_image_path,
                        self.slider_epoch.value(),
                        self.train_save_path,
                        self.lineEdit_name.text(),
                        self.comm.progress_signal,
                        self.comm.training_finished
                    )
                    self.training_thread.start()
                else:
                    QMessageBox.warning(self, "경고", "저장할 패스를 설정하세요.")
                    self.Button_model_creation.setEnabled(True)
            else:
                QMessageBox.warning(self, "경고", "학습할 이미지 패스를 설정하세요.")
                self.Button_model_creation.setEnabled(True)
        else:
            QMessageBox.warning(self, "경고", "모델의 이름이 없습니다.")
            self.Button_model_creation.setEnabled(True)

    def tab2_train_image_path_save(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "디렉토리 선택", options=options)
        if directory:
            self.train_image_path = directory
            self.Label_image_path.setText(directory)
            print(self.train_image_path)
        else:
            QMessageBox.warning(self, "경고", "잘못된 경로 입니다.")


    def tab2_train_save_path_save(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "디렉토리 선택", options=options)
        if directory:
            self.train_save_path = directory
            self.Label_save_path.setText(directory)
            print(self.train_save_path)
        else:
            QMessageBox.warning(self, "경고", "잘못된 경로 입니다.")

            
    def Label_update(self, value):  # 슬라이드바 값 표시
        self.Label_slider_value.setText(f"Value: {value}")

    def save_result_image(self, model_name, label):
        if not self.selected_directory:
            QMessageBox.warning(self, "경고", "저장할 디렉토리를 선택하세요.")
            return

        if label.pixmap() is not None:
            image = label.pixmap().toImage()
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 날짜와 시간을 YYYYMMDD_HHMMSS 형식으로 가져오기
            image_name = f"{current_datetime}_{model_name}_{os.path.basename(self.selected_image_path)}"
            save_path = os.path.join(self.selected_directory, image_name)
            if not image.save(save_path):
                QMessageBox.critical(self, "오류", "이미지를 저장하는 중 오류가 발생했습니다.")
            else:
                QMessageBox.information(self, "성공", "이미지가 성공적으로 저장되었습니다.")
        else:
            QMessageBox.warning(self, "경고", "저장할 이미지가 없습니다.")

    def reset_img(self):
        self.Label_img.clear()
        self.Label_img_2.clear()
        self.Label_img_3.clear()
        self.Label_img_name.clear()
        self.selected_image_path = None

    def open_directory(self):
        if self.selected_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.selected_directory))
        else:
            QMessageBox.warning(self, "경고", "디렉토리가 선택되지 않았습니다.")

    def directory_save(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "디렉토리 선택", options=options)
        if directory:
            self.selected_directory = directory
            self.Label_directory.setText(directory)
            print(self.selected_directory)

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "이미지 파일 (*.png *.jpg *.bmp *.gif)",
                                                    options=options)
        if image_path:
            self.selected_image_path = image_path
            image = cv2.imread(image_path)
            if image is not None:
                # OpenCV는 BGR 채널 순서를 사용하므로 RGB로 변환
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # QImage 생성
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                # QPixmap 생성 및 QLabel에 표시
                pixmap = QPixmap.fromImage(q_image)
                self.Label_img.setPixmap(pixmap)  # 이미지 크기 조절
                self.Label_img.setScaledContents(True)  # 이미지를 QLabel에 맞게 조절
                image_name = os.path.basename(image_path)
                self.Label_img_name.setText(f"선택된 이미지 파일: {image_name}")
            else:
                QMessageBox.warning(self, "오류", "이미지를 열 수 없습니다.")


    def model02_run(self):
        print(self.selected_image_path)
        if self.selected_image_path is not None:
            resultImage, results = model02.process_images(self.selected_image_path, self.model, self.class_labels)
            self.result_images['model02'] = resultImage
            self.setImageToLabel(resultImage, self.Label_img_2)
            print(results)
        else:
            QMessageBox.warning(self, "경고", "이미지가 선택되지 않았습니다.")

    def model03_run(self):
        print(self.selected_image_path)
        if self.selected_image_path is not None:
            resultImage, results = model03.process_images(self.selected_image_path, self.model, self.class_labels)
            self.result_images['model03'] = resultImage
            self.setImageToLabel(resultImage, self.Label_img_3)
            print(results)
        else:
            QMessageBox.warning(self, "경고", "이미지가 선택되지 않았습니다.")

    def setImageToLabel(self, image, label):
        # Convert the color from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qImg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap.scaled(label.size(), aspectRatioMode=True))

    def populateComboBox(self):
        self.comboBox_model.clear()  # 기존 항목들을 지우고 다시 추가
        directory = "./models"  # 파일이 있는 디렉토리 경로
        if os.path.exists(directory) and os.path.isdir(directory):
            files = os.listdir(directory)
            for file in files:
                if os.path.isfile(os.path.join(directory, file)):
                    self.comboBox_model.addItem(file)
                    self.model_dir = directory + "/" + file
                    print(self.model_dir)

        # 선택된 모델이 있으면 해당 모델 로드
        selected_model = self.comboBox_model.currentText()
        self.model_dir = os.path.join(directory, selected_model)
        if os.path.exists(self.model_dir):
            self.model = load_model(self.model_dir)
        else:
            self.model = None

    def onActivated(self, text):  # 선택된 모델 이름 출력
        self.Label_model.setText("선택된 모델: " + text)