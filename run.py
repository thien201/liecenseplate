from cv2 import cv2
import numpy as np
from train import get_model

my_model = get_model()
# Load Weights in computer
checkpoint = my_model.load_weights(r"E:\Pycharm\NhanDangBienSoXe\Model\weights--01--1.00.hdf5")


def convert(candidates):
    first_line = []
    second_line = []

    for candidate, coordinate in candidates:
        if coordinate[0] < 100:
            first_line.append((candidate, coordinate[1]))
        elif coordinate[0] > 100:
            second_line.append((candidate, coordinate[1]))

    def take_second(s):
        return s[1]

    first_line = sorted(first_line, reverse=False, key=take_second)
    second_line = sorted(second_line, reverse=False, key=take_second)

    if len(second_line) == 0:
        license_plate = "".join([str(ele[0]) for ele in first_line])
    else:
        license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join(
            [str(ele[0]) for ele in second_line])

    return license_plate


def run(model, image_file=None):
    global x0, y0, w0, h0, a, b

    labels = ["0", "1", "2", "3", "4", "5", "6", "7",
              "8", "9", "A", "B", "C", "D", "E", "F",
              "G", "H", "K", "L", "M", "N", "P", "R",
              "S", "T", "U", "V", "X", "Y", "Z"]

    if image_file is not None:
        capture = cv2.VideoCapture(image_file)
    else:
        capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            continue

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, img_thre = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        cont, _ = cv2.findContours(img_thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cont:
            x0, y0, w0, h0 = cv2.boundingRect(contour)
            area = w0 * h0

            if 4000 <= area <= 5000:
                cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0))
                a, b, c, d = x0, y0, w0, h0
                img_crop = frame[y0:y0 + h0, x0:x0 + w0]
                img_resize = cv2.resize(img_crop, (500, 500))
                # Save Cropped plate
                cv2.imwrite(r"E:\Pycharm\NhanDangBienSoXe\Cropped_Image\Biensoxe_crop.jpg", img_crop)

        img_resize_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        ret, frame1 = cv2.threshold(img_resize_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(frame1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        characters = []
        coordinates = []
        candidates = []
        result_idx = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            aspect_ratio = w / float(h)

            if w < h and (0.1 <= aspect_ratio <= 0.5) and h >= 150:
                img = img_resize[y:y + h, x:x + w]
                image_test = cv2.resize(img, dsize=(128, 128))
                image_test = np.expand_dims(image_test, axis=0)

                predict = model.predict(image_test)
                idx = np.argmax(predict[0])
                print("Number: ", labels[idx], predict[0])
                print(np.max(predict[0], axis=0))

                if np.max(predict[0]) >= 0.8:
                    character = labels[idx]
                    characters.append(character)
                    result_idx.append(idx)
                    coordinates.append((y, x))

        for i in range(len(result_idx)):
            candidates.append((characters[i], coordinates[i]))

        string = convert(candidates)

        cv2.putText(frame, string, (a, b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Image file's directory in computer
    run(my_model, image_file=r"E:\Pycharm\NhanDangBienSoXe\TestImage\0251_07090_b.jpg")
