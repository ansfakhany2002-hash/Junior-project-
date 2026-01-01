import cv2
import torch
import pytesseract
import numpy as np
import re  # لإضافة التنظيف باستخدام التعبيرات المنتظمة

# مسار Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# تحميل نموذج YOLOv5
import torch
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

def preprocess_plate(plate_img):
    # 1. التكبير (لزيادة الدقة)
    scale_factor = 3
    plate_img_resized = cv2.resize(plate_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 2. تحويل للرمادي
    gray = cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2GRAY)

    # 3. تحسين التباين (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    gray_clahe = clahe.apply(gray)

    # 4. إزالة الضوضاء (Gaussian Blur - أكثر نعومة من Median)
    denoised = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

    # 5. التحديد الثنائي باستخدام Otsu (مع INVERSE للحصول على نص أبيض على خلفية سوداء إذا لزم الأمر)
    # يمكن تجربة THRESH_BINARY + THRESH_OTSU إذا كانت اللوحة داكنة
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 6. التآكل والتمدد (لإغلاق الفجوات)
    kernel = np.ones((2, 2), np.uint8)
    processed_img = cv2.dilate(thresh, kernel, iterations=1)
    processed_img = cv2.erode(processed_img, kernel, iterations=1)

    return processed_img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    if len(results.xyxy[0]) > 0:
        for det in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, det[:4])
            plate_img = frame[y1:y2, x1:x2]

            # تطبيق المعالجة المسبقة
            processed_img = preprocess_plate(plate_img)

            # 🧠 إعدادات Tesseract (محدثة)
            custom_config = r'--oem 3 --psm 6 -l ara+eng -c tessedit_char_whitelist=0123456789أبجدسرحصطكعمناوةيل'

            num_text = pytesseract.image_to_string(processed_img, config=custom_config)

            # 🧹 التنظيف والمعالجة اللاحقة
            # السماح فقط بالأحرف المحددة في القائمة البيضاء + المسافات
            char_whitelist = r'0-9أبجدسرحصطكعمناوةيل'
            cleaned_text = re.sub(f'[^{char_whitelist}]', '', num_text)
            cleaned_text = cleaned_text.replace(" ", "").replace("\n", "").strip()

            # 🎨 عرض النتائج
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, cleaned_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Detected Plate:", cleaned_text)

            cv2.imshow("Processed Plate", processed_img)

    cv2.imshow("Car Plate Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()