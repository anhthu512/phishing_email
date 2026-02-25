import joblib
import os

MODEL_PATH = 'phishing_model.pkl'

def load_model_and_predict():
    # Kiểm tra xem file model có tồn tại không
    if not os.path.exists(MODEL_PATH):
        print("LỖI: Chưa tìm thấy file mô hình (phishing_model.pkl).")
        print("Vui lòng chạy 'python train_model.py' trước.")
        return

    # Load mô hình
    print("--- HỆ THỐNG PHÁT HIỆN EMAIL LỪA ĐẢO ---")
    print("Đang tải mô hình...")
    model = joblib.load(MODEL_PATH)
    print("Sẵn sàng!")
    
    while True:
        print("\n" + "="*40)
        email_text = input("Nhập nội dung email (hoặc gõ 'exit' để thoát): \n>> ")
        
        if email_text.lower() == 'exit':
            break
        
        if not email_text.strip():
            continue
            
        # Dự đoán
        # Vì đã dùng Pipeline lúc train, ta đưa thẳng text vào predict
        prediction = model.predict([email_text])[0]
        proba = model.predict_proba([email_text])[0]
        
        if prediction == 1:
            print(f"\n⚠️  CẢNH BÁO: ĐÂY LÀ EMAIL LỪA ĐẢO (PHISHING)!")
            print(f"Độ tin cậy: {proba[1]*100:.2f}%")
        else:
            print(f"\n✅ AN TOÀN: Email này có vẻ hợp lệ.")
            print(f"Độ tin cậy: {proba[0]*100:.2f}%")

if __name__ == "__main__":
    load_model_and_predict()