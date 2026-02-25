import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cấu hình đường dẫn
DATA_PATH = 'data/phishing_email.csv'
MODEL_PATH = 'phishing_model.pkl'
IMG_PATH = 'confusion_matrix.png' # Tên file ảnh sẽ lưu

def train_v2():
    print("--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH (V2 - CHUYÊN SÂU) ---")
    
    # 1. Đọc và xử lý
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy {DATA_PATH}")
        return

    df = df.dropna()
    df = df[df['Email Text'] != 'empty']
    
    # Gán nhãn: Safe=0, Phishing=1
    df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['Email Text'], df['label'], test_size=0.2, random_state=42
    )
    
    # 2. Pipeline
    print("Đang training...")
    model = make_pipeline(
        TfidfVectorizer(stop_words='english', max_features=5000),
        MultinomialNB()
    )
    model.fit(X_train, y_train)
    
    # 3. Dự đoán
    y_pred = model.predict(X_test)
    
    # 4. Đánh giá chi tiết
    acc = accuracy_score(y_test, y_pred)
    print(f"\nĐỘ CHÍNH XÁC: {acc*100:.2f}%")
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
    
    # 5. VẼ BIỂU ĐỒ CONFUSION MATRIX (PHẦN MỚI)
    print(f"Đang vẽ và lưu biểu đồ vào '{IMG_PATH}'...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Phishing'], 
                yticklabels=['Safe', 'Phishing'])
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.savefig(IMG_PATH) # Lưu ảnh vào đĩa
    print("   - Đã lưu ảnh xong!")
    
    # 6. Lưu model
    joblib.dump(model, MODEL_PATH)
    print(f"Đã lưu mô hình tại: {MODEL_PATH}")

if __name__ == "__main__":
    train_v2()