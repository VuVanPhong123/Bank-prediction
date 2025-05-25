# Bank-prediction
---

#  Dự án Logistic Regression with Momentum

##  Mô tả

Đây là một dự án triển khai thuật toán **logistic regression** kết hợp với **momentum** để cải thiện tốc độ hội tụ. Dữ liệu được xử lý với các bước như: làm sạch, mã hóa biến phân loại, phát hiện và loại bỏ outliers, chuẩn hóa (scaling), và xử lý mất cân bằng dữ liệu bằng **SMOTE**. Mô hình được tự xây dựng từ đầu sử dụng `numpy`.

##  Mô hình chính

* Lớp `LogisticRegressionWithMomentum` tự cài đặt gồm:

  * **Regularization (L2)**
  * **Momentum**
  * **Early stopping** với `patience`
  * Giới hạn `weights` để tránh overflow
  * Khởi tạo ngẫu nhiên `weights`, `bias`

##  Cấu trúc dữ liệu

* **`train.csv`**: Dữ liệu huấn luyện
* **`test.csv`**: Dữ liệu kiểm thử
* Biến mục tiêu: `loan_status` (nhị phân)

##  Các bước xử lý

1. **Load dữ liệu**
2. **Xử lý missing values** và **categorical features** (dùng `LabelEncoder`)
3. **Feature selection** dựa trên Pearson correlation với `loan_status`
4. **Outlier detection** bằng IQR và z-score
5. **Feature scaling** với `StandardScaler`
6. **Oversampling** với **SMOTE**
7. **Train** mô hình bằng gradient descent with momentum
8. **Test** mô hình trên tập kiểm thử và đưa ra dự đoán

##  Thư viện sử dụng

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `imblearn` (SMOTE)

##  Hình ảnh minh họa

* **Correlation heatmap**
* **Boxplot** cho outlier analysis
* **Loss plot** nếu có vẽ `loss vs. epochs`

##  Cách sử dụng

```bash
# Cài thư viện cần thiết
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

# Chạy notebook
jupyter notebook logistic_regression_with_momentum.ipynb
```

##  Điểm nổi bật

* Mô hình **logistic regression** tự cài đặt từ đầu, không dùng thư viện ngoài như `sklearn`
* Có áp dụng **momentum** và **L2 regularization**
* **Early stopping** giúp tránh overfitting
* Xử lý **imbalanced dataset** bằng **SMOTE**
* Tiền xử lý dữ liệu kỹ lưỡng: từ xử lý thiếu dữ liệu đến feature selection và scaling


