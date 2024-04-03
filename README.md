# Data Mining Project - Nhóm 4

News Classification: Sử dụng Machine Learning và Deep Learning models để dự đoán nhãn của các bài báo tiếng anh từ nhiều nguồn ví dụ như: [NBC](https://www.nbcnews.com/)

## Team Members

1.Vũ Minh Tiến - 22022645

2.Phạm Thị Kim Huệ - 22022540

3.Phạm Quang Vinh - 22022648

### Installation Requirements:
    pip install -r requirements.txt

## Folder Structure:
  * train.py: Chứa các class để huấn luyện các mô hình khác nhau.
    
  * predict.py: Sử dụng các class từ train.py để dự đoán nhãn của bài báo.
    
  * crawl.py: Bao gồm các script để crawl và thu thập dữ liệu từ các nguồn khác nhau, như các trang web tin tức nước ngoài.
    
  * preprocessing.py: Tiền xử lý dữ liệu trước khi cho vào quá trình training.
    
  * data.csv: Tập dữ liệu chứa khoảng ~1700 bài báo. 500 hàng đầu tiên được gán nhãn bằng tay để kiểm tra, và dữ liệu còn lại được sử dụng cho huấn luyện và phát triển.

### References:
[1] [TEXT CLASSIFICATION: A PERSPECTIVE OF DEEP LEARNING METHODS](https://arxiv.org/pdf/2309.13761.pdf)

[2] [News Classification Using Machine Learning](https://www.ijisrt.com/assets/upload/files/IJISRT21MAY852.pdf)

[3] [Online news classification using Deep Learning Technique](https://www.irjet.net/archives/V3/i10/IRJET-V3I10107.pdf)

[4] [Deep Learning Based Text Classification: A Comprehensive Review](https://arxiv.org/pdf/2004.03705.pdf)

[5] [Topic Classification of Online News Articles Using Optimized Machine Learning Models](https://www.mdpi.com/2073-431X/12/1/16)
