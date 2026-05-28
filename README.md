# Thị giác máy: Từ xử lý ảnh đến học sâu

Repository này cung cấp mã nguồn, dữ liệu và tài nguyên học tập đi kèm giáo trình **“Thị giác máy: Từ xử lý ảnh đến học sâu”**.

Nội dung của repository bao gồm:

* Các ví dụ minh họa bằng Python/OpenCV/PyTorch
* Notebook chạy trực tiếp trên Google Colab
* Ảnh, video và dữ liệu mẫu phục vụ thực hành
* Các ví dụ trực quan hóa thuật toán thị giác máy
* Các mô hình học sâu minh họa cho bài toán phân loại, phát hiện, phân đoạn và tái tạo ảnh

Giáo trình được xây dựng theo hướng kết nối giữa:

* Xử lý ảnh số truyền thống
* Trích rút đặc trưng thủ công
* Các bài toán thị giác máy cổ điển
* Học sâu và Transformer trong thị giác máy hiện đại

---

# Ví dụ minh họa bằng notebook

| Mục     | Tên bài                                | Colab                                                                                                                                              |
| ------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.3.4   | Minh họa biểu diễn dữ liệu 3D          | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/3.3.4_Minh_hoa_bieu_dien_du_lieu_3D.ipynb)            |
| 3.4.3.3 | Hiệu chỉnh camera                      | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/3.4.3.3_Hieu_chinh_camera.ipynb)                      |
| 4.1.1   | Lọc tuyến tính                         | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/4.1.1_Loc_tuyen_tinh.ipynb)                           |
| 4.1.2   | Làm mượt ảnh                           | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/4.1.2_Lam_muot_anh.ipynb)                             |
| 4.1.4.4 | Tách cạnh Sobel                        | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/4.1.4.4_Tach_canh_Sobel.ipynb)                        |
| 4.2.1   | Biến đổi Fourier                       | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/4.2.1_Bien_doi_Fourier.ipynb)                         |
| 4.2.3   | Lọc ảnh trong miền tần số              | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/4.2.3_Loc_anh_trong_mien_tan_so.ipynb)                |
| 4.3.1.1 | Khử nhiễu cộng                         | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/4.3.1.1_Khu_nhieu_cong.ipynb)                         |
| 4.3.1.2 | Khử nhiễu xung                         | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/4.3.1.2_Khu_nhieu_xung.ipynb)                         |
| 5.2.1   | Tách cạnh Canny                        | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/5.2.1_Tach_canh_Canny.ipynb)                          |
| 5.4.4   | Trích xuất đặc trưng SIFT và HoG       | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/5.4.4_Trich_xuat_dac_trung_SIFT_va_HoG.ipynb)         |
| 6.2     | Căn chỉnh ảnh                          | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/6.2_Can_chinh_anh.ipynb)                              |
| 6.7.1   | Tái tạo 3D từ ảnh stereo               | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/6.7.1_Tai_tao_3D_tu_anh_stereo.ipynb)                 |
| 7.2.1   | Xây dựng và huấn luyện mạng LeNet-5    | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/7.2.1_Xay_dung_va_huan_luyen_mang_LeNet5.ipynb)       |
| 7.2.2.1 | Trực quan hóa Self-Attention trong ViT | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/7.2.2.1_Truc_quan_hoa_Self_Attention_trong_ViT.ipynb) |
| 8.4     | Phân đoạn ảnh với U-Net                | [Open in Colab](https://colab.research.google.com/github/lthavnu/cv-book/blob/main/notebooks/8.4_Phan_doan_anh_UNet.ipynb)                         |

---

# Dữ liệu và tài nguyên cho các dự án thực hành

| Dự án   | Nội dung                             | Dữ liệu/Tài nguyên                                         |
| ------- | ------------------------------------ | ---------------------------------------------------------- |
| Dự án 1 | Xử lý ảnh cơ bản                     | Ảnh mẫu, histogram, ảnh nhiễu                              |
| Dự án 2 | Tăng cường chất lượng ảnh            | Bộ dữ liệu ảnh mờ, ảnh tối, ảnh nhiễu                      |
| Dự án 3 | Phát hiện biên và đặc trưng          | Ảnh cảnh tự nhiên, ảnh texture                             |
| Dự án 4 | Căn chỉnh và ghép ảnh                | Ảnh panorama, ảnh stereo                                   |
| Dự án 5 | Theo dõi đối tượng                   | Video theo dõi chuyển động                                 |
| Dự án 6 | Phân loại ảnh bằng CNN               | CIFAR-10, Tiny ImageNet                                    |
| Dự án 7 | Phát hiện và phân đoạn đối tượng     | Pascal VOC, MS COCO                                        |
| Dự án 8 | Transformer và thị giác máy hiện đại | ViT, Swin Transformer, MaxViT notebook và pretrained model |

---

# Yêu cầu môi trường

Các notebook được thiết kế để chạy trên:

* Google Colab
* Python 3.10+
* OpenCV
* NumPy
* Matplotlib
* PyTorch
* torchvision
* timm
* scikit-image
* scikit-learn

---

# Tác giả

**Lê Thanh Hà**
Giảng viên và nghiên cứu viên trong lĩnh vực thị giác máy, xử lý ảnh và học sâu.

---
