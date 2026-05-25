# Thực nghiệm ModernBERT trên GLUE NLI và ANLI Round 1

## 1. Tổng quan

Báo cáo này trình bày quá trình tái hiện một phần kết quả thực nghiệm của paper **ModernBERT** trên nhóm tác vụ **Natural Language Inference (NLI)** trong benchmark **GLUE**, đồng thời đánh giá khả năng tổng quát hóa của mô hình trên một dataset ngoài là **ANLI Round 1**.

Trong phần GLUE, chúng tôi thực nghiệm với ba tác vụ thuộc nhóm NLI:

- **MNLI**
- **QNLI**
- **RTE**

Mục tiêu chính là tái hiện kết quả gần với **Table 5** trong paper ModernBERT. Các mô hình được sử dụng gồm:

- `answerdotai/ModernBERT-base`
- `answerdotai/ModernBERT-large`

Ngoài ra, với tác vụ **RTE**, chúng tôi không chỉ fine-tune trực tiếp từ checkpoint pretrained ban đầu, mà còn thực hiện đúng theo mô tả trong paper: **RTE được train starting from MNLI checkpoint**. Cụ thể, mô hình được fine-tune trên MNLI trước, lưu checkpoint tốt nhất, sau đó dùng checkpoint này để tiếp tục fine-tune trên RTE. Cách làm này đặc biệt quan trọng vì RTE là dataset nhỏ, dễ gây dao động kết quả và dễ overfit nếu fine-tune trực tiếp từ pretrained checkpoint.

Sau phần tái hiện GLUE, chúng tôi tiếp tục đánh giá các mô hình trên **ANLI Round 1**. Đây là dataset NLI ngoài paper, có cùng format premise-hypothesis giống MNLI nhưng được xây dựng theo hướng adversarial, do đó thường khó hơn các dataset NLI tiêu chuẩn.

---

## 2. Mô hình và dataset

### 2.1. Mô hình

Các checkpoint ModernBERT được lấy trực tiếp từ Hugging Face:

| Mô hình | Checkpoint |
|---|---|
| ModernBERT-base | `answerdotai/ModernBERT-base` |
| ModernBERT-large | `answerdotai/ModernBERT-large` |

### 2.2. Dataset GLUE

Các dataset GLUE được load bằng thư viện `datasets`:

```python
load_dataset("nyu-mll/glue", task_name)
```

Các task được thực nghiệm:

| Task | Input | Nhãn |
|---|---|---|
| MNLI | premise + hypothesis | entailment / neutral / contradiction |
| QNLI | question + sentence | entailment / not_entailment |
| RTE | sentence1 + sentence2 | entailment / not_entailment |

Vì nhãn của GLUE test set không được công khai, toàn bộ kết quả tái hiện được đánh giá trên **validation/dev set**, tương ứng với setup của Table 5 trong paper.

### 2.3. Dataset ngoài: ANLI Round 1

Sau khi thực nghiệm trên GLUE, chúng tôi sử dụng **ANLI Round 1** làm dataset ngoài để kiểm tra khả năng tổng quát hóa của mô hình.

ANLI cũng là một bài toán NLI với format:

```text
Premise + Hypothesis → Label
```

Tuy nhiên, khác với MNLI, ANLI được xây dựng theo hướng **adversarial**, tức là các mẫu dữ liệu được thiết kế để gây khó cho các mô hình NLI. Vì vậy, ANLI thường khó hơn và phù hợp để đánh giá khả năng generalization ngoài benchmark gốc.

---

## 3. Thiết lập thực nghiệm

Các thực nghiệm được triển khai bằng:

- `transformers`
- `datasets`
- `evaluate`
- PyTorch

Metric được sử dụng cho MNLI, QNLI, RTE và ANLI là **accuracy**.

### 3.1. Batch size

Paper không công bố batch size trong Table 6, nên batch size được điều chỉnh theo tài nguyên GPU khi chạy trên Kaggle.

Đối với GLUE:
| Mô hình | Train batch size |
|---|---:|
| ModernBERT-base | 16 |
| ModernBERT-large MNLI | 4 |
| ModernBERT-large RTE | 8 |

Đối với ANLI Round 1:
| Thiết lập | Train batch size | Gradient accumulation |
|---|---:|---:|
| Base / Base + MNLI checkpoint | 8 | 4 |
| Large / Large + MNLI checkpoint | 8 | 4 |

Với ModernBERT-large, do mô hình lớn hơn và tốn VRAM hơn, batch size được giảm xuống 4 để tránh lỗi CUDA out of memory. Trong một số trường hợp, có thể dùng thêm gradient accumulation để tăng effective batch size mà không vượt quá giới hạn bộ nhớ GPU.

---

## 4. Hyperparameter fine-tuning

Các tham số fine-tuning được đặt theo **Table 6** trong paper ModernBERT. Cụ thể, chúng tôi giữ nguyên learning rate, weight decay và số epoch cho từng task.

### 4.1. ModernBERT-base

| Task | Learning rate | Weight decay | Epochs |
|---|---:|---:|---:|
| MNLI | `5e-5` | `5e-6` | 1 |
| QNLI | `8e-5` | `5e-6` | 2 |
| RTE | `5e-5` | `1e-5` | 3 |

### 4.2. ModernBERT-large

| Task | Learning rate | Weight decay | Epochs |
|---|---:|---:|---:|
| MNLI | `3e-5` | `1e-5` | 1 |
| QNLI | `3e-5` | `5e-6` | 2 |
| RTE | `5e-5` | `8e-6` | 3 |

### 4.3. Quy trình fine-tuning cho RTE

Ban đầu, khi fine-tune trực tiếp ModernBERT trên RTE, kết quả thấp hơn đáng kể so với paper. Sau khi kiểm tra lại mô tả thực nghiệm, chúng tôi nhận thấy paper sử dụng chiến lược **intermediate fine-tuning**: RTE checkpoint được train bắt đầu từ checkpoint MNLI.

Do đó, quy trình đúng cho RTE là:

```text
ModernBERT pretrained checkpoint
        ↓
Fine-tune trên MNLI
        ↓
Lưu best MNLI checkpoint
        ↓
Khởi tạo mô hình RTE từ best MNLI checkpoint
        ↓
Fine-tune trên RTE
        ↓
Evaluate trên RTE validation/dev set
```

Vì MNLI có 3 nhãn còn RTE có 2 nhãn, classification head cần được khởi tạo lại khi chuyển từ checkpoint MNLI sang RTE.

---

## 5. Kết quả tái hiện trên GLUE NLI

Bảng dưới đây so sánh kết quả thực nghiệm của chúng tôi với kết quả trong Table 5 của paper ModernBERT.

| Task | ModernBERT-base Paper | ModernBERT-base Ours | Chênh lệch | ModernBERT-large Paper | ModernBERT-large Ours | Chênh lệch |
|---|---:|---:|---:|---:|---:|---:|
| MNLI | 89.1 | 88.70 | -0.40 | 90.8 | 90.62 | -0.18 |
| QNLI | 93.9 | 93.46 | -0.44 | 95.2 | 94.91 | -0.29 |
| RTE | 87.4 | 87.73 | +0.33 | 92.1 | 88.81 | -3.29 |

Nhìn chung, kết quả tái hiện trên MNLI và QNLI khá gần với kết quả được báo cáo trong paper. Với ModernBERT-base, kết quả RTE sau khi sử dụng MNLI checkpoint thậm chí cao hơn nhẹ so với paper. Tuy nhiên, với ModernBERT-large trên RTE, kết quả vẫn thấp hơn paper khoảng 3.29 điểm. Nguyên nhân có thể đến từ sự khác biệt về seed, batch size, GPU, early stopping, hoặc một số chi tiết triển khai không được mô tả đầy đủ trong paper.

---

## 6. Thực nghiệm trên ANLI Round 1

Sau khi tái hiện kết quả trên GLUE NLI, chúng tôi tiếp tục đánh giá trên **ANLI Round 1**. Mục tiêu là kiểm tra xem mô hình ModernBERT đã học tốt từ các task NLI trong GLUE có thể tổng quát hóa sang một dataset khó hơn hay không.

Chúng tôi thực nghiệm 4 thiết lập:

1. ModernBERT-base fine-tune trực tiếp trên ANLI Round 1
2. ModernBERT-base khởi tạo từ best MNLI checkpoint, sau đó fine-tune trên ANLI Round 1
3. ModernBERT-large fine-tune trực tiếp trên ANLI Round 1
4. ModernBERT-large khởi tạo từ best MNLI checkpoint, sau đó fine-tune trên ANLI Round 1

### 6.1. Kết quả ANLI Round 1

| Thiết lập mô hình | Accuracy trên ANLI Round 1 |
|---|---:|
| ModernBERT-base | 47.60% |
| ModernBERT-base + MNLI checkpoint | 58.00% |
| ModernBERT-large | 60.70% |
| ModernBERT-large + MNLI checkpoint | 67.50% |

---

## 7. Phân tích kết quả

Kết quả trên ANLI Round 1 cho thấy checkpoint MNLI có tác động rất rõ rệt đến hiệu năng của mô hình trên dataset ngoài.

Với **ModernBERT-base**, accuracy tăng từ **47.60%** lên **58.00%** khi sử dụng MNLI checkpoint, tương đương mức cải thiện tuyệt đối **10.40 điểm phần trăm**.

Với **ModernBERT-large**, accuracy tăng từ **60.70%** lên **67.50%**, tương đương mức cải thiện tuyệt đối **6.80 điểm phần trăm**.

Điều này cho thấy intermediate fine-tuning trên MNLI giúp mô hình học được biểu diễn tốt hơn cho bài toán NLI, từ đó transfer tốt hơn sang ANLI Round 1. Vì ANLI được xây dựng theo hướng adversarial, việc accuracy thấp hơn so với GLUE là hợp lý. ANLI không chỉ yêu cầu mô hình nhận diện quan hệ entailment đơn giản, mà còn chứa nhiều ví dụ được thiết kế để đánh lừa các mô hình NLI thông thường.

Kết quả tốt nhất đạt được là **67.50%** với thiết lập **ModernBERT-large + MNLI checkpoint**. Điều này cho thấy cả hai yếu tố đều quan trọng:

- mô hình lớn hơn có khả năng biểu diễn tốt hơn;
- checkpoint đã được fine-tune trên MNLI giúp mô hình thích nghi tốt hơn với bài toán NLI.

---

## 8. Kết luận

Trong thực nghiệm này, chúng tôi đã tái hiện một phần kết quả của ModernBERT trên nhóm task NLI của GLUE, bao gồm MNLI, QNLI và RTE. Các kết quả trên MNLI và QNLI khá gần với Table 5 trong paper. Với RTE, việc sử dụng best MNLI checkpoint là yếu tố quan trọng để đưa kết quả về gần với paper, đặc biệt vì RTE là dataset nhỏ và dễ dao động.

Bên cạnh đó, chúng tôi đánh giá thêm trên ANLI Round 1 như một dataset ngoài paper. Kết quả cho thấy các mô hình khởi tạo từ MNLI checkpoint đều vượt trội hơn so với các mô hình fine-tune trực tiếp từ pretrained checkpoint. Điều này xác nhận vai trò của intermediate fine-tuning trong việc cải thiện khả năng tổng quát hóa của ModernBERT trên các bài toán NLI khó hơn.

Nhìn chung, ModernBERT cho kết quả ổn định trên các task NLI của GLUE và có khả năng transfer tương đối tốt sang ANLI, đặc biệt khi sử dụng chiến lược fine-tuning thông qua MNLI checkpoint.
