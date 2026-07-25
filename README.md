# ICAI-FAI 2026: The 2nd International Conference on AI - AI Native for University

[![Conference Website](https://img.shields.io/badge/Website-icai.cmcu.edu.vn-blue)](https://icai.cmcu.edu.vn)
[![Peer Review Service](https://img.shields.io/badge/Peer--Review-Microsoft%20CMT-0078D4)](https://cmt3.research.microsoft.com)
[![IEEE Style Template](https://img.shields.io/badge/Template-IEEE%20Conference-orange)](https://www.ieee.org/conferences/publishing/templates)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official Research Artifact Repository and LaTeX Paper Source for **The Second International Conference on AI: AI Native for University (ICAI-2026 / ICAI-FAI 2026)**.

---

## 📢 Thông báo Mời Viết Bài / Call for Papers (ICAI-2026)

**Hội nghị quốc tế lần thứ hai về Trí tuệ nhân tạo: Kiến tạo Đại học AI Native**  
*(The 2nd International Conference on AI: AI Native for University – ICAI-2026 / ICAI-FAI 2026)*

Trí tuệ nhân tạo (AI) đang trở thành công nghệ nền tảng, tạo động lực cho đổi mới sáng tạo, nâng cao năng suất và thúc đẩy nhiều chuyển đổi sâu rộng trong kinh tế, công nghiệp, đời sống xã hội, khoa học và giáo dục. Bên cạnh những cơ hội lớn, việc nghiên cứu, phát triển, ứng dụng và quản trị AI cũng đặt ra yêu cầu cấp thiết về dữ liệu, an toàn, đạo đức, độ tin cậy và khả năng triển khai bền vững trong thực tiễn.

Để góp phần thúc đẩy trao đổi học thuật, hợp tác nghiên cứu, đổi mới sáng tạo và ứng dụng AI, **Trường Đại học CMC** phối hợp cùng **Trường Đại học Steinbeis (Đức)** và **Trường Quốc tế Sau đại học – Đại học Thanh Hoa (Trung Quốc)** tổ chức Hội nghị khoa học quốc tế lần thứ 2 về Trí tuệ nhân tạo AI Tiên phong 2026 (ICAI-FAI 2026).

---

## 🗓️ Các Mốc Thời Gian Quan Trọng / Important Dates

* **Hạn nộp toàn văn bài viết (Full Paper Submission Deadline):** 30/09/2026
* **Thông báo kết quả chấp nhận bài (Acceptance Notification):** 30/10/2026
* **Hạn nộp bản hoàn thiện sau phản biện (Camera-Ready Submission):** 15/11/2026
* **Thời gian tổ chức Hội nghị (Conference Date):** **03/12/2026** tại Hà Nội, Việt Nam.

---

## 🌐 Thông Tin Liên Hệ & Đăng Ký / Contact & Official Links

* **Website chính thức của Hội nghị:** [https://icai.cmcu.edu.vn](https://icai.cmcu.edu.vn)
* **Email liên hệ Ban Tổ chức:** [ost@cmcu.edu.vn](mailto:ost@cmcu.edu.vn)
* **Đơn vị phụ trách:** TS. Lê Hữu Tôn, Phụ trách Phòng Khoa học và Công nghệ, Trường Đại học CMC (SĐT/Zalo: 0385794025).
* **Xuất bản (Proceedings):** Kỷ yếu Hội nghị được xuất bản với mã số **ISBN**. Các bài viết xuất sắc sẽ được giới thiệu đăng trên các tạp chí khoa học uy tín trong nước và quốc tế.

---

## 📄 IEEE Conference Paper Templates

Authors are required to prepare manuscripts using official IEEE conference paper templates:

1. **Microsoft Word Template:** `Conference-template-a4`
2. **LaTeX Template (Included in this repo):** `conference-latex-template.zip` & `IEEEtranBST2.zip`
3. **LaTeX Main Source File:** [icaifai.tex](file:///d:/tu_projects/LatexProject/icai-fai2026/icaifai.tex) (Formatted strictly to IEEE standard).
4. **Official IEEE Template Reference:** [https://www.ieee.org/conferences/publishing/templates](https://www.ieee.org/conferences/publishing/templates)

---

## 🔬 Featured Paper & Research Artifact

### Paper Title
**"AI-Native Autonomous Security Platform: An Event-Driven Edge-Cloud Architecture for University Networks"**

### Authors & Affiliations
* **Van A. Nguyen** - *Department of Computer Science & Cybersecurity, CMC University, Hanoi, Vietnam* (`ost@cmcu.edu.vn`)
* **Hans Schmidt** - *School of Advanced AI Systems, Steinbeis University, Berlin, Germany* (`h.schmidt@steinbeis.de`)
* **Wei Zhang** - *Shenzhen International Graduate School, Tsinghua University, Shenzhen, China* (`zhang.wei@sz.tsinghua.edu.cn`)

### Paper Abstract
Modern university campus networks present high device heterogeneity, open Wi-Fi topologies, and massive concurrent telemetry streams. Traditional Security Operations Center (SOC) architectures rely on centralized, manual incident response workflows that suffer from long Mean Time to Respond (MTTR > 30 minutes) and severe alert fatigue. In this paper, we propose an **AI-Native Autonomous Security Platform** engineered for university environments. By decoupling detection into lightweight edge anomaly classifiers (Isolation Forests) and streaming high-confidence events over a Redis Stream bus to a cloud-based dynamic risk resolver and automated SOAR engine, our platform achieves sub-10ms event-to-mitigation latency. Experimental results demonstrate an end-to-end response latency of 4.2 ms, a 99.9% reduction in MTTR compared to manual triage, an F1-score of 0.94 at optimal decision threshold ($\tau=0.65$), and minimal edge CPU overhead (14.3% utilization at 10,000 events/sec).

---

## 💻 Research Artifact Directory (`AI-Native-Security-Platform/`)

The repository contains the complete reproducible research software and benchmark code supporting the paper:

```
icai-fai2026/
├── README.md                           # Main Conference & Paper Overview
├── icaifai.tex                         # Official IEEE Standard LaTeX Source
├── IEEEtran.cls                        # IEEE Class formatting specification
├── IEEEtran.bst                        # IEEE Bibliography style specification
├── conference-latex-template.zip       # Official IEEE LaTeX zip template
├── IEEEtranBST2.zip                    # Official IEEE BibTeX zip template
└── AI-Native-Security-Platform/        # Complete Reproducible Research Artifact
    ├── README.md                       # Sub-repository architecture docs
    ├── requirements.txt                # Dependencies (numpy, scikit-learn, redis)
    ├── docker-compose.yml              # Microservice orchestration spec
    ├── paper/                          # Paper sources, references & figures
    ├── architecture/                   # PlantUML & DrawIO diagrams
    ├── docs/                           # Academic design specifications
    ├── prototype/                      # Edge Detector, Identity Fusion, SOAR
    └── evaluation/                     # Reproducible benchmark experiments
```

### Reproducing Benchmark Experiments
```bash
cd AI-Native-Security-Platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python evaluation/scripts/generate_figures.py
```

---

## 🙏 Acknowledgement & CMT Service

The **Microsoft CMT** service was used for managing the peer-reviewing process for this conference. This service was provided for free by Microsoft, who bore all expenses including costs for Azure cloud services as well as software development and support.
