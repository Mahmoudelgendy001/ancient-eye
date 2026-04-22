# 𓂀 Ancient Eye — Egyptian Monument & Pharaoh Classifier

> Upload a photo of any Egyptian monument or pharaoh — the model identifies it instantly and tells you its full story in Arabic.

---

## Features

- 🎯 **96.7% accuracy** on 21 ancient Egyptian categories
- 🧠 **EfficientNetB0** Transfer Learning (2-phase training)
- ⚠️ **Smart threshold** — warns user if image is outside model scope
- 📖 **Rich Arabic info** — story, historical significance, fun facts
- 🏛️ **Categories dropdown** — shows all supported monuments
- 🌙 **Dark pharaonic UI** — custom design inspired by ancient Egypt

---

## Demo

![App Demo](demo.gif)

---

## Dataset

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/monaabdelrazek/finaldataset/data?select=FinalDataset)

| Split | Images | Categories |
|---|---|---|
| Train | 14,159 | 21 |
| Val | 4,062 | 21 |
| Test | 2,035 | 21 |
| **Total** | **20,256** | **21** |

---

## Categories

| Pharaohs & Queens | Monuments & Artifacts |
|---|---|
| Akhenaten ☀️ | Great Sphinx 🦁 |
| Nefertiti 💎 | Abu Simbel 🌅 |
| Ramesses II 👑 | Pyramid of Djoser 🏗️ |
| Hatshepsut 👸 | Khafre Pyramid 🔺 |
| Thutmose III ⚔️ | Menkaure Pyramid 🔻 |
| Amenhotep III 🏛️ | Bent Pyramid 📐 |
| Mask of Tutankhamun 🏺 | Colossi of Memnon 🗿 |
| Goddess Isis 👑 | Temple of Hatshepsut 🌿 |
| Tutankhamun & Ankhesenamun 💑 | Temple of Isis, Philae 🌊 |
| Statue of King Djoser 🗿 | Temple of Kom Ombo 🐊 |
| | Ramesseum 🏛️ |

---

## Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB0 (pretrained ImageNet — frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU) → Dropout(0.5)
    ↓
Dense(256, ReLU) → Dropout(0.3)
    ↓
Dense(21, Softmax)
```

**Training Strategy:**
- Phase 1 — Train head only (15 epochs, lr=1e-3)
- Phase 2 — Fine-tune last 30 layers (15 epochs, lr=1e-5)
- Class weights to handle imbalanced data
- Confidence threshold: 65%

---

## Model Download

The model file (`ancienteye.h5`) is not included in this repo.

**Download:** *(https://drive.google.com/drive/folders/1I_7hGVPP9LocVKYxinFdfmtFzoFtu-Y0?usp=sharing)*

After downloading, place it here:
```
ancient_eye/
└── models/
    └── ancienteye.h5
```

---

## Project Structure

```
ancient_eye/
├── app.py                  # Flask backend + prediction logic
├── train.py                # Model training script
├── requirements.txt
├── Procfile
├── templates/
│   └── index.html          # Full frontend (HTML/CSS/JS)
└── models/
    ├── class_indices.json  # Label mapping
    ├── history.json        # Training history
    └── test_results.json   # Final test accuracy
```

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/Mahmoudelgendy001/ancient-eye.git
cd ancient-eye

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model and place in models/ folder

# 4. Run
python app.py

# Open http://localhost:7860
```

---

## Limitations

- Works best on **clear, front-facing images** — blurry or angled photos reduce accuracy
- The model only recognizes the **21 trained categories** — other Egyptian artifacts will trigger a warning
- Some categories look visually similar (e.g. Ramesses II statues vs King Thutmose III) which can cause confusion
- Performance may drop on **heavily edited or illustrated images**

---

## Future Improvements

- [ ] Expand dataset to 50+ Egyptian categories
- [ ] Add English support alongside Arabic (multilingual output)
- [ ] Deploy a full production version (scalable backend + cloud hosting)
- [ ] Build a mobile application for easier access
- [ ] Integrate Augmented Reality (AR) to display monuments information interactively
- [ ] Implement user feedback system to improve predictions over time
- [ ] Search by text (e.g., "Show me temples of Ramses II")

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![EfficientNet](https://img.shields.io/badge/EfficientNetB0-Transfer%20Learning-green)

---

## Author

**Mahmoud** — Passionate about AI & Egyptian History

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/mahmoud-ahmed-elgendy/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Mahmoudelgendy001)

---

*If you found this project useful, please consider giving it a ⭐*
