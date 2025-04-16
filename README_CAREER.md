# 🎯 Career Compass – A Career Recommendation System

Career Compass is a web-based career recommendation system that helps students and individuals discover the most suitable career paths based on their interests, skills, and preferences. The system uses a machine learning model (Random Forest) trained on a custom dataset to predict career roles and present meaningful insights.

---

## 🚀 Features

- User Registration and Login System
- Interactive Career Assessment Questionnaire
- Machine Learning-based Career Recommendations
- Dashboard with History Tracking
- Visualizations of User Trends and Preferences
- Smart Model Loading via Google Drive

---

## 🧠 Tech Stack

- **Frontend:** HTML, CSS (Tailwind), JavaScript, Jinja2
- **Backend:** Python, Flask
- **ML:** Scikit-learn (Random Forest Classifier)
- **Database:** SQLite via SQLAlchemy ORM

---

## 📦 Project Structure

```
Career_Compass/
├── app.py
├── career_recommender.py
├── templates/
│   ├── index.html, login.html, signup.html, questionnaire.html, results.html, dashboard.html, history.html
├── static/css/
├── instance/
├── models/
├── career_recommendation_model.pkl  ← (downloaded at runtime)
├── expanded_career_interests_original_dataset.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/saigaganpareda/career_compass.git
cd career_compass
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

### 5. Access in Your Browser

Open `http://127.0.0.1:5000` in your browser.

---

## 🧠 Model Download Note

⚠️ The machine learning model file `career_recommendation_model.pkl` is not stored in this repo due to GitHub size limits.

Instead, it will be **automatically downloaded** from Google Drive at runtime using the link below:

🔗 [Download model at runtime](https://drive.google.com/uc?export=download&id=1NFPrZoO-IPO3lprRltGe2qELleefxFwz)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Scikit-learn for machine learning tools
- Flask for backend web development
- TailwindCSS for styling
- Chart.js for visualization
