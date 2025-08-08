🏢 Lead Generation Dashboard

A web application for managing company leads.
It includes:

* **Lead Scoring** with a trained `RandomForestClassifier`
* **Company Finder** with duplicate detection
* **Interactive Dashboard** showing lead statistics and charts
* **CSV dataset-based search & filtering**

## 📦 Features

* **Company Search**: Filter companies by industry and location.
* **Lead Management**: Add, view, and remove leads.
* **Duplicate Detection**: Prevents duplicate leads with similarity checking.
* **Lead Scoring Model**: Predicts lead quality using machine learning.
* **Dashboard Analytics**: Displays total leads, average lead score, distribution charts, and top industries.

## ⚙️ Requirements

* Python **3.8+**
* `pip` package manager

## 📥 Installation

1. **Clone the repository**

   git clone https://github.com/RainerYesaya/Saasquatch-Leads-Rainer.git

2. **Create a virtual environment (optional but recommended)**

   ```
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare dataset**

   * Place your `Companies.csv` file in the `data/` folder.
   * Ensure it contains required columns like:
     `id, company_name, website, industry, company_size, revenue_range, location, technologies, contact_name, contact_title, contact_email, contact_phone, email_response, website_visits, content_downloads, meeting_booked, last_contact_date, lead_score, is_duplicate, match_with_company_id`

---
🤖 Train the Lead Scoring Model

Run this on your terminal :

python train_model.py


This will:

* Preprocess the dataset
* Train the `RandomForestClassifier`
* Save `lead_scoring_model.pkl` and `label_encoders.pkl` in the `models/` folder

🚀 Run the Application

Start the Flask server:
open you terminal and write : python app.py


Mau saya buatkan?
