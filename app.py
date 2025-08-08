from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import joblib
from fuzzywuzzy import fuzz
import numpy as np
from sentence_transformers import SentenceTransformer 




app = Flask(__name__)
app.secret_key = 'your-very-secret-key-12345'  




try:
    df = pd.read_csv('data/Companies.csv')
    model = joblib.load('models/lead_scoring_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
except FileNotFoundError as e:
    print(f"Error loading data or models: {e}. Make sure 'data/Companies.csv', 'models/lead_scoring_model.pkl', and 'models/label_encoders.pkl' exist.")
    exit() 



@app.before_request
def before_request():
    if 'user_leads' not in session:
        session['user_leads'] = []



df['target'] = df['meeting_booked'].apply(lambda x: 1 if x == 'Yes' else 0)
df['tech_count'] = df['technologies'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)





def find_similar_companies(company_name, threshold=85):
    matches = []
    if 'df' not in globals():
        print("DataFrame 'df' not loaded in find_similar_companies.")
        return []


    for _, row in df.iterrows():
      
        score = fuzz.token_set_ratio(company_name.lower(), row['company_name'].lower())
        if score >= threshold:
            matches.append({
                'id': row['id'],
                'name': row['company_name'],
                'score': score,
                'industry': row['industry'],
                'location': row['location']
            })
    return matches





def get_dropdown_options():
    industries = sorted(df['industry'].unique().tolist())
    locations = sorted(df['location'].unique().tolist())
    return {'industries': industries, 'locations': locations}





def calculate_lead_scores(leads_df):
    features = ['industry', 'company_size', 'revenue_range', 'website_visits',
               'content_downloads', 'email_response', 'tech_count']
    X = leads_df[features].copy()
   
    for col in ['industry', 'company_size', 'revenue_range', 'email_response']:

        if col in encoders:
            X[col] = encoders[col].transform(X[col])
        else:
            print(f"Warning: Encoder for column '{col}' not found.")
            X[col] = 0 
   
    leads_df['lead_score'] = model.predict_proba(X)[:, 1] * 100
    return leads_df.sort_values('lead_score', ascending=False)




@app.route('/')
def home():
    options = get_dropdown_options()
   
    if not session['user_leads']:
        return render_template('home.html',
                            stats=None,
                            top_leads=None,
                            industries=options['industries'],
                            locations=options['locations'])
   
    user_leads_df = df[df['id'].isin(session['user_leads'])].copy()
    scored_leads = calculate_lead_scores(user_leads_df)
    top_leads = scored_leads.head(10).to_dict('records')
   
    
    stats = {
        'total_leads': len(user_leads_df),
        'meeting_booked': user_leads_df['meeting_booked'].value_counts().to_dict(),
        'top_industries': user_leads_df['industry'].value_counts().head(3).to_dict(),
        'lead_score_distribution': {
            'Hot (80-100)': len(user_leads_df[user_leads_df['lead_score'] >= 80]),
            'Warm (50-79)': len(user_leads_df[(user_leads_df['lead_score'] >= 50) & (user_leads_df['lead_score'] < 80)]),
            'Cold (<50)': len(user_leads_df[user_leads_df['lead_score'] < 50])
        }
    }
   
    return render_template('home.html',
                         top_leads=top_leads,
                         stats=stats,
                         industries=options['industries'],
                         locations=options['locations'])




@app.route('/scraper', methods=['GET', 'POST'])
def scraper():
    options = get_dropdown_options()
   
    if request.method == 'POST':
        industry = request.form.get('industry')
        location = request.form.get('location')
       
        
        filtered = df.copy()
        if industry:
            filtered = filtered[filtered['industry'].str.contains(industry, case=False, na=False)] 
        if location:
            filtered = filtered[filtered['location'].str.contains(location, case=False, na=False)] 
       
        return render_template('scraper.html',
                             companies=filtered.to_dict('records'),
                             search=True,
                             industry=industry,
                             location=location,
                             industries=options['industries'],
                             locations=options['locations'],
                             user_leads=session['user_leads']) 
   
    return render_template('scraper.html',
                         search=False,
                         industries=options['industries'],
                         locations=options['locations'],
                         user_leads=session['user_leads'])te




@app.route('/add_lead', methods=['POST'])
def add_lead():
    company_id = int(request.form.get('company_id'))
    company_to_add = df[df['id'] == company_id].iloc[0]

    if 'user_leads' not in session:
        session['user_leads'] = []

    
    if company_id in session['user_leads']:
        return jsonify({'status': 'success'})  

    
    all_similar_companies = find_similar_companies(company_to_add['company_name'])

    
    existing_duplicates_in_user_leads = [
        d for d in all_similar_companies
        if d['id'] in session['user_leads'] and d['id'] != company_id
    ]

    if len(existing_duplicates_in_user_leads) > 0:
        return jsonify({
            'status': 'duplicate',
            'company': company_to_add.to_dict(),
            'duplicates': existing_duplicates_in_user_leads
        })
    else:
        
        session['user_leads'].append(company_id)
        session.modified = True
        return jsonify({'status': 'success'})





@app.route('/remove_lead', methods=['POST'])
def remove_lead():
    company_id = int(request.form.get('company_id'))
   
    if company_id in session['user_leads']:
        session['user_leads'].remove(company_id)
        session.modified = True
   
    return jsonify({'status': 'success'})




if __name__ == '__main__':
    app.run(debug=True)


