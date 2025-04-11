from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for

import pandas as pd
from career_recommender import CareerRecommendationSystem

app = Flask(__name__)  # <-- Make sure this is here
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///careercompass.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class AssessmentHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, default=pd.Timestamp.utcnow)
    data = db.Column(db.Text)
    results = db.Column(db.Text)


#from flask import Flask, render_template, request, redirect, url_for

recommender = CareerRecommendationSystem()

# Try to load the model
try:
    recommender.load_model('career_recommendation_model.pkl')
except:
    print("Training new model...")
    recommender.preprocess_data()
    recommender.save_model('career_recommendation_model.pkl')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    # Get data for the form
    education_levels = ['High School', 'Associate Degree', 'Bachelor\'s', 'Master\'s', 'Professional Degree', 'PhD']
    gender_options = ['Male', 'Female', 'Non-Binary', 'Prefer Not to Say']
    categories = list(recommender.career_categories.keys())
    work_envs = ['Remote', 'Hybrid', 'On-site', 'Flexible', 'Travel-based']
    
    skill_descriptions = {
        'stem_interest_score': "STEM (Science, Tech, Engineering, Math) Interest",
        'creativity_score': "Creativity",
        'social_skills_score': "Social Skills",
        'analytical_skills_score': "Analytical Skills",
        'problem_solving_score': "Problem Solving",
        'technical_aptitude_score': "Technical Aptitude",
        'communication_score': "Communication",
        'leadership_potential_score': "Leadership Potential",
        'work_life_balance_score': "Work-Life Balance Importance",
        'global_mobility_score': "Global Mobility Interest"
    }
    
    return render_template('questionnaire.html', 
                           education_levels=education_levels,
                           gender_options=gender_options,
                           categories=categories,
                           work_envs=work_envs,
                           skill_descriptions=skill_descriptions)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        user = User(email=email, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/history')
@login_required
def view_history():
    history = AssessmentHistory.query.filter_by(user_id=current_user.id).order_by(AssessmentHistory.timestamp.desc()).all()
    return render_template('history.html', history=history)

@app.route('/submit', methods=['POST'])
def submit():
    # Process form data
    user_data = {
        'age': int(request.form['age']),
        'education_level': request.form['education_level'],
        'gender': request.form['gender'],
        'career_category': request.form['career_category'],
        'work_environment_preference': request.form['work_environment_preference'],
        'annual_salary_expectation': (int(request.form['min_salary']) + int(request.form['max_salary'])) // 2,
        'job_growth_interest': int(request.form['job_growth_interest'])
    }
    
    # Add skill scores
    skills = [
        'stem_interest_score', 'creativity_score', 'social_skills_score',
        'analytical_skills_score', 'problem_solving_score',
        'technical_aptitude_score', 'communication_score',
        'leadership_potential_score', 'work_life_balance_score',
        'global_mobility_score'
    ]
    
    for skill in skills:
        user_data[skill] = int(request.form[skill])
    
    # Get recommendations
    recommendations = recommender.predict_career(user_data)
    
    # Format career information for template
    career_infos = []
    for i, row in recommendations.iterrows():
        career = row['Career']
        score = row['Match Score']
        
        # Find category
        category = None
        for cat, careers in recommender.career_categories.items():
            if career in careers:
                category = cat
                break
        
        # Get description
        description = recommender.career_descriptions.get(career, "Information not available")
        
        # Get resources
        resources = recommender.career_resources.get(career, ["Information not available"])
        
        career_infos.append({
            'name': career,
            'score': score,
            'category': category,
            'description': description,
            'resources': resources
        })
        if current_user.is_authenticated:
    # Check if history already exists
            history = AssessmentHistory.query.filter_by(user_id=current_user.id).first()
            if history:
        # Update existing history
                history.timestamp = pd.Timestamp.utcnow()
                history.data = str(user_data)
                history.results = str(career_infos)
            else:
        # Create new history
                history = AssessmentHistory(
                    user_id=current_user.id,
                    data=str(user_data),
                    results=str(career_infos)
                )
                db.session.add(history)
            db.session.commit()

    
    return render_template('results.html', career_infos=career_infos)

if __name__ == '__main__':
    app.run(debug=True)