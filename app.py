from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for
from flask import flash
from flask_login import login_required
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
    name = db.Column(db.String(150), nullable=False)  # 👈 new
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
@login_required
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
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("signup.html")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please log in.", "warning")
            return redirect(url_for("login"))

        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
           login_user(user)
           return redirect(url_for('dashboard'))

        else:
            error = "Invalid email or password."
            return render_template('login.html', error=error)

    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    raw_history = AssessmentHistory.query.filter_by(user_id=current_user.id).all()

    import ast
    from collections import Counter

    top_careers = []

    for entry in raw_history:
        try:
            results = ast.literal_eval(entry.results)
            if isinstance(results, list) and results:
                top_careers.append(results[0]['name'])
        except:
            continue

    # Count most frequent top careers
    career_counts = dict(Counter(top_careers))

    return render_template("dashboard.html",
                           current_user=current_user,
                           career_counts=career_counts)



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

import ast  # at the top of your file if not already imported

@app.route('/history')
@login_required
def view_history():
    raw_history = AssessmentHistory.query.filter_by(user_id=current_user.id).order_by(AssessmentHistory.timestamp.desc()).all()

    history = []
    for item in raw_history:
        try:
            results = ast.literal_eval(item.results)
        except:
            results = []

        top_career = results[0]['name'] if results else 'N/A'

        history.append({
            'timestamp': item.timestamp,
            'top_career': top_career,
            'results': results
        })

    return render_template('history.html', history=history)

#  @app.route('/submit', methods=['POST'])
# @login_required
# def submit():
#     skill_descriptions = {
#         "stem_interest_score": "Interest in STEM fields",
#         "creativity_score": "Creative thinking ability",
#         "social_skills_score": "Interpersonal and social skills",
#         "analytical_skills_score": "Analytical thinking and reasoning",
#         "problem_solving_score": "Problem-solving capability",
#         "technical_aptitude_score": "Comfort with technical tools or topics",
#         "communication_score": "Communication skills",
#         "leadership_potential_score": "Leadership and initiative",
#         "work_life_balance_score": "Preference for work-life balance",
#         "global_mobility_score": "Willingness to work internationally"
#     }

#     try:
#         age = int(request.form['age'])
#         min_salary = int(request.form['min_salary'])
#         max_salary = int(request.form['max_salary'])

#         if not (18 <= age <= 60):
#             flash("Age must be between 18 and 60.", "danger")
#             return render_template("questionnaire.html", error="Age must be between 18 and 60.", skill_descriptions=skill_descriptions)

#         if min_salary < 0:
#             flash("Minimum salary must be 0 or greater.", "danger")
#             return render_template("questionnaire.html", error="Minimum salary must be 0 or greater.", skill_descriptions=skill_descriptions)

#         if max_salary <= min_salary:
#             flash("Maximum salary must be greater than minimum salary.", "danger")
#             return render_template("questionnaire.html", error="Maximum salary must be greater than minimum salary.", skill_descriptions=skill_descriptions)

#         user_data = {
#             'age': age,
#             'education_level': request.form['education_level'],
#             'gender': request.form['gender'],
#             'career_category': request.form['career_category'],
#             'work_environment_preference': request.form['work_environment_preference'],
#             'annual_salary_expectation': (min_salary + max_salary) // 2,
#             'job_growth_interest': int(request.form['job_growth_interest'])
#         }

#         skills = [
#             'stem_interest_score', 'creativity_score', 'social_skills_score',
#             'analytical_skills_score', 'problem_solving_score',
#             'technical_aptitude_score', 'communication_score',
#             'leadership_potential_score', 'work_life_balance_score',
#             'global_mobility_score'
#         ]
#         for skill in skills:
#             user_data[skill] = int(request.form[skill])

#         recommendations = recommender.predict_career(user_data)
#         career_infos = []

#         for i, row in recommendations.iterrows():
#             career = row['Career']
#             score = row['Match Score']

#             category = None
#             for cat, careers in recommender.career_categories.items():
#                 if career in careers:
#                     category = cat
#                     break

#             description = recommender.career_descriptions.get(career, "Information not available")
#             resources = recommender.career_resources.get(career)
#             if not resources:
#                 resources = [
#                     "Research relevant degree programs in this field",
#                     "Look for professional certifications that can enhance your credentials",
#                     "Join professional organizations related to this field",
#                     "Build a portfolio or gain experience through internships or entry-level positions"
#                 ]

#             career_infos.append({
#                 'name': career,
#                 'score': score,
#                 'category': category,
#                 'description': description,
#                 'resources': resources
#             })

#         if current_user.is_authenticated:
#             history = AssessmentHistory.query.filter_by(user_id=current_user.id).first()
#             if history:
#                 history.timestamp = pd.Timestamp.utcnow()
#                 history.data = str(user_data)
#                 history.results = str(career_infos)
#             else:
#                 history = AssessmentHistory(
#                     user_id=current_user.id,
#                     data=str(user_data),
#                     results=str(career_infos)
#                 )
#                 db.session.add(history)
#             db.session.commit()

#         return render_template('results.html', career_infos=career_infos)

#     except ValueError:
#         flash("Invalid input. Please enter valid numeric values.", "danger")
#         return render_template("questionnaire.html", error="Invalid input detected.", skill_descriptions=skill_descriptions)
@app.route('/submit', methods=['POST'])
@login_required
def submit():
    skill_descriptions = {
        "stem_interest_score": "Interest in STEM fields",
        "creativity_score": "Creative thinking ability",
        "social_skills_score": "Interpersonal and social skills",
        "analytical_skills_score": "Analytical thinking and reasoning",
        "problem_solving_score": "Problem-solving capability",
        "technical_aptitude_score": "Comfort with technical tools or topics",
        "communication_score": "Communication skills",
        "leadership_potential_score": "Leadership and initiative",
        "work_life_balance_score": "Preference for work-life balance",
        "global_mobility_score": "Willingness to work internationally"
    }

    try:
        age = int(request.form['age'])
        min_salary = int(request.form['min_salary'])
        max_salary = int(request.form['max_salary'])

        if not (18 <= age <= 60):
            flash("Age must be between 18 and 60.", "danger")
            return render_template("questionnaire.html", error="Age must be between 18 and 60.", skill_descriptions=skill_descriptions)

        if min_salary < 0:
            flash("Minimum salary must be 0 or greater.", "danger")
            return render_template("questionnaire.html", error="Minimum salary must be 0 or greater.", skill_descriptions=skill_descriptions)

        if max_salary <= min_salary:
            flash("Maximum salary must be greater than minimum salary.", "danger")
            return render_template("questionnaire.html", error="Maximum salary must be greater than minimum salary.", skill_descriptions=skill_descriptions)

        user_data = {
            'age': age,
            'education_level': request.form['education_level'],
            'gender': request.form['gender'],
            'career_category': request.form['career_category'],
            'work_environment_preference': request.form['work_environment_preference'],
            'annual_salary_expectation': (min_salary + max_salary) // 2,
            'job_growth_interest': int(request.form['job_growth_interest'])
        }

        skills = [
            'stem_interest_score', 'creativity_score', 'social_skills_score',
            'analytical_skills_score', 'problem_solving_score',
            'technical_aptitude_score', 'communication_score',
            'leadership_potential_score', 'work_life_balance_score',
            'global_mobility_score'
        ]
        for skill in skills:
            user_data[skill] = int(request.form[skill])

        recommendations = recommender.predict_career(user_data)
        career_infos = []

        for i, row in recommendations.iterrows():
            career = row['Career']
            score = row['Match Score']

            category = None
            for cat, careers in recommender.career_categories.items():
                if career in careers:
                    category = cat
                    break

            description = recommender.career_descriptions.get(career, "Information not available")
            resources = recommender.career_resources.get(career)
            if not resources:
                resources = [
                    "Research relevant degree programs in this field",
                    "Look for professional certifications that can enhance your credentials",
                    "Join professional organizations related to this field",
                    "Build a portfolio or gain experience through internships or entry-level positions"
                ]

            career_infos.append({
                'name': career,
                'score': score,
                'category': category,
                'description': description,
                'resources': resources
            })

        # Create a new history entry each time, instead of updating an existing one
        if current_user.is_authenticated:
            new_history = AssessmentHistory(
                user_id=current_user.id,
                timestamp=pd.Timestamp.utcnow(),
                data=str(user_data),
                results=str(career_infos)
            )
            db.session.add(new_history)
            db.session.commit()

        return render_template('results.html', career_infos=career_infos)

    except ValueError:
        flash("Invalid input. Please enter valid numeric values.", "danger")
        return render_template("questionnaire.html", error="Invalid input detected.", skill_descriptions=skill_descriptions)

if __name__ == '__main__':
    app.run(debug=True)