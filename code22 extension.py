from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib
import os
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = "career_recommendation_secret_key"

# Path to the model
MODEL_PATH = 'career_recommendation_model.pkl'

# Load the recommender class
class CareerRecommendationSystem:
    def __init__(self, dataset_path=None):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.career_categories = None
        self.career_descriptions = self.get_career_descriptions()
        self.career_resources = self.get_career_resources()
    
    def get_career_descriptions(self):
        """Dictionary of career descriptions"""
        return {
            # Technology careers
            'Software Engineer': "Designs, develops, and maintains software systems and applications. Works with various programming languages and frameworks to create efficient, scalable software solutions.",
            'Data Scientist': "Analyzes and interprets complex data to help organizations make better decisions. Combines statistics, mathematics, and programming to extract insights from data.",
            'Cybersecurity Analyst': "Protects computer systems and networks from information disclosure, theft, and damage. Monitors for security breaches and implements security measures.",
            'User Experience Designer': "Creates meaningful and relevant experiences for users. Designs the entire process of acquiring and integrating a product, including aspects of branding, design, usability, and function.",
            'Cloud Solutions Architect': "Designs and implements cloud computing solutions for organizations. Develops migration strategies, application designs, and manages cloud infrastructure.",
            'Machine Learning Engineer': "Develops artificial intelligence systems that can learn and apply knowledge without specific directions. Creates and implements machine learning algorithms.",
            'DevOps Engineer': "Combines software development and IT operations. Works to shorten the development lifecycle while delivering features, fixes, and updates frequently.",
            'Blockchain Developer': "Develops and implements blockchain architecture and solutions. Creates decentralized applications and smart contracts.",
            'AI Research Scientist': "Conducts research to advance the field of artificial intelligence. Develops new algorithms, approaches, and applications for AI systems.",
            'Network Security Specialist': "Focuses on protecting an organization's network infrastructure. Designs and implements security measures for network systems.",
            
            # Science careers
            'Biologist': "Studies living organisms and their relationship to the environment. May specialize in particular types of organisms or specific aspects of life.",
            'Physicist': "Studies matter, energy, and their interactions. Develops theories and models to explain the properties of the natural world.",
            'Environmental Scientist': "Studies the effects of human activity on the environment. Works to identify, control, or eliminate sources of pollutants or hazards affecting the environment.",
            'Geneticist': "Studies genes, genetic variations, and heredity. Works to understand how traits are passed from generation to generation.",
            'Chemical Engineer': "Applies principles of chemistry, physics, and mathematics to solve problems involving the production or use of chemicals and other products.",
            'Astronomer': "Studies celestial objects, space, and the physical universe. Observes, researches, and interprets astronomical phenomena.",
            'Forensic Scientist': "Applies scientific principles and techniques to the investigation of crimes. Collects and analyzes physical evidence.",
            'Marine Biologist': "Studies marine organisms and their behaviors and interactions with the environment. May specialize in certain species, behaviors, or ecosystems.",
            'Geologist': "Studies the physical structure and processes of the Earth. Examines rocks, minerals, and the processes that shape the Earth's surface.",
            'Zoologist': "Studies animals and their interactions with ecosystems. Analyzes animal behavior, genetics, and life processes.",
            
            # Healthcare careers
            'Doctor/Physician': "Diagnoses and treats injuries and illnesses. May specialize in specific areas of medicine such as cardiology, neurology, or pediatrics.",
            'Psychologist': "Studies cognitive, emotional, and social processes and behavior. Applies research to help improve processes and behaviors.",
            'Nurse Practitioner': "Provides advanced nursing care. Can prescribe medication, examine patients, diagnose illnesses, and provide treatments.",
            'Pharmacist': "Prepares and dispenses medications. Advises patients and healthcare professionals on the selection, dosages, and side effects of medications.",
            'Dentist': "Diagnoses and treats problems with teeth and tissues in the mouth. Works to prevent problems and improve patients' appearance and confidence.",
            'Physical Therapist': "Helps injured or ill people improve movement and manage pain. Develops plans using treatment techniques to promote ability to move, reduce pain, and restore function.",
            'Veterinarian': "Diagnoses and treats diseases and injuries in animals. May specialize in certain types of animals or in specific areas of medicine.",
            'Medical Researcher': "Conducts research aimed at improving overall human health. May work to develop new treatments, medications, or medical devices.",
            'Nutritionist': "Advises people on what to eat to lead a healthy lifestyle or achieve a specific health-related goal. Studies and communicates the effects of food and nutrition on health.",
            'Psychiatrist': "Specializes in mental health, including substance use disorders. Diagnoses and treats mental, emotional, and behavioral disorders.",
            
            # Creative careers
            'Graphic Designer': "Creates visual concepts, using computer software or by hand, to communicate ideas that inspire, inform, and captivate consumers.",
            'Writer/Author': "Develops written content for various media. May work on books, articles, scripts, or web content.",
            'Multimedia Artist': "Creates special effects, animation, or other visual images using film, video, computers, or other electronic tools and media.",
            'Film Director': "Directs the making of a film. Controls a film's artistic and dramatic aspects and visualizes the script while guiding the technical crew and actors.",
            'Video Game Designer': "Designs core gameplay elements including storylines, role-play mechanics, and character biographies. Balances the gameplay mechanics to ensure the intended experience.",
            'Interior Designer': "Plans, designs, and furnishes interiors of residential, commercial, or industrial buildings. Creates functional, safe, and aesthetically pleasing spaces.",
            'Fashion Designer': "Designs clothing, footwear, and accessories. Creates original garments, works with design teams, and oversees the creation of prototypes.",
            'Music Producer': "Oversees and manages the recording of an artist's music. Controls the recording sessions and guides the artists and technical team during the recording process.",
            'Animator': "Creates multiple images (frames) that create an illusion of movement when displayed in rapid sequence. Works in films, video games, television, or internet.",
            'Art Director': "Determines the overall visual appearance and how it communicates visually with its audience. Directs others who develop artwork or layouts.",
            
            # More careers (continuing for other categories)...
            'Teacher': "Educates students of various ages in different subjects. Plans lessons, assesses student progress, and creates an engaging learning environment.",
            'Financial Analyst': "Evaluates investment opportunities. Researches and analyzes financial information to help companies make business decisions.",
            'Digital Marketer': "Promotes brands using digital channels. Develops, implements, and manages marketing campaigns that promote a company and its products/services.",
            'Entrepreneur': "Starts and runs own business ventures. Identifies opportunities, secures resources, and builds organizations to capitalize on those opportunities.",
            
            # Adding more examples for different categories...
            'Journalist': "Researches, writes, and reports news stories. Investigates and presents information to help people understand events, issues, and trends.",
            'Civil Engineer': "Designs, builds, and maintains infrastructure projects and systems. Works on roads, buildings, airports, tunnels, dams, bridges, and water supply systems.",
            'Sociologist': "Studies society and social behavior. Examines groups, cultures, organizations, social institutions, and processes that people develop.",
            'Chef': "Prepares food in restaurants or other food service establishments. Creates recipes, plans menus, and directs food preparation activities."
        }
    
    def get_career_resources(self):
        """Dictionary of resources for pursuing careers"""
        return {
            # Technology resources
            'Software Engineer': [
                "Degree programs: Computer Science, Software Engineering",
                "Certifications: AWS Certified Developer, Microsoft Certified: Azure Developer",
                "Online platforms: Codecademy, LeetCode, GitHub",
                "Professional organizations: IEEE Computer Society, ACM"
            ],
            'Data Scientist': [
                "Degree programs: Data Science, Statistics, Computer Science",
                "Certifications: IBM Data Science Professional, Google Data Analytics",
                "Online platforms: Kaggle, DataCamp, Coursera",
                "Professional organizations: Data Science Association, INFORMS"
            ],
            'Cybersecurity Analyst': [
                "Degree programs: Cybersecurity, Information Technology",
                "Certifications: CompTIA Security+, Certified Information Systems Security Professional (CISSP)",
                "Online platforms: TryHackMe, HackTheBox, Cybrary",
                "Professional organizations: (ISC)², ISACA"
            ],
            
            # Healthcare resources
            'Doctor/Physician': [
                "Degree programs: Pre-medicine undergraduate, followed by Medical Doctor (MD) or Doctor of Osteopathic Medicine (DO)",
                "Examinations: MCAT, USMLE (for MDs), COMLEX (for DOs)",
                "Resources: AAMC, MedEdPORTAL, UpToDate",
                "Professional organizations: American Medical Association, specialty-specific organizations"
            ],
            'Psychologist': [
                "Degree programs: Psychology (PhD or PsyD for clinical practice)",
                "Certifications: State licensure required for clinical practice",
                "Resources: American Psychological Association, PsycINFO",
                "Professional organizations: American Psychological Association, Association for Psychological Science"
            ],
            
            # Creative resources
            'Graphic Designer': [
                "Degree programs: Graphic Design, Visual Communications",
                "Software skills: Adobe Creative Suite (Photoshop, Illustrator, InDesign)",
                "Portfolio platforms: Behance, Dribbble",
                "Professional organizations: AIGA (American Institute of Graphic Arts)"
            ],
            'Writer/Author': [
                "Degree programs: Creative Writing, English, Journalism",
                "Resources: Masterclass, Writer's Digest, Reedsy",
                "Communities: NaNoWriMo, Critique Circle",
                "Professional organizations: Authors Guild, Society of Children's Book Writers and Illustrators"
            ],
            
            # Business resources
            'Financial Analyst': [
                "Degree programs: Finance, Accounting, Economics",
                "Certifications: CFA (Chartered Financial Analyst), FRM (Financial Risk Manager)",
                "Resources: Bloomberg Terminal, Wall Street Journal, Financial Times",
                "Professional organizations: CFA Institute, Global Association of Risk Professionals"
            ],
            'Entrepreneur': [
                "Degree programs: Business Administration, Entrepreneurship",
                "Resources: Y Combinator Startup School, Entrepreneurship magazines (Inc., Entrepreneur)",
                "Networks: Startup incubators, local entrepreneur meetups",
                "Organizations: Entrepreneurs' Organization, Young Entrepreneurs Council"
            ],
            
            # Adding more examples...
            'Teacher': [
                "Degree programs: Education, subject-specific degrees with teaching certification",
                "Certifications: State teaching license, National Board Certification",
                "Resources: Edutopia, Khan Academy Teacher Resources",
                "Professional organizations: National Education Association, subject-specific teacher associations"
            ],
            'Civil Engineer': [
                "Degree programs: Civil Engineering",
                "Certifications: Professional Engineer (PE) license",
                "Resources: AutoCAD, Civil engineering handbooks",
                "Professional organizations: American Society of Civil Engineers"
            ]
        }
    
    def load_model(self, model_path):
        """Load the model and preprocessor"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_names = model_data['feature_names']
            self.career_categories = model_data['career_categories']
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_career(self, user_data):
        """Predict career based on user data"""
        if self.model is None:
            print("Model not loaded yet.")
            return None
        
        # Convert user data to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Transform user data
        user_transformed = self.preprocessor.transform(user_df)
        
        # Get probabilities for all careers
        career_probs = self.model.predict_proba(user_transformed)
        
        # Get indices of top 5 careers
        top_indices = career_probs[0].argsort()[-5:][::-1]
        
        # Get career names
        career_names = [self.model.classes_[i] for i in top_indices]
        
        # Get probabilities
        probabilities = [career_probs[0][i] for i in top_indices]
        
        # Create recommendations list
        recommendations = [
            {'career': career, 'match_score': f"{prob:.1%}", 'match_value': prob} 
            for career, prob in zip(career_names, probabilities)
        ]
        
        return recommendations

    def get_career_details(self, career):
        """Get detailed information about a career"""
        # Find category
        category = None
        for cat, careers in self.career_categories.items():
            if career in careers:
                category = cat
                break
        
        # Get description
        description = self.career_descriptions.get(career, "Information not available")
        
        # Get resources
        resources = self.career_resources.get(career, [
            "Research relevant degree programs in this field",
            "Look for professional certifications that can enhance your credentials",
            "Join professional organizations related to this field",
            "Build a portfolio or gain experience through internships or entry-level positions"
        ])
        
        return {
            'career': career,
            'category': category,
            'description': description,
            'resources': resources
        }

# Initialize the recommender
recommender = CareerRecommendationSystem()

# Check if model exists and load it
if os.path.exists(MODEL_PATH):
    recommender.load_model(MODEL_PATH)
else:
    print("Model file not found. Please train and save the model first.")

# Common chat responses for the AI assistant
chat_responses = {
    "greeting": [
        "Hello! I'm your Career Assistant. How can I help you today?",
        "Welcome to the Career Recommendation System! What would you like to know?",
        "Hi there! I'm here to help with your career questions. What's on your mind?"
    ],
    "about_system": [
        "This Career Recommendation System uses machine learning to suggest careers based on your skills, interests, and preferences. Just fill out the questionnaire to get personalized recommendations!",
        "Our system analyzes your profile against successful career patterns to find matches that suit your unique combination of skills and interests."
    ],
    "how_it_works": [
        "The system works by comparing your profile to patterns found in career data. First, complete the questionnaire with your skills and preferences. Then our AI model will find careers that match your profile and rank them by compatibility.",
        "Our recommendation engine uses a Random Forest algorithm to analyze how your unique combination of skills, interests, and preferences align with different career paths."
    ],
    "fallback": [
        "I'm not sure I understand. Could you rephrase your question? I'm here to help with career-related queries.",
        "I'm still learning! For specific career advice, I recommend completing our questionnaire to get personalized recommendations.",
        "That's a bit outside my expertise. I'm best at helping with career recommendations and information about different career paths."
    ]
}

# Store user chat history
chat_history = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    # Get all career categories for the form
    categories = list(recommender.career_categories.keys()) if recommender.career_categories else []
    return render_template('questionnaire.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user data from form
        user_data = {
            'age': int(request.form.get('age')),
            'education_level': request.form.get('education_level'),
            'gender': request.form.get('gender'),
            'stem_interest_score': int(request.form.get('stem_interest_score')),
            'creativity_score': int(request.form.get('creativity_score')),
            'social_skills_score': int(request.form.get('social_skills_score')),
            'analytical_skills_score': int(request.form.get('analytical_skills_score')),
            'problem_solving_score': int(request.form.get('problem_solving_score')),
            'technical_aptitude_score': int(request.form.get('technical_aptitude_score')),
            'communication_score': int(request.form.get('communication_score')),
            'leadership_potential_score': int(request.form.get('leadership_potential_score')),
            'work_life_balance_score': int(request.form.get('work_life_balance_score')),
            'global_mobility_score': int(request.form.get('global_mobility_score')),
            'career_category': request.form.get('career_category'),
            'work_environment_preference': request.form.get('work_environment_preference'),
            'annual_salary_expectation': int(request.form.get('annual_salary_expectation')),
            'job_growth_interest': int(request.form.get('job_growth_interest'))
        }
        
        # Get career recommendations
        recommendations = recommender.predict_career(user_data)
        
        if recommendations:
            # Store recommendations in session
            session['recommendations'] = recommendations
            session['user_data'] = user_data
            
            # Get detailed information for recommended careers
            career_details = []
            for rec in recommendations:
                career = rec['career']
                details = recommender.get_career_details(career)
                details['match_score'] = rec['match_score']
                details['match_value'] = rec['match_value']
                career_details.append(details)
            
            return render_template('results.html', recommendations=career_details, user_data=user_data)
        else:
            return render_template('error.html', message="Error generating recommendations. Please try again.")
    
    return render_template('questionnaire.html')

@app.route('/career_details/<career>')
def career_details(career):
    details = recommender.get_career_details(career)
    return render_template('career_details.html', details=details)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').lower()
    user_id = request.json.get('user_id', '')
    
    # Create a unique user ID if not provided
    if not user_id:
        user_id = str(uuid.uuid4())
    
    # Initialize chat history for new users
    if user_id not in chat_history:
        chat_history[user_id] = []
    
    # Add user message to history
    chat_history[user_id].append({
        'role': 'user',
        'message': user_message,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Determine response based on user message
    response = ""
    
    if any(greeting in user_message for greeting in ['hi', 'hello', 'hey', 'greetings']):
        response = np.random.choice(chat_responses["greeting"])
    
    elif any(word in user_message for word in ['about', 'what is', 'what does', 'purpose']):
        response = np.random.choice(chat_responses["about_system"])
    
    elif any(word in user_message for word in ['how', 'work', 'process', 'algorithm']):
        response = np.random.choice(chat_responses["how_it_works"])
    
    elif any(word in user_message for word in ['recommend', 'suggest', 'career', 'job']):
        response = "To get personalized career recommendations, please complete our questionnaire. It takes into account your skills, interests, education, and preferences to suggest careers that might be a good fit for you."
    
    elif any(word in user_message for word in ['thank', 'thanks']):
        response = "You're welcome! I'm happy to help with your career exploration journey. Is there anything else you'd like to know?"
    
    else:
        response = np.random.choice(chat_responses["fallback"])
    
    # Add assistant response to history
    chat_history[user_id].append({
        'role': 'assistant',
        'message': response,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    return jsonify({
        'response': response,
        'user_id': user_id
    })

@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    feedback_data = request.json
    
    # In a production environment, you would save this to a database
    # For now, we'll just print it
    print(f"Feedback received: {feedback_data}")
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)