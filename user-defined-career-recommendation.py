import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import random
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

# -------------------------
# Step 1: Dataset Creation with User-Defined Interests
# -------------------------

np.random.seed(42)

# Career information with detailed descriptions
career_info = {
    'Software Engineer': {
        'description': 'Develops applications and systems using programming languages and software development tools',
        'required_skills': ['Programming', 'Problem-solving', 'Logical thinking', 'Debugging'],
        'education': 'Bachelor\'s in Computer Science or related field',
        'salary_range': '$70,000 - $150,000',
        'growth_prospects': 'Excellent, with 22% growth expected through 2030',
        'related_keywords': ['programming', 'coding', 'software', 'development', 'algorithms', 'web', 'app', 
                          'computer', 'technology', 'debugging', 'logic', 'mathematics', 'problem solving']
    },
    'Data Scientist': {
        'description': 'Analyzes and interprets complex data to help organizations make better decisions',
        'required_skills': ['Statistics', 'Programming', 'Machine Learning', 'Data Visualization'],
        'education': 'Master\'s in Data Science, Statistics, or related field',
        'salary_range': '$95,000 - $180,000',
        'growth_prospects': 'Very high, with 31% growth expected through 2030',
        'related_keywords': ['data', 'analysis', 'statistics', 'machine learning', 'ai', 'mathematics', 
                          'programming', 'visualization', 'research', 'big data', 'modeling', 'algorithms']
    },
    'Doctor/Physician': {
        'description': 'Diagnoses and treats injuries or illnesses and advises patients on health and wellness',
        'required_skills': ['Clinical Knowledge', 'Empathy', 'Problem-solving', 'Communication'],
        'education': 'Medical Degree (MD or DO) plus residency',
        'salary_range': '$180,000 - $300,000+',
        'growth_prospects': 'Steady, with 7% growth expected through 2030',
        'related_keywords': ['medicine', 'health', 'biology', 'anatomy', 'patients', 'healthcare', 'medical', 
                          'treatment', 'disease', 'diagnosis', 'healing', 'caring', 'physiology', 'surgery']
    },
    'Graphic Designer': {
        'description': 'Creates visual concepts to communicate ideas that inspire, inform, or captivate consumers',
        'required_skills': ['Creativity', 'Design Software', 'Typography', 'Visual Communication'],
        'education': 'Bachelor\'s in Graphic Design or related field',
        'salary_range': '$45,000 - $85,000',
        'growth_prospects': 'Moderate, with 3% growth expected through 2030',
        'related_keywords': ['design', 'art', 'creative', 'visual', 'graphics', 'illustration', 'drawing', 
                          'typography', 'layout', 'color', 'aesthetics', 'branding', 'digital art']
    },
    'Writer/Author': {
        'description': 'Develops written content for various media including books, websites, magazines, and more',
        'required_skills': ['Writing', 'Storytelling', 'Research', 'Editing'],
        'education': 'Bachelor\'s in English, Journalism, or Communications recommended',
        'salary_range': '$49,000 - $90,000',
        'growth_prospects': 'Average, with 4% growth expected through 2030',
        'related_keywords': ['writing', 'literature', 'language', 'storytelling', 'books', 'journalism', 
                          'content', 'editing', 'creativity', 'communication', 'grammar', 'publishing', 'blogging']
    },
    'Biologist': {
        'description': 'Studies living organisms and their relationships to the environment',
        'required_skills': ['Scientific Method', 'Lab Techniques', 'Data Analysis', 'Research'],
        'education': 'Bachelor\'s in Biology, Master\'s or PhD for research positions',
        'salary_range': '$65,000 - $120,000',
        'growth_prospects': 'Good, with 9% growth expected through 2030',
        'related_keywords': ['biology', 'science', 'research', 'laboratory', 'organisms', 'nature', 
                          'ecology', 'genetics', 'environment', 'cells', 'experiments', 'life sciences']
    },
    'Physicist': {
        'description': 'Studies the fundamental properties of matter, energy, space, and time',
        'required_skills': ['Advanced Mathematics', 'Problem-solving', 'Lab Techniques', 'Research'],
        'education': 'PhD in Physics or related field',
        'salary_range': '$75,000 - $150,000',
        'growth_prospects': 'Average, with 8% growth expected through 2030',
        'related_keywords': ['physics', 'science', 'mathematics', 'research', 'energy', 'mechanics', 
                          'space', 'time', 'astronomy', 'quantum', 'relativity', 'particles', 'theoretical']
    },
    'Psychologist': {
        'description': 'Studies cognitive, emotional, and social processes and behavior',
        'required_skills': ['Empathy', 'Research', 'Analysis', 'Communication'],
        'education': 'PhD in Psychology or PsyD',
        'salary_range': '$60,000 - $130,000',
        'growth_prospects': 'Excellent, with 14% growth expected through 2030',
        'related_keywords': ['psychology', 'mind', 'behavior', 'mental health', 'counseling', 'therapy', 
                          'emotions', 'cognition', 'human behavior', 'research', 'brain', 'assessment']
    },
    'Teacher': {
        'description': 'Educates students on various subjects and helps them develop knowledge and skills',
        'required_skills': ['Communication', 'Patience', 'Organization', 'Adaptability'],
        'education': 'Bachelor\'s degree and teaching certification',
        'salary_range': '$45,000 - $85,000',
        'growth_prospects': 'Steady, with 8% growth expected through 2030',
        'related_keywords': ['education', 'teaching', 'learning', 'students', 'classroom', 'knowledge', 
                          'instruction', 'curriculum', 'mentoring', 'school', 'academics', 'children']
    },
    'Environmental Scientist': {
        'description': 'Studies and protects the environment and human health from environmental hazards',
        'required_skills': ['Research', 'Data Analysis', 'Fieldwork', 'Scientific Writing'],
        'education': 'Bachelor\'s in Environmental Science or related field',
        'salary_range': '$55,000 - $100,000',
        'growth_prospects': 'Excellent, with 15% growth expected through 2030',
        'related_keywords': ['environment', 'conservation', 'sustainability', 'nature', 'ecology', 
                          'climate', 'pollution', 'research', 'earth', 'science', 'protection', 'green']
    },
    'Financial Analyst': {
        'description': 'Evaluates investment opportunities and provides financial guidance to businesses',
        'required_skills': ['Financial Modeling', 'Data Analysis', 'Research', 'Communication'],
        'education': 'Bachelor\'s in Finance, Accounting, or Economics',
        'salary_range': '$65,000 - $125,000',
        'growth_prospects': 'Good, with 6% growth expected through 2030',
        'related_keywords': ['finance', 'investing', 'stocks', 'economics', 'analysis', 'money', 
                          'markets', 'banking', 'business', 'accounting', 'trading', 'portfolio']
    },
    'Digital Marketer': {
        'description': 'Promotes products or services using digital channels and strategies',
        'required_skills': ['Social Media', 'Analytics', 'Content Creation', 'SEO/SEM'],
        'education': 'Bachelor\'s in Marketing or related field',
        'salary_range': '$50,000 - $110,000',
        'growth_prospects': 'Excellent, with 18% growth expected through 2030',
        'related_keywords': ['marketing', 'digital', 'advertising', 'social media', 'content', 'branding', 
                          'analytics', 'seo', 'online', 'strategy', 'campaigns', 'promotion', 'audience']
    },
    'Cybersecurity Analyst': {
        'description': 'Protects computer systems and networks from information disclosure and security breaches',
        'required_skills': ['Network Security', 'Threat Analysis', 'Cryptography', 'Programming'],
        'education': 'Bachelor\'s in Cybersecurity, Computer Science or related field',
        'salary_range': '$75,000 - $150,000',
        'growth_prospects': 'Exceptional, with 33% growth expected through 2030',
        'related_keywords': ['security', 'hacking', 'networks', 'protection', 'cyber', 'threats', 
                          'firewalls', 'encryption', 'defense', 'privacy', 'vulnerabilities', 'computer']
    },
    'User Experience Designer': {
        'description': 'Designs products and interfaces that provide meaningful and relevant experiences to users',
        'required_skills': ['User Research', 'Wireframing', 'Prototyping', 'Visual Design'],
        'education': 'Bachelor\'s in Design, HCI, or related field',
        'salary_range': '$70,000 - $130,000',
        'growth_prospects': 'Excellent, with 13% growth expected through 2030',
        'related_keywords': ['ux', 'ui', 'design', 'user experience', 'interfaces', 'usability', 
                          'interaction', 'user research', 'prototyping', 'information architecture', 'accessibility']
    },
    'Civil Engineer': {
        'description': 'Designs, builds and maintains infrastructure projects and systems',
        'required_skills': ['Technical Drawing', 'Mathematics', 'Problem-solving', 'Project Management'],
        'education': 'Bachelor\'s in Civil Engineering',
        'salary_range': '$65,000 - $110,000',
        'growth_prospects': 'Good, with 8% growth expected through 2030',
        'related_keywords': ['engineering', 'construction', 'infrastructure', 'design', 'buildings', 
                          'structures', 'bridges', 'roads', 'environment', 'project management', 'technical']
    }
}

# Function to generate synthetic dataset with free-form text interests
def generate_text_based_dataset(num_samples=1000):
    data = []
    
    # Create a pool of possible interest phrases for each career
    career_interest_pool = {}
    for career, info in career_info.items():
        # Generate sample interests based on keywords
        interest_pool = []
        for keyword in info['related_keywords']:
            # Create variations and specific phrases from keywords
            variations = [
                f"I enjoy {keyword}",
                f"I like working with {keyword}",
                f"I'm interested in {keyword}",
                f"I have experience in {keyword}",
                f"I'm passionate about {keyword}",
                f"I want to learn more about {keyword}",
                f"{keyword} is my hobby",
                f"I'm skilled in {keyword}",
                f"I've always been fascinated by {keyword}"
            ]
            interest_pool.extend(variations)
        
        career_interest_pool[career] = interest_pool
    
    # Generate samples
    for _ in range(num_samples):
        # Select a random career
        career = random.choice(list(career_info.keys()))
        
        # 80% of the time use interests related to the chosen career, 20% random
        if random.random() < 0.8:
            # Select 3-7 interests, mostly from the related pool
            num_related = random.randint(2, 5)
            related_interests = random.sample(career_interest_pool[career], min(num_related, len(career_interest_pool[career])))
            
            # Add some random interests from other careers
            other_careers = [c for c in career_info.keys() if c != career]
            if other_careers:
                random_career = random.choice(other_careers)
                num_other = random.randint(1, 2)
                other_interests = random.sample(career_interest_pool[random_career], min(num_other, len(career_interest_pool[random_career])))
                interests = related_interests + other_interests
            else:
                interests = related_interests
        else:
            # Completely random interests
            all_interests = []
            for pool in career_interest_pool.values():
                all_interests.extend(pool)
            
            num_interests = random.randint(3, 7)
            interests = random.sample(all_interests, min(num_interests, len(all_interests)))
        
        # Combine interests into a single text entry
        interests_text = ". ".join(interests)
        
        # Add to dataset
        data.append({
            'interests_text': interests_text,
            'career': career
        })
    
    return pd.DataFrame(data)

# Generate the dataset
dataset = generate_text_based_dataset(1000)
dataset.to_csv('career_interests_text_dataset.csv', index=False)
print("Text-based synthetic dataset created and saved as 'career_interests_text_dataset.csv'")

# -------------------------
# Step 2: Data Preprocessing for Text Interests
# -------------------------

# Create a text processing pipeline
text_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=1000, stop_words='english')),
])

# Prepare X (features) and y (target)
X = dataset['interests_text']
y = dataset['career']

# Encode the target variable (careers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Transform the text data
X_train_transformed = text_pipeline.fit_transform(X_train)
X_test_transformed = text_pipeline.transform(X_test)

# -------------------------
# Step 3: Train the Random Forest Model
# -------------------------

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_transformed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_transformed)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature importance (for the vectorized features)
feature_names = text_pipeline.named_steps['vectorizer'].get_feature_names_out()
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Feature Words:")
print(feature_importance.head(10))

# -------------------------
# Step 4: Career Recommendation Function with User Input
# -------------------------

def recommend_career_from_text():
    """
    Get user interests as free text and recommend careers
    """
    print("\n==== Career Recommendation System ====")
    print("Please describe your interests, skills, and what you enjoy doing.")
    print("\nYou might want to mention your interests related to:")
    print("- Technical topics (programming, mathematics, engineering, science)")
    print("- Creative pursuits (art, design, writing, music)")
    print("- Working with people (teaching, healthcare, counseling)")
    print("- Business and finance (economics, investment, marketing)")
    print("- Natural world (environment, biology, conservation)")
    
    # Give example to help users understand the expected input format
    print("\nFor example: 'I enjoy programming and solving mathematical problems. ")
    print("I'm interested in AI and data analysis. I also like designing visual interfaces.'")
    
    interests_text = input("\nYour interests: ").strip()
    
    if not interests_text:
        print("No interests provided. Please try again.")
        return None
    
    # Transform the input text using the same pipeline
    user_input_transformed = text_pipeline.transform([interests_text])
    
    # Get career predictions with probabilities
    probs = model.predict_proba(user_input_transformed)[0]
    sorted_idx = np.argsort(probs)[::-1]
    
    # Get top 3 career recommendations
    top_careers = [(le.classes_[sorted_idx[i]], probs[sorted_idx[i]]) for i in range(3)]
    
    # Display results
    print("\n==== Your Career Recommendations ====")
    
    for i, (career, prob) in enumerate(top_careers, 1):
        print(f"\n{i}. {career} ({prob:.2%} match)")
        
        # Display career information
        if career in career_info:
            info = career_info[career]
            print(f"   Description: {info['description']}")
            print(f"   Required Skills: {', '.join(info['required_skills'])}")
            print(f"   Education: {info['education']}")
            print(f"   Salary Range: {info['salary_range']}")
            print(f"   Growth Prospects: {info['growth_prospects']}")
            
            # Analyze which keywords from the input match this career
            keywords = info['related_keywords']
            matching_keywords = [keyword for keyword in keywords 
                               if keyword.lower() in interests_text.lower()]
            
            if matching_keywords:
                print(f"   Matched Keywords: {', '.join(matching_keywords)}")
        else:
            print("   Detailed information not available for this career.")
    
    return top_careers

# -------------------------
# Step 5: Save Model and Components
# -------------------------

# Create a full pipeline including the text processing
full_pipeline = Pipeline([
    ('text_processor', text_pipeline),
    ('classifier', model)
])

# Save the complete pipeline and other components
joblib.dump(full_pipeline, 'career_recommendation_pipeline.pkl')
joblib.dump(le, 'career_label_encoder.pkl')
joblib.dump(career_info, 'career_info.pkl')

print("\nComplete pipeline and components saved.")

# -------------------------
# Step 6: Programmatic API Function
# -------------------------

def get_career_recommendations_from_text(interests_text):
    """
    Non-interactive function to get career recommendations based on text input
    
    Args:
        interests_text: String describing user interests
        
    Returns:
        List of dictionaries with career recommendations and details
    """
    if not interests_text:
        return None, "No interests provided"
    
    # Transform the input text
    user_input_transformed = text_pipeline.transform([interests_text])
    
    # Get career predictions with probabilities
    probs = model.predict_proba(user_input_transformed)[0]
    sorted_idx = np.argsort(probs)[::-1]
    
    # Get top 3 career recommendations with info
    recommendations = []
    for i in range(3):
        career = le.classes_[sorted_idx[i]]
        probability = probs[sorted_idx[i]]
        
        career_details = career_info.get(career, {
            'description': 'Information not available',
            'required_skills': [],
            'education': 'Information not available',
            'salary_range': 'Information not available',
            'growth_prospects': 'Information not available',
            'related_keywords': []
        })
        
        # Find matching keywords
        matching_keywords = [keyword for keyword in career_details.get('related_keywords', [])
                           if keyword.lower() in interests_text.lower()]
        
        recommendations.append({
            'career': career,
            'probability': probability,
            'info': career_details,
            'matching_keywords': matching_keywords
        })
    
    return recommendations

# Example usage
example_interests = """I love working with computers and solving complex problems. 
I enjoy programming and developing software. Mathematics has always been my strong point,
and I'm interested in data analysis and statistics."""

recommendations = get_career_recommendations_from_text(example_interests)

if recommendations:
    print("\nExample recommendations based on interests text:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['career']} ({rec['probability']:.2%} match)")
        print(f"   Description: {rec['info']['description']}")
        if rec['matching_keywords']:
            print(f"   Matching Keywords: {', '.join(rec['matching_keywords'])}")

# -------------------------
# Step 7: Create a Simple Application Entry Point
# -------------------------

if __name__ == "__main__":
    print("Career Recommendation System")
    print("1. Run the interactive recommendation system")
    print("2. Exit")
    
    choice = input("Enter your choice (1-2): ")
    
    if choice == "1":
        recommend_career_from_text()
    else:
        print("Goodbye!")


