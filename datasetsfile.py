import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Expanded and more comprehensive career categories
career_categories = {
    'Technology': [
        'Software Engineer', 
        'Data Scientist', 
        'Cybersecurity Analyst', 
        'User Experience Designer', 
        'Cloud Solutions Architect', 
        'Machine Learning Engineer', 
        'DevOps Engineer', 
        'Blockchain Developer', 
        'AI Research Scientist', 
        'Network Security Specialist'
    ],
    'Science': [
        'Biologist', 
        'Physicist', 
        'Environmental Scientist', 
        'Geneticist', 
        'Chemical Engineer', 
        'Astronomer', 
        'Forensic Scientist', 
        'Marine Biologist', 
        'Geologist', 
        'Zoologist'
    ],
    'Healthcare': [
        'Doctor/Physician', 
        'Psychologist', 
        'Nurse Practitioner', 
        'Pharmacist', 
        'Dentist', 
        'Physical Therapist', 
        'Veterinarian', 
        'Medical Researcher', 
        'Nutritionist', 
        'Psychiatrist'
    ],
    'Creative': [
        'Graphic Designer', 
        'Writer/Author', 
        'Multimedia Artist', 
        'Film Director', 
        'Video Game Designer', 
        'Interior Designer', 
        'Fashion Designer', 
        'Music Producer', 
        'Animator', 
        'Art Director'
    ],
    'Education': [
        'Teacher', 
        'University Professor', 
        'Educational Consultant', 
        'School Administrator', 
        'Curriculum Developer', 
        'Special Education Specialist', 
        'Corporate Trainer', 
        'Online Learning Specialist', 
        'Educational Psychologist', 
        'Museum Educator'
    ],
    'Business': [
        'Financial Analyst', 
        'Digital Marketer', 
        'Management Consultant', 
        'Entrepreneur', 
        'Investment Banker', 
        'Supply Chain Manager', 
        'Human Resources Manager', 
        'Product Manager', 
        'Business Intelligence Analyst', 
        'Startup Founder'
    ],
    'Engineering': [
        'Civil Engineer', 
        'Mechanical Engineer', 
        'Aerospace Engineer', 
        'Electrical Engineer', 
        'Robotics Engineer', 
        'Environmental Engineer', 
        'Software Engineering Manager', 
        'Structural Engineer', 
        'Chemical Process Engineer', 
        'Automotive Engineer'
    ],
    'Social Sciences': [
        'Sociologist', 
        'Anthropologist', 
        'Social Worker', 
        'Policy Analyst', 
        'Political Scientist', 
        'Urban Planner', 
        'International Relations Specialist', 
        'Human Rights Advocate', 
        'Cultural Researcher', 
        'Community Development Manager'
    ],
    'Arts and Performance': [
        'Actor', 
        'Professional Musician', 
        'Choreographer', 
        'Theatre Director', 
        'Opera Singer', 
        'Dancer', 
        'Composer', 
        'Sound Engineer', 
        'Stage Designer', 
        'Arts Administrator'
    ],
    'Media and Communication': [
        'Journalist', 
        'Public Relations Specialist', 
        'Content Creator', 
        'Media Strategist', 
        'Broadcast Journalist', 
        'Technical Writer', 
        'Podcast Producer', 
        'Social Media Manager', 
        'Communications Consultant', 
        'Copywriter'
    ],
    'Culinary and Hospitality': [
        'Chef', 
        'Restaurant Manager', 
        'Food Critic', 
        'Sommelier', 
        'Pastry Chef', 
        'Hospitality Manager', 
        'Event Planner', 
        'Food Scientist', 
        'Nutritional Consultant', 
        'Catering Manager'
    ],
    'Finance and Economics': [
        'Economist', 
        'Actuary', 
        'Financial Planner', 
        'Cryptocurrency Analyst', 
        'Risk Manager', 
        'Corporate Treasurer', 
        'Securities Trader', 
        'Real Estate Appraiser', 
        'Credit Analyst', 
        'Fundraising Manager'
    ]
}

# Function to generate synthetic career interests dataset
def generate_career_interests_dataset(num_samples=10000):
    # Initialize lists to store data
    data = {
        'age': [],
        'education_level': [],
        'gender': [],
        'stem_interest_score': [],
        'creativity_score': [],
        'social_skills_score': [],
        'analytical_skills_score': [],
        'problem_solving_score': [],
        'technical_aptitude_score': [],
        'communication_score': [],
        'leadership_potential_score': [],
        'career_category': [],
        'specific_career': [],
        'annual_salary_expectation': [],
        'work_environment_preference': [],
        'job_growth_interest': [],
        'work_life_balance_score': [],
        'global_mobility_score': []
    }
    
    # Generate synthetic data
    for _ in range(num_samples):
        # Age distribution (20-50)
        data['age'].append(round(np.random.normal(35, 10)))
        
        # Education levels (expanded)
        education_levels = [
            'High School', 
            'Associate Degree', 
            'Bachelor\'s', 
            'Master\'s', 
            'Professional Degree', 
            'PhD'
        ]
        data['education_level'].append(random.choices(
            education_levels, 
            weights=[0.1, 0.1, 0.4, 0.2, 0.1, 0.1]
        )[0])
        
        # Gender (more inclusive)
        data['gender'].append(random.choices(
            ['Male', 'Female', 'Non-Binary', 'Prefer Not to Say'], 
            weights=[0.45, 0.45, 0.05, 0.05]
        )[0])
        
        # Skills and interest scores (0-100)
        data['stem_interest_score'].append(round(np.random.normal(50, 20)))
        data['creativity_score'].append(round(np.random.normal(50, 20)))
        data['social_skills_score'].append(round(np.random.normal(50, 20)))
        data['analytical_skills_score'].append(round(np.random.normal(50, 20)))
        data['problem_solving_score'].append(round(np.random.normal(50, 20)))
        data['technical_aptitude_score'].append(round(np.random.normal(50, 20)))
        data['communication_score'].append(round(np.random.normal(50, 20)))
        data['leadership_potential_score'].append(round(np.random.normal(50, 20)))
        data['work_life_balance_score'].append(round(np.random.normal(60, 20)))
        data['global_mobility_score'].append(round(np.random.normal(50, 20)))

        
        # Career category and specific career
        category = random.choice(list(career_categories.keys()))
        data['career_category'].append(category)
        specific_career = random.choice(career_categories[category])
        data['specific_career'].append(specific_career)
        
        # Salary expectation (expanded ranges)
        salary_ranges = {
            'Software Engineer': (80000, 180000),
            'Data Scientist': (95000, 200000),
            'Doctor/Physician': (180000, 350000),
            'Graphic Designer': (45000, 100000),
            'Teacher': (45000, 90000),
            'Financial Analyst': (65000, 150000),
            'Biologist': (65000, 130000),
            'Entrepreneur': (50000, 500000),
            'Actor': (30000, 200000),
            'Journalist': (40000, 120000)
        }
        min_salary, max_salary = salary_ranges.get(specific_career, (50000, 120000))
        data['annual_salary_expectation'].append(round(np.random.normal((min_salary + max_salary)/2, 30000)))
        
        # Work environment preference (expanded)
        work_environments = ['Remote', 'Hybrid', 'On-site', 'Flexible', 'Travel-based']
        data['work_environment_preference'].append(random.choices(
            work_environments, 
            weights=[0.3, 0.3, 0.2, 0.1, 0.1]
        )[0])
        
        # Job growth interest (0-100)
        data['job_growth_interest'].append(round(np.random.normal(70, 20)))

    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Handle any out-of-range values
    df['age'] = df['age'].clip(20, 50)
    score_columns = [
        'stem_interest_score', 'creativity_score', 'social_skills_score', 
        'analytical_skills_score', 'problem_solving_score', 
        'technical_aptitude_score', 'communication_score', 
        'leadership_potential_score', 'work_life_balance_score', 
        'global_mobility_score', 'job_growth_interest'
    ]
    for score_col in score_columns:
        df[score_col] = df[score_col].clip(0, 100)
    
    return df

# Set random seed for reproducibility
np.random.seed(42)

# Generate dataset
career_dataset = generate_career_interests_dataset(10000)

# Preprocessing Pipeline
def create_preprocessing_pipeline():
    # Identify column types
    numeric_features = [
        'age', 'stem_interest_score', 'creativity_score', 
        'social_skills_score', 'analytical_skills_score', 
        'problem_solving_score', 'technical_aptitude_score', 
        'communication_score', 'annual_salary_expectation', 
        'job_growth_interest', 'leadership_potential_score',
        'work_life_balance_score', 'global_mobility_score'
    ]
    categorical_features = [
        'education_level', 'gender', 
        'career_category', 'work_environment_preference'
    ]
    
    # Create preprocessors
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Create the original dataset as a CSV
career_dataset.to_csv('expanded_career_interests_original_dataset.csv', index=False)

# Prepare features and target
X = career_dataset.drop(['specific_career'], axis=1)
y = career_dataset['specific_career']

# Create and fit preprocessor
preprocessor = create_preprocessing_pipeline()

# Fit and transform the data
X_transformed = preprocessor.fit_transform(X)

# Get feature names after transformation
numeric_features = [
    'age', 'stem_interest_score', 'creativity_score', 
    'social_skills_score', 'analytical_skills_score', 
    'problem_solving_score', 'technical_aptitude_score', 
    'communication_score', 'annual_salary_expectation', 
    'job_growth_interest', 'leadership_potential_score',
    'work_life_balance_score', 'global_mobility_score'
]
categorical_features = [
    'education_level', 'gender', 
    'career_category', 'work_environment_preference'
]

# Get feature names after one-hot encoding
onehot_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features)

# Combine feature names
feature_names = numeric_features + list(cat_feature_names)

# Convert to DataFrame
X_preprocessed = pd.DataFrame(
    X_transformed, 
    columns=feature_names
)

# Add target variable
X_preprocessed['specific_career'] = y

# Save preprocessed dataset
X_preprocessed.to_csv('expanded_career_interests_preprocessed_dataset.csv', index=False)

# Split into train and test sets
X_train, X_test = train_test_split(X_preprocessed, test_size=0.2, random_state=42)

# Save train and test sets
X_train.to_csv('expanded_career_interests_train_dataset.csv', index=False)
X_test.to_csv('expanded_career_interests_test_dataset.csv', index=False)

# Print dataset information
print("Expanded Preprocessed Dataset Information:")
print(f"Total Samples: {len(X_preprocessed)}")
print(f"Number of Features: {len(feature_names)}")
print(f"Number of Career Categories: {len(set(career_categories.keys()))}")
print(f"Total Number of Specific Careers: {sum(len(careers) for careers in career_categories.values())}")
print(f"Training Set Size: {len(X_train)}")
print(f"Testing Set Size: {len(X_test)}")

print("\nCareer Categories:")
for category, careers in career_categories.items():
    print(f"{category}: {len(careers)} careers")

print("\nDataset Files Generated:")
print("1. expanded_career_interests_original_dataset.csv")
print("2. expanded_career_interests_preprocessed_dataset.csv")
print("3. expanded_career_interests_train_dataset.csv")
print("4. expanded_career_interests_test_dataset.csv")