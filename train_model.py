from app import CareerRecommendationSystem

# Create recommendation system
recommender = CareerRecommendationSystem()

# Train and save the model
recommender.preprocess_data()
recommender.save_model('career_recommendation_model.pkl')

print("Model trained and saved successfully!")