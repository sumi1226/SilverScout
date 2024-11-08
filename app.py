from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
import pandas as pd


embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
sentiment_analyzer = pipeline("sentiment-analysis", device=0)
candidates_df = pd.read_csv("dummy_candidates_withmodule.csv")

print("Available columns in candidates_df:", candidates_df.columns)
app = FastAPI()

class JobDescription(BaseModel):
    Job_Title: str
    Responsibilities: str
    Required_Skills: str
    Experience_Level: str
    Qualifications: str
    job_code: str
    location: str
    level: str
    Required_experience: str
    Modules: List[str]
    Min_module_score: float

class CandidateRecommendation(BaseModel):
    name: str
    score: float
    explanation: List[str]
    candidate_location: str
    candidate_level: str
    candidate_jobcode: str

@app.post("/recommend_candidates/", response_model=List[CandidateRecommendation])
def recommend_candidates(new_job_description: JobDescription):
    job_description = new_job_description.dict()
    print(job_description)

    top_candidates = suggest_top_candidates_with_advanced_filtering(
        job_description,
        skill_weights={},
        top_n=5 # Do we want to increase the number of recommendations?
    )
    

    recommendations = [
        CandidateRecommendation(
            name=candidate["name"],
            score=candidate["score"],
            module_scores=candidate["module_scores"],
            modules_interview=candidate["modules_interview"],
            candidate_location=candidate["location"],
            candidate_level=candidate["level"],
            candidate_jobcode=candidate["jobcode"],
            explanation=candidate["explanation"]
            
        ) for candidate in top_candidates
    ]
    print(recommendations)
    return recommendations

def generate_embedding(text):
    if not isinstance(text, str):
        text = "" 
    inputs = embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = embedding_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding.flatten()

def get_sentiment_score(feedback):
    if not isinstance(feedback, str) or feedback.strip() == "":
        return 0.5  # Neutral score if no feedback
    print(feedback)
    sentiment_result = sentiment_analyzer(feedback)[0]
    return 1 if sentiment_result["label"] == "POSITIVE" else 0

def calculate_module_and_experience_fit(candidate, required_experience, modules_required, min_module_score):
    experience_text = candidate.get("experience", "")
    experience_similarity = cosine_similarity(
        [generate_embedding(required_experience)], 
        [generate_embedding(experience_text)]
    )[0][0]

    candidate_modules = set(eval(candidate.get("modules_interview", "[]")))  # Convert string to list
    modules_matched = len(candidate_modules.intersection(modules_required)) / len(modules_required)

    
    # Filter candidates based on minimum module score
    candidate_module_score = float(candidate.get("module_scores", 0))
    if candidate_module_score < min_module_score:
        modules_matched = 0  # Set to 0 if candidate's score is below the threshold
    
    print(f"Experience Similarity: {experience_similarity}, Modules Matched: {modules_matched}")
    return experience_similarity, modules_matched

def suggest_top_candidates_with_advanced_filtering(new_job_description, skill_weights, top_n=5):
    job_code = new_job_description["job_code"]
    location = new_job_description["location"]
    level = new_job_description["level"]
    required_experience = new_job_description["Required_experience"]
    modules_required = new_job_description["Modules"]
    min_module_score = new_job_description["Min_module_score"]

    filtered_candidates = filter_candidates_by_job_attributes(candidates_df, job_code, location, level)

    job_text = prepare_job_text(new_job_description)
    candidate_scores = []
    for _, candidate in filtered_candidates.iterrows():
        try:
            score, explanation = calculate_score_with_explanation(
                new_job_description, candidate, skill_weights, required_experience, modules_required, min_module_score
            )
            candidate_scores.append({"name": candidate["name"], "score": score,
            "explanation": explanation, "module_scores": candidate["module_scores"], "modules_interview": candidate["modules_interview"],
            "location": candidate["location"], "level":candidate["level"],"jobcode": candidate["job_code"]})
            
        except Exception as e:
            continue
    
    top_candidates = sorted(candidate_scores, key=lambda x: x["score"], reverse=True)[:top_n]
    print(top_candidates)
    return top_candidates

def filter_candidates_by_job_attributes(candidates, job_code=None, location=None, level=None):
    filters = []
    if job_code:
        filters.append(candidates['job_code'] == job_code)
    if location:
        filters.append(candidates['location'] == location)
    if level:
        filters.append(candidates['level'] == level)

    if not filters:
        return candidates

    combined_filter = filters[0]
    for f in filters[1:]:
        combined_filter |= f

    return candidates[combined_filter]

def prepare_job_text(job):
    return (
        f"{job['Job_Title']}. Responsibilities: {job['Responsibilities']} "
        f"Required Skills: {job['Required_Skills']}. Experience Level: {job['Experience_Level']}. "
        f"Qualifications: {job['Qualifications']}"
    )

def calculate_score_with_explanation(job_description, candidate, skill_weights, required_experience, modules_required, min_module_score,alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
    
    job_text = prepare_job_text(job_description)
    job_vector = generate_embedding(job_text)
    candidate_skills_vector = generate_embedding(candidate.get("skills", ""))
    # skill_weights = {}
    # for skill in skill_weights.split(","):
    #     skill = skill.strip()
    #     if skill:
    #         weight = st.slider(f"Weight for {skill}", 0.1, 2.0, 1.0)
    #         skill_weights[skill] = weight
    skill_similarity = cosine_similarity([job_vector], [candidate_skills_vector])[0][0]
    weighted_skill_similarity =skill_similarity
    weighted_skill_similarity = sum(
        [skill_similarity * skill_weights.get(skill.strip(), 1) for skill in candidate.get("skills", "").split(",")]
    ) / len(skill_weights) if skill_weights else skill_similarity

    sentiment_score = get_sentiment_score(candidate.get("interview_feedback", ""))
    print(f"Skill Similarity: {weighted_skill_similarity}, Sentiment Score: {sentiment_score}")

    experience_similarity, modules_matched = calculate_module_and_experience_fit(candidate, required_experience, modules_required, min_module_score)
    
    
    explanations = [
        f"Skills Match: {'High' if weighted_skill_similarity > 0.7 else 'Moderate' if weighted_skill_similarity > 0.4 else 'Low'} (Matched on skills like {candidate.get('skills', '')}).",
        f"Experience Relevance: {'High' if experience_similarity > 0.7 else 'Moderate' if experience_similarity > 0.4 else 'Low'} (Relevant to the required experience).",
        f"Modules Matched: {'High' if modules_matched > 0.7 else 'Moderate' if modules_matched > 0.4 else 'Low'} (Modules: {candidate.get('modules_interview', '')}).",
        f"Interview Sentiment: {'Positive' if sentiment_score > 0.5 else 'Negative'} (Feedback analysis).",
        
    ]

    final_score = (
        alpha * weighted_skill_similarity +
        beta * sentiment_score +
        gamma * experience_similarity +
        delta * modules_matched
    )
    
    return final_score, explanations
