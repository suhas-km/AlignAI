#!/usr/bin/env python
"""
Seed script to populate the database with initial data for development.
"""

import datetime
import os
import sys
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from models import Base, User, Policy, Prompt, TokenRisk, ApiKey, prompt_policy_association
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database connection
SQLALCHEMY_DATABASE_URL = "sqlite:///./alignai.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

def seed_policies():
    """Add EU AI Act policies to the database."""
    
    policies = [
        {
            "title": "High-Risk AI Systems Transparency",
            "article": "Article 13",
            "category": "Transparency",
            "content": """High-risk AI systems shall be designed and developed in such a way to ensure that their operation is sufficiently transparent to enable users to interpret the system's output and use it appropriately. 

1. High-risk AI systems shall be accompanied by instructions for use in an appropriate digital format or otherwise that include concise, complete, correct and clear information that is relevant, accessible and comprehensible to users.

2. The instructions for use referred to in paragraph 1 shall specify:
(a) the identity and the contact details of the provider and, where applicable, of its authorised representative;
(b) the characteristics, capabilities and limitations of performance of the high-risk AI system, including:
    (i) its intended purpose;
    (ii) the level of accuracy, robustness and cybersecurity referred to in Article 15 against which the high-risk AI system has been tested and validated and which can be expected, and any known and foreseeable circumstances that may have an impact on that expected level of accuracy, robustness and cybersecurity;
    (iii) any known or foreseeable circumstance, related to the use of the high-risk AI system in accordance with its intended purpose or under conditions of reasonably foreseeable misuse, which may lead to risks to the health and safety or fundamental rights;
    (iv) its performance as regards the persons or groups of persons on which the system is intended to be used;
    (v) when appropriate, specifications for the input data, or any other relevant information in terms of the training, validation and testing data sets used, taking into account the intended purpose of the AI system.""",
            "summary": "Requires high-risk AI systems to be transparent and provide detailed documentation on their capabilities and limitations.",
            "risk_level": "high",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023R2805"
        },
        {
            "title": "Prohibition of AI Systems for Social Scoring",
            "article": "Article 5.1(c)",
            "category": "Prohibited Practices",
            "content": """The placing on the market, putting into service or use of AI systems by public authorities or on their behalf for the evaluation or classification of the trustworthiness of natural persons over a certain period of time based on their social behaviour or known or predicted personal or personality characteristics, where the social score leads to either or both of the following:
(i) detrimental or unfavourable treatment of certain natural persons or whole groups thereof in social contexts which are unrelated to the contexts in which the data was originally generated or collected;
(ii) detrimental or unfavourable treatment of certain natural persons or whole groups thereof that is unjustified or disproportionate to their social behaviour or its gravity;""",
            "summary": "Prohibits AI systems used for social scoring by public authorities that lead to discriminatory outcomes.",
            "risk_level": "high",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023R2805"
        },
        {
            "title": "Data Governance Measures",
            "article": "Article 10",
            "category": "Data Quality",
            "content": """1. High-risk AI systems which make use of techniques involving the training of models with data shall be developed on the basis of training, validation and testing data sets that meet the quality criteria referred to in paragraphs 2 to 5.

2. Training, validation and testing data sets shall be subject to appropriate data governance and management practices. Those practices shall concern in particular:
(a) the relevant design choices;
(b) data collection;
(c) relevant data preparation processing operations, such as annotation, labelling, cleaning, enrichment and aggregation;
(d) the formulation of relevant assumptions, notably with respect to the information that the data are supposed to measure and represent;
(e) a prior assessment of the availability, quantity and suitability of the data sets that are needed;
(f) examination in view of possible biases that are likely to affect health and safety of natural persons or lead to discrimination prohibited by Union law;
(g) the identification of any possible data gaps or shortcomings, and how those gaps and shortcomings can be addressed.""",
            "summary": "Requires high-quality data governance for training AI systems, including measures to prevent bias and discrimination.",
            "risk_level": "medium",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023R2805"
        },
        {
            "title": "Risk Management System",
            "article": "Article 9",
            "category": "Risk Management",
            "content": """1. A risk management system shall be established, implemented, documented and maintained in relation to high-risk AI systems.

2. The risk management system shall consist of a continuous iterative process run throughout the entire life cycle of a high-risk AI system, requiring regular systematic updating. It shall comprise the following steps:
(a) identification and analysis of the known and foreseeable risks associated with each high-risk AI system;
(b) estimation and evaluation of the risks that may emerge when the high-risk AI system is used in accordance with its intended purpose and under conditions of reasonably foreseeable misuse;
(c) evaluation of other possibly arising risks based on the analysis of data gathered from the post-market monitoring system referred to in Article 61;
(d) adoption of suitable risk management measures in accordance with the provisions of the following paragraphs.""",
            "summary": "Mandates continuous risk management processes for high-risk AI systems throughout their lifecycle.",
            "risk_level": "medium",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023R2805"
        },
        {
            "title": "Human Oversight",
            "article": "Article 14",
            "category": "Oversight",
            "content": """1. High-risk AI systems shall be designed and developed in such a way, including with appropriate human-machine interface tools, that they can be effectively overseen by natural persons during the period in which the AI system is in use.

2. Human oversight shall aim at preventing or minimising the risks to health, safety or fundamental rights that may emerge when a high-risk AI system is used in accordance with its intended purpose or under conditions of reasonably foreseeable misuse, in particular when such risks persist notwithstanding the application of other requirements set out in this Chapter.

3. Human oversight shall be ensured through either one or all of the following measures:
(a) identified and built, when technically feasible, into the high-risk AI system by the provider before it is placed on the market or put into service;
(b) identified by the provider before placing the high-risk AI system on the market or putting it into service and that are appropriate to be implemented by the user.""",
            "summary": "Requires that high-risk AI systems be designed to allow for effective human oversight during operation.",
            "risk_level": "medium",
            "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32023R2805"
        }
    ]
    
    for policy_data in policies:
        policy = Policy(
            title=policy_data["title"],
            article=policy_data["article"],
            category=policy_data["category"],
            content=policy_data["content"],
            summary=policy_data["summary"],
            risk_level=policy_data["risk_level"],
            source_url=policy_data["source_url"],
            created_at=datetime.datetime.utcnow()
        )
        db.add(policy)
    
    db.commit()
    print(f"Added {len(policies)} policies to the database.")

def seed_users():
    """Add test users to the database."""
    
    users = [
        {
            "email": "admin@alignai.com",
            "username": "admin",
            "password": "admin123",  # Would use stronger password in production
            "is_admin": True
        },
        {
            "email": "user@example.com",
            "username": "testuser",
            "password": "test123",  # Would use stronger password in production
            "is_admin": False
        }
    ]
    
    for user_data in users:
        user = User(
            email=user_data["email"],
            username=user_data["username"],
            hashed_password=pwd_context.hash(user_data["password"]),
            is_active=True,
            is_admin=user_data["is_admin"],
            created_at=datetime.datetime.utcnow()
        )
        db.add(user)
    
    db.commit()
    print(f"Added {len(users)} users to the database.")

def seed_prompts():
    """Add sample prompts with analysis to the database."""
    
    # Get the first user
    user = db.query(User).first()
    
    # Get some policies
    policies = db.query(Policy).all()
    
    prompts = [
        {
            "content": "Create a system that monitors employee social media activity and assigns a trust score that can affect their promotion opportunities.",
            "risk_score": 0.85,
            "analysis_summary": "This prompt requests the creation of a social scoring system for employees based on social media, which could violate Article 5.1(c) of the EU AI Act that prohibits social scoring systems.",
            "token_risks": [
                {
                    "start_pos": 34,
                    "end_pos": 57,
                    "risk_type": "prohibited_practice",
                    "risk_score": 0.9,
                    "explanation": "Monitoring social media for scoring purposes may violate privacy rights."
                },
                {
                    "start_pos": 62,
                    "end_pos": 73,
                    "risk_type": "prohibited_practice",
                    "risk_score": 0.95,
                    "explanation": "Assigning trust scores to individuals is explicitly prohibited for public authorities and could be problematic in private contexts as well."
                }
            ],
            "related_policies": [1]  # Index of the social scoring policy
        },
        {
            "content": "Develop an AI assistant that helps customers find products on our e-commerce site based on their preferences.",
            "risk_score": 0.15,
            "analysis_summary": "This prompt requests a low-risk AI system for product recommendations, which is not specifically regulated under the EU AI Act as high-risk. Basic transparency and data governance best practices should still apply.",
            "token_risks": [
                {
                    "start_pos": 63,
                    "end_pos": 85,
                    "risk_type": "data_quality",
                    "risk_score": 0.2,
                    "explanation": "Should ensure preference data is collected with proper consent and stored securely."
                }
            ],
            "related_policies": [2]  # Index of the data governance policy
        }
    ]
    
    for i, prompt_data in enumerate(prompts):
        prompt = Prompt(
            user_id=user.id,
            content=prompt_data["content"],
            risk_score=prompt_data["risk_score"],
            analysis_summary=prompt_data["analysis_summary"],
            created_at=datetime.datetime.utcnow()
        )
        db.add(prompt)
        db.flush()  # Get the prompt ID
        
        # Add token risks
        for risk_data in prompt_data["token_risks"]:
            token_risk = TokenRisk(
                prompt_id=prompt.id,
                start_pos=risk_data["start_pos"],
                end_pos=risk_data["end_pos"],
                risk_type=risk_data["risk_type"],
                risk_score=risk_data["risk_score"],
                explanation=risk_data["explanation"]
            )
            db.add(token_risk)
        
        # Add related policies
        for policy_index in prompt_data["related_policies"]:
            if policy_index < len(policies):
                prompt.policies.append(policies[policy_index])
    
    db.commit()
    print(f"Added {len(prompts)} prompts with analysis to the database.")

def seed_api_keys():
    """Add API keys for test users."""
    
    # Get the users
    users = db.query(User).all()
    
    for user in users:
        api_key = ApiKey(
            user_id=user.id,
            key=os.urandom(32).hex(),  # Generate a random API key
            name=f"{user.username}'s API Key",
            is_active=True,
            created_at=datetime.datetime.utcnow()
        )
        db.add(api_key)
    
    db.commit()
    print(f"Added API keys for {len(users)} users.")

def main():
    """Main seeding function."""
    try:
        # Check if tables exist
        inspector = inspect(engine)
        if "users" not in inspector.get_table_names() or "policies" not in inspector.get_table_names():
            print("Tables don't exist. Run migrations first with 'alembic upgrade head'")
            sys.exit(1)
        
        # Check if data already exists
        if db.query(Policy).count() > 0:
            print("Data already exists in the database. Skipping seeding.")
            return
        
        seed_policies()
        seed_users()
        seed_prompts()
        seed_api_keys()
        
        print("Database seeded successfully!")
    
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main()
