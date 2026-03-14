"""
Phase 1: Synthetic Data Generation for Home DIY Repair Q&A

This module generates 20 synthetic home repair Q&A pairs using 5 different
prompt templates (one per repair category). It uses the Instructor library
to ensure LLM outputs match the RepairQA Pydantic model structure.

About Instructor:
Instructor is a Python library that wraps OpenAI's API to enforce structured
outputs. It uses Pydantic models to guarantee the LLM returns valid JSON with
all required fields and correct types. This eliminates the need for manual
parsing and validation of LLM responses.

Generation Strategy:
- 5 prompt templates covering different repair domains
- Random template selection for diversity
- 20 total samples generated
- All outputs validated against RepairQA schema
"""

import json
import random
import os
from typing import List
from openai import OpenAI
import instructor
from dotenv import load_dotenv

from models import RepairQA

# Load environment variables from .env.local file
# About dotenv:
# python-dotenv loads environment variables from a .env file into os.environ.
# This keeps sensitive data (like API keys) out of source code and version control.
# The .env.local file should be added to .gitignore to prevent accidental commits.
load_dotenv(dotenv_path="../.env.local")

# Get API credentials from environment variables
# Using os.getenv() with error handling to ensure required variables are set
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env.local file in the mini-projects directory with your API key."
    )

# Initialize Instructor-wrapped OpenAI client
# Instructor patches the OpenAI client to support response_model parameter
client = instructor.from_openai(
    OpenAI(
        api_key=api_key,
        base_url=base_url
    )
)


# =============================================================================
# PROMPT TEMPLATES - One per repair category
# =============================================================================

PROMPT_TEMPLATES = {
    "appliance_repair": """You are an expert home appliance repair technician with 20+ years of experience.

Generate a realistic home appliance repair Q&A pair for a common household appliance issue.

Focus on:
- Common household appliances (refrigerators, washing machines, dryers, dishwashers, ovens)
- Technical details that are practical for homeowners
- Clear, actionable repair steps
- Appropriate safety warnings
- Realistic tools that homeowners typically have or can easily obtain

The question should be from a homeowner experiencing a specific appliance problem.
The answer should provide detailed, practical guidance that a homeowner can follow.
""",

    "plumbing_repair": """You are a professional plumber with extensive residential experience.

Generate a realistic home plumbing repair Q&A pair for a common plumbing issue.

Focus on:
- Common plumbing problems (leaks, clogs, fixture repairs, pipe problems)
- Safety considerations for homeowner attempts
- Realistic solutions that don't require professional-only tools
- When to call a professional vs. DIY
- Water safety and damage prevention

The question should be from a homeowner with a specific plumbing concern.
The answer should balance practical DIY guidance with appropriate safety warnings.
""",

    "electrical_repair": """You are a licensed electrician specializing in safe home electrical repairs.

Generate a realistic home electrical repair Q&A pair for SAFE homeowner-level electrical work.

CRITICAL FOCUS:
- ONLY safe homeowner-level electrical work (outlet replacement, switch repair, light fixture installation)
- Extensive safety warnings about electrical hazards
- Clear guidance on when to call a professional
- Emphasis on turning off power at breaker
- Warning signs that indicate professional help is needed

The question should be from a homeowner with a basic electrical issue.
The answer MUST prioritize safety and include strong warnings about electrical hazards.
""",

    "hvac_maintenance": """You are an HVAC technician specializing in homeowner maintenance and basic troubleshooting.

Generate a realistic HVAC maintenance Q&A pair for basic homeowner-level tasks.

Focus on:
- Basic HVAC maintenance (filter changes, thermostat issues, vent cleaning, basic troubleshooting)
- Seasonal considerations and preventive maintenance
- Energy efficiency tips
- When DIY maintenance is appropriate vs. professional service
- Cost-saving maintenance practices

The question should be from a homeowner about HVAC maintenance or a simple issue.
The answer should provide practical maintenance guidance and troubleshooting steps.
""",

    "general_home_repair": """You are a skilled handyperson with general home repair expertise.

Generate a realistic general home repair Q&A pair for common household repairs.

Focus on:
- Common repairs (drywall repair, door/window problems, flooring issues, basic carpentry)
- Material specifications and where to purchase supplies
- Practical DIY solutions with common tools
- Techniques that produce professional-looking results
- Cost-effective repair methods

The question should be from a homeowner with a general repair need.
The answer should provide detailed, practical DIY guidance with material specifications.
"""
}


def generate_repair_qa(category: str) -> RepairQA:
    """
    Generate a single repair Q&A pair using the specified category template.
    
    Args:
        category: One of the 5 repair categories (appliance_repair, plumbing_repair, etc.)
    
    Returns:
        RepairQA: Validated Pydantic model with all 7 required fields
    
    Raises:
        ValueError: If category is not recognized
        Exception: If LLM generation or validation fails
    
    About the Process:
    1. Retrieves the appropriate prompt template for the category
    2. Sends prompt to LLM via Instructor with response_model=RepairQA
    3. Instructor ensures the response matches RepairQA structure
    4. Returns validated RepairQA object (guaranteed to have all fields)
    """
    if category not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown category: {category}. Must be one of {list(PROMPT_TEMPLATES.keys())}")
    
    prompt = PROMPT_TEMPLATES[category]
    
    try:
        # Instructor enforces RepairQA structure on LLM response
        # The response_model parameter tells Instructor to validate against RepairQA
        repair_qa = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4 for higher quality synthetic data
            response_model=RepairQA,  # Pydantic model enforces structure
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates realistic home repair Q&A pairs."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            # Temperature: Controls randomness in LLM responses
            # Range: 0.0-2.0 (though 0.0-1.0 is most commonly used)
            # 0.0 = Deterministic (same input → same output)
            # 0.3 = Focused, consistent responses
            # 0.7-1.0 = Creative, diverse responses (ideal for synthetic data)
            # >1.0 = Very random, rarely used (can be incoherent)
            # We use 0.8 for varied but coherent synthetic data
            temperature=0.8,
            
            # Max Tokens: Limits the maximum length of the LLM's response
            # 1 token ≈ 4 characters or 0.75 words
            # 1000 tokens ≈ 750 words, sufficient for detailed repair instructions
            max_tokens=1000
        )
        
        return repair_qa
    
    except Exception as e:
        print(f"Error generating Q&A for category '{category}': {e}")
        raise


def generate_dataset(num_samples: int = 20, seed: int = 42) -> List[RepairQA]:
    """
    Generate a dataset of synthetic repair Q&A pairs.
    
    Args:
        num_samples: Number of Q&A pairs to generate (default: 20)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        List[RepairQA]: List of validated repair Q&A pairs
    
    Process:
    1. Sets random seed for reproducible template selection
    2. For each sample:
       - Randomly selects one of 5 repair categories
       - Generates Q&A pair using that category's template
       - Validates against RepairQA model (automatic via Instructor)
    3. Returns list of all generated samples
    
    Why Random Selection:
    Random selection ensures diversity across repair categories while maintaining
    realistic distribution. Over 20 samples, this typically gives good coverage
    of all 5 categories without forcing exact quotas.
    """
    random.seed(seed)  # For reproducible results
    
    categories = list(PROMPT_TEMPLATES.keys())
    dataset = []
    
    print(f"Generating {num_samples} synthetic repair Q&A pairs...")
    print(f"Categories: {', '.join(categories)}\n")
    
    for i in range(num_samples):
        # Randomly select a category for diversity; note use random -- Python's built-in random module
        category = random.choice(categories)
        
        print(f"Sample {i+1}/{num_samples}: Generating {category}...")
        
        try:
            repair_qa = generate_repair_qa(category)
            dataset.append(repair_qa)
            print(f"  ✓ Generated: {repair_qa.question[:60]}...")
        
        except Exception as e:
            print(f"  ✗ Failed to generate sample {i+1}: {e}")
            # Continue with next sample rather than stopping entire generation
            continue
    
    print(f"\n✓ Successfully generated {len(dataset)} samples")
    return dataset


def save_dataset(dataset: List[RepairQA], output_file: str = "data/synthetic_data.json") -> None:
    """
    Save the generated dataset to a JSON file.
    
    Args:
        dataset: List of RepairQA objects to save
        output_file: Output filename (default: data/synthetic_data.json)
    
    About Pydantic Serialization:
    Pydantic models have a .model_dump() method that converts them to dictionaries.
    This ensures proper JSON serialization of all fields, including lists and nested objects.
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert Pydantic models to dictionaries for JSON serialization
    data_dicts = [sample.model_dump() for sample in dataset]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_dicts, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset saved to {output_file}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.2f} KB")


def main():
    """
    Main execution function for Phase 1: Data Generation
    
    Steps:
    1. Generate 20 synthetic repair Q&A pairs
    2. Save to synthetic_data.json
    3. Display summary statistics
    """
    print("=" * 70)
    print("PHASE 1: SYNTHETIC DATA GENERATION")
    print("Home DIY Repair Q&A Pairs")
    print("=" * 70)
    print()
    
    # Generate dataset
    dataset = generate_dataset(num_samples=20, seed=42)
    
    # Save to JSON
    save_dataset(dataset, output_file="data/synthetic_data.json")
    
    # Display category distribution
    print("\nCategory Distribution:")
    # Note: We don't track categories in the RepairQA model, so this is just informational
    # In a production system, you might add a 'category' field to track this
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review synthetic_data.json")
    print("2. Proceed to Phase 2: Validation")


if __name__ == "__main__":
    main()
