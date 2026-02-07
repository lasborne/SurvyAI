"""Example usage of SurvyAI agent with AutoCAD integration."""

from agent import SurvyAIAgent
from utils.logger import setup_logger

logger = setup_logger()


def example_autocad_area():
    """Example: Calculate area from AutoCAD drawing."""
    print("\n" + "="*60)
    print("Example 1: AutoCAD Area Calculation")
    print("="*60)
    
    agent = SurvyAIAgent()
    
    query = """Open survey_data.dwg and calculate the total area of all 
    closed shapes verged in red. Report the area in square meters and hectares."""
    
    result = agent.process_query(query)
    print(f"\nQuery: {query}")
    print(f"\nResponse:\n{result.get('response', 'No response')}")


def example_autocad_find_owner():
    """Example: Find owner name from survey drawing."""
    print("\n" + "="*60)
    print("Example 2: Find Property Owner")
    print("="*60)
    
    agent = SurvyAIAgent()
    
    query = """Open survey_data.dxf and find the property owner's name.
    Look for text containing 'property of' or 'landed property'."""
    
    result = agent.process_query(query)
    print(f"\nQuery: {query}")
    print(f"\nResponse:\n{result.get('response', 'No response')}")


def example_survey_analysis():
    """Example: Complete survey analysis."""
    print("\n" + "="*60)
    print("Example 3: Complete Survey Analysis")
    print("="*60)
    
    agent = SurvyAIAgent()
    
    query = """Analyze survey_data.dwg:
    1. Find all text to identify the property owner and location
    2. Calculate the total area of the surveyed land (red boundaries)
    3. Report the findings with areas in both metric and imperial units"""
    
    result = agent.process_query(query)
    print(f"\nQuery: {query}")
    print(f"\nResponse:\n{result.get('response', 'No response')}")


if __name__ == "__main__":
    print("SurvyAI Example Usage")
    print("="*60)
    print("\nREQUIREMENTS:")
    print("  - AutoCAD installed and running")
    print("  - API keys configured in .env file")
    print("  - Sample survey files (DWG, DXF)")
    print("\n" + "="*60)
    
    # Uncomment examples to run:
    # example_autocad_area()
    # example_autocad_find_owner()
    # example_survey_analysis()
    
    print("\nTo run examples, uncomment them in the script.")
