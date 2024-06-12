"""
This script sets up and executes a CrewAI framework to collect and analyze product reviews for a specific product type. 
The primary functionalities of this script include:

You will also need to setup crewai - www.crewai.com

You will need to use a local OLLAM model, or setup your own model keys.
You will also need an exasearch key - https://exa.ai/
You will also need a serpapi key https://serper.dev/

1. **Environment Setup**: 
   - Loads environment variables from a `.env` file using the `dotenv` package.
   - Defines the product type, review attributes, source hints, and types to avoid for collecting reviews.

2. **Custom Tools Implementation**:
   - **MarkdownTableTool**: Generates a markdown table from structured data and saves it to a specified file.
   - **CSVWriterTool**: Generates a CSV file from structured data and saves it to a specified file.
   - **CustomScrapeWebsiteTool**: Extends the ScrapeWebsiteTool to increment a global counter (`total_sources`) each time it is used, to track the total number of sources utilized.

3. **Agent Creation**:
   - Defines agents for data collection (`data_collector`) and data analysis (`data_analyzer`) with specific goals and tools.
   - Agents use tools to gather and process data.

4. **Task Creation**:
   - **collect_reviews_task**: Collects product reviews based on specified attributes and source hints.
   - **analyze_reviews_markdown_task**: Analyzes collected reviews and generates a markdown table.
   - **analyze_reviews_csv_task**: Analyzes collected reviews and generates a CSV file.
   - Tasks are designed to work sequentially, where data collected by one task is processed by subsequent tasks.

5. **Crew Execution**:
   - Creates and configures a `Crew` with defined agents and tasks.
   - Executes the crew to perform the entire workflow of data collection, analysis, and output generation.
   - Logs the total number of sources used during the process and the time taken for execution.

6. **Example Usage**:
   - The script includes commented-out examples for various product types (e.g., hosting providers, t-shirt companies, LLM hosting providers, smartphones, etc.) showing how to set up and execute the crew for different review scenarios.

**How to Use**:
1. Ensure all necessary environment variables are defined in a `.env` file.
2. Customize the `product_type`, `review_attributes`, `source_hints`, and `avoid_types` as needed.
3. Run the script to collect and analyze reviews for the specified product type.
4. Check the output markdown and CSV files for the generated analysis.
"""


import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from loguru import logger
from dotenv import load_dotenv

from crewai_tools.tools import (
    FileReadTool,
    DirectoryReadTool,
    DirectorySearchTool,
    SerperDevTool,
    ScrapeWebsiteTool,
    EXASearchTool,
    
)






# Load environment variables from .env file
load_dotenv()



# Example usage for routers
product_type = 'NETGEAR RAXE300 Reviews'
review_attributes = 'price range, ratings, title, urls, name, ports available, and features description, mobile dropping connection issues'
source_hints = 'like tech review sites, amazon reviews,'
avoid_types = 'Do Not look at other devices.'



#outputs
markdown_output = 'router_review.md'
csv_output = 'router_review.csv'



# Custom implementation of MarkdownTableTool
class MarkdownTableTool(BaseTool):
    name: str = "MarkdownTableTool"
    description: str = (
        "Generates a markdown table from structured data. "
        "This tool accepts a list of dictionaries where each dictionary represents a row of data. "
        "The keys of the dictionaries should be consistent to ensure proper formatting. "
        "Options are available to specify the output file name and whether to append to an existing file or overwrite it. "
        "The resulting markdown table will be saved to the specified file."
    )

    def _run(self, data: list, file_name: str = markdown_output, append: bool = False) -> str:
        logger.debug(f"Generating markdown table. Data: {data}, File: {file_name}, Append: {append}")
        df = pd.DataFrame(data)
        markdown_table = df.to_markdown(index=False)
        
        mode = 'a' if append else 'w'
        with open(file_name, mode) as f:
            if append:
                f.write('\n\n')
            f.write(markdown_table)
        
        logger.info(f"Markdown table generated and saved as {file_name}")
        return f"Markdown table generated and saved as {file_name}"

# Custom implementation of CSVWriterTool
class CSVWriterTool(BaseTool):
    name: str = "CSVWriterTool"
    description: str = "Generates a CSV file from structured data."

    def _run(self, data: list, file_name: str = csv_output) -> str:
        logger.debug(f"Generating CSV file. Data: {data}, File: {file_name}")
        df = pd.DataFrame(data)
        df.to_csv(file_name, index=False)
        logger.info(f"CSV file generated and saved as {file_name}")
        return f"CSV file generated and saved as {file_name}"

# Function to create agents
def create_agents(data_collector_goal, data_analyzer_goal, data_collector_tools, data_analyzer_tools):
    logger.info("Creating agents.")
    data_collector = Agent(
        role='Data Collector',
        goal=data_collector_goal,
        backstory='An efficient agent focused on gathering relevant data from various sources . Use all your tools',
        tools=data_collector_tools,
        verbose=True
    )

    data_analyzer = Agent(
        role='Data Analyzer',
        goal=data_analyzer_goal,
        backstory='A detailed-oriented agent responsible for analyzing data and creating structured outputs.',
        tools=data_analyzer_tools,
        verbose=True
    )

    logger.info("Agents created successfully.")
    return data_collector, data_analyzer

# Function to create tasks
def create_tasks(data_collector, data_analyzer, product_type, review_attributes):
    logger.info(f"Creating tasks for {product_type}.")
    collect_reviews_task = Task(
        description=f'Collect product reviews for the top {product_type}, including {review_attributes}. Use these type of {source_hints}',
        expected_output='A JSON object containing the collected reviews.',
        agent=data_collector,
        tools=[ScrapeWebsiteTool()]
    )

    analyze_reviews_markdown_task = Task(
        description=f'Analyze collected reviews and generate a markdown table for {product_type}, including {review_attributes}. Be sure to avoide {avoid_types}',
        expected_output='A markdown table saved to reviews.md.',
        agent=data_analyzer,
        tools=[MarkdownTableTool()],
        context=[collect_reviews_task],
        output_file=markdown_output
    )

    analyze_reviews_csv_task = Task(
        description=f'Analyze collected reviews of {product_type}, including {review_attributes}. Be sure to avoide {avoid_types} and generate a CSV file.',
        expected_output='A CSV file saved to reviews.csv.',
        agent=data_analyzer,
        tools=[CSVWriterTool()],
        context=[collect_reviews_task],
        output_file=csv_output
    )

    logger.info("Tasks created successfully.")
    return collect_reviews_task, analyze_reviews_markdown_task, analyze_reviews_csv_task

# Function to create and kickoff the crew
def create_and_kickoff_crew(product_type, review_attributes):
    logger.info(f"Starting crew creation for {product_type}.")
    
    data_collector_goal = f'Collect product reviews for the top {product_type}, including {review_attributes}. Be sure to avoide {avoid_types}'
    data_analyzer_goal = f'Analyze collected reviews and generate markdown and CSV outputs for {product_type}, including {review_attributes}. Be sure to avoide {avoid_types}'
    
    data_collector_tools = [SerperDevTool(), EXASearchTool(), ScrapeWebsiteTool()]
    data_analyzer_tools = [MarkdownTableTool(), CSVWriterTool()]
    
    data_collector, data_analyzer = create_agents(data_collector_goal, data_analyzer_goal, data_collector_tools, data_analyzer_tools)
    
    collect_reviews_task, analyze_reviews_markdown_task, analyze_reviews_csv_task = create_tasks(data_collector, data_analyzer, product_type, review_attributes)
    
    reviews_crew = Crew(
        agents=[data_collector, data_analyzer],
        tasks=[collect_reviews_task, analyze_reviews_markdown_task, analyze_reviews_csv_task],
        process=Process.sequential,
        verbose=2
    )
    
    result = reviews_crew.kickoff()
    
    logger.info("Crew kickoff completed.")
    return result



result = create_and_kickoff_crew(product_type, review_attributes)
print(result)

"""
# Example usage for hosting providers
product_type = 'hosting providers'
review_attributes = 'price, features, support, uptime, and customer reviews'
markdown_output = 'hosting_providers_review.md'
csv_output = 'hosting_providers_review.csv'

# Example usage for t-shirt companies
product_type = 't-shirt companies'
review_attributes = 'price, quality, material, customer reviews, and available sizes'
markdown_output = 'tshirt_companies_review.md'
csv_output = 'tshirt_companies_review.csv'

# Example usage for LLM hosting providers
product_type = 'LLM hosting providers'
review_attributes = 'price, performance, features, support, and customer reviews'
markdown_output = 'llm_hosting_providers_review.md'
csv_output = 'llm_hosting_providers_review.csv'

# Example usage for smartphones
product_type = 'smartphones'
review_attributes = 'price, performance, battery life, camera quality, customer reviews, and available features'
markdown_output = 'smartphones_review.md'
csv_output = 'smartphones_review.csv'

# Example usage for laptops
product_type = 'laptops'
review_attributes = 'price, performance, battery life, build quality, customer reviews, and available features'
markdown_output = 'laptops_review.md'
csv_output = 'laptops_review.csv'

# Example usage for online course platforms
product_type = 'online course platforms'
review_attributes = 'price, course quality, instructor support, platform usability, customer reviews, and available courses'
markdown_output = 'online_course_platforms_review.md'
csv_output = 'online_course_platforms_review.csv'

# Example usage for software as a service (SaaS)
product_type = 'SaaS applications'
review_attributes = 'price, features, support, usability, integration capabilities, and customer reviews'
markdown_output = 'saas_applications_review.md'
csv_output = 'saas_applications_review.csv'

# Example usage for electric vehicles
product_type = 'electric vehicles'
review_attributes = 'price, range, performance, build quality, customer reviews, and available features'
markdown_output = 'electric_vehicles_review.md'
csv_output = 'electric_vehicles_review.csv'

# Example usage for fitness trackers
product_type = 'fitness trackers'
review_attributes = 'price, battery life, features, accuracy, build quality, and customer reviews'
markdown_output = 'fitness_trackers_review.md'
csv_output = 'fitness_trackers_review.csv'

# Example usage for streaming services
product_type = 'streaming services'
review_attributes = 'price, content library, streaming quality, usability, customer support, and customer reviews'
markdown_output = 'streaming_services_review.md'
csv_output = 'streaming_services_review.csv'

# Example usage for kitchen appliances
product_type = 'kitchen appliances'
review_attributes = 'price, performance, durability, features, customer reviews, and ease of use'
markdown_output = 'kitchen_appliances_review.md'
csv_output = 'kitchen_appliances_review.csv'

# Example usage for travel agencies
product_type = 'travel agencies'
review_attributes = 'price, service quality, customer support, offered packages, customer reviews, and reliability'
markdown_output = 'travel_agencies_review.md'
csv_output = 'travel_agencies_review.csv'

# Example usage for insurance providers
product_type = 'insurance providers'
review_attributes = 'price, coverage options, customer support, claim process, customer reviews, and reliability'
markdown_output = 'insurance_providers_review.md'
csv_output = 'insurance_providers_review.csv'
"""