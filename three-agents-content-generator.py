import openai
import streamlit as st
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

# Streamlit page configuration must come first
st.set_page_config(page_title="Multi-Agents Collaborative Working", page_icon="📰", layout="wide")

# Load environment variables
load_dotenv()
# openai.api_key = "------"

class Agent:
    def __init__(self, role, goal, backstory, tools=None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []

    def execute(self, prompt, model="gpt-4o", temperature=0.7):
        messages = [
            {"role": "system", "content": self.backstory},
            {"role": "user", "content": prompt}
        ]

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1500
        )

        return response["choices"][0]["message"]["content"].strip()

class Task:
    def __init__(self, description, expected_output, agent):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent

    def run(self, inputs):
        prompt = self.description.format(**inputs)
        print(f"Debugging Prompt for {self.agent.role}: {prompt}")
        return self.agent.execute(prompt)

class WebScraperAgent(Agent):
    def execute(self, website_url, topic):
        try:
            # Fetch website content
            response = requests.get(website_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Collect textual content (adjust tag selection as needed)
            text_content = " ".join([p.get_text() for p in soup.find_all('p')])
            return f"Scraped content related to {topic} from {website_url}:\n{text_content[:1500]}..."  # Limit content length
        except Exception as e:
            return f"Error occurred while scraping {website_url}: {str(e)}"
        
def generate_content(topic, temperature, website_url):
    # Define the agents

    web_scraper = WebScraperAgent(
        role="Web Scraper",
        goal="Perform a detailed analysis of the business processes by scraping data from all pages of the given website, including news, blogs, services, and products sections.",
        backstory=f"""
            You are a web scraping agent. Your primary task is to thoroughly explore and scrape all relevant text content from the given website URL {website_url}.
            Special focus should be given to the news, blogs, services, and products pages to gather comprehensive information for a detailed analysis of the business processes.
            The data you collect will be used for in-depth research and to provide valuable insights into the business operations related to the topic: {topic}.
        """
    )

    scraped_data = web_scraper.execute(website_url, topic)

    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources",
        backstory="""
            You're an expert research analyst with advanced web research skills. You excel at finding, analyzing, and synthesizing information from across the internet. You're skilled at distinguishing reliable sources from unreliable ones, fact-checking, cross-referencing information, and identifying key patterns and insights. Provide well-organized research briefs with proper citations and source verification.
        """
    )

    content_writer = Agent(
        role="Content Writer",
        goal="Transform research findings into engaging blog posts while maintaining accuracy",
        backstory="""
            You're a skilled content writer specialized in creating engaging, accessible content from technical research. You excel at maintaining the perfect balance between informative and entertaining writing, ensuring all facts and citations from the research are properly incorporated. Make complex topics approachable without oversimplifying them.
        """
    )

    # Define the tasks
    research_task = Task(
        description="""
            Conduct comprehensive research on {topic}, using the following data:
            {scraped_data}
              Evaluate credibility, fact-check, and summarize the findings into a research brief.
        """,
        expected_output="""
            A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns
        """,
        agent=senior_research_analyst
    )

    writing_task = Task(
        description="""
            Using the following research findings, create an engaging blog post:
            {research_results}

            The blog post should:
            - Transform technical information into accessible content
            - Maintain all factual accuracy and citations from the research
            - Include an attention-grabbing introduction, structured body sections, and a compelling conclusion
            - Preserve source citations in [Source: URL] format
            - Include a References section at the end
        """,
        expected_output="""
            A polished blog post in markdown format that:
            - Engages readers while maintaining accuracy
            - Contains properly structured sections
            - Includes inline citations hyperlinked to the original source URL
            - Follows proper markdown formatting, using H1 for the title and H3 for sub-sections
        """,
        agent=content_writer
    )

    # Run tasks
    research_results = research_task.run(inputs={"topic": topic, "scraped_data": scraped_data})
    blog_post = writing_task.run(inputs={"research_results": research_results})

    return blog_post

# Apply global CSS for font size
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.header("📰 Multi-Agents Collaborative Working")
st.markdown("Agent-1: Web scraper")
st.markdown("Agent-2: Research Analyst")
st.markdown("Agent-3: Content Writer")

# Sidebar
# Sidebar
with st.sidebar:
    # Centered logo
    st.title("About Project")
    st.markdown("Three agents work collaboratively to analyze information from a given website.")


    # Header for topic input
    st.header("Enter website and topic:")

    website_url = st.text_input("Enter website URL:", placeholder="https://example.com")
    # Text area for topic input
    topic = st.text_area(
        "",
        height=100,
        placeholder="Enter the topic you want to generate content about..."
    )

    # Slider for temperature
    temperature = st.slider("Temperature", 0.0, 1.0, 0.6)

    # Generate button
    generate_button = st.button("Generate Content")


# Main content area
if generate_button:
    if not topic.strip() or not website_url.strip():
        st.error("Please enter both a topic and a website URL.")
    else:
        with st.spinner('Generating content... This may take a moment.'):
            try:
                # Generate the content and store it in session state
                result = generate_content(topic, temperature, website_url)
                st.session_state["generated_content"] = result  # Save content to session state
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display the generated content if available in session state
if "generated_content" in st.session_state and st.session_state["generated_content"]:
    st.markdown("### Generated Content")
    st.markdown(st.session_state["generated_content"])

    # Download button
    st.download_button(
        label="Download Content",
        data=st.session_state["generated_content"],
        file_name=f"{topic.lower().replace(' ', '_')}_article.md",
        mime="text/markdown"
    )


# Footer
st.markdown("---")
st.markdown("Built with OpenAI GPT-4 and Streamlit")
