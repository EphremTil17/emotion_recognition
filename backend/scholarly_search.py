# scholarly_search.py
import os
import sys
import json
import time
import logging
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the src directory to Python path
current_dir = Path(__file__).parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))

# Load environment variables
load_dotenv()

# Configure APIs
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Define paths
BACKEND_DIR = Path(__file__).parent
ANALYTICS_DIR = BACKEND_DIR / "analytics"
CONTENT_ANALYSIS_FILE = ANALYTICS_DIR / "content_analysis.json"
SCHOLARLY_RESULTS_FILE = ANALYTICS_DIR / "scholarly_results.json"

# Create directories if they don't exist
ANALYTICS_DIR.mkdir(exist_ok=True)

async def optimize_search_query(text, use_gemini=True):
    """Optimize the text for scholarly search with basic keyword extraction"""
    if not use_gemini or not GOOGLE_API_KEY:
        # If Gemini isn't configured, extract keywords manually
        return extract_keywords(text)
    
    try:
        # Configure the model with the correct model name
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Use a simpler model that's more likely to work
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Extract 5-7 key academic concepts from this text as a simple comma-separated list:
        
        {text}
        
        Return only the comma-separated list without any other text.
        """
        
        response = model.generate_content(prompt)
        query = response.text.strip()
        
        logging.info(f"Optimized query: {query}")
        return query
    except Exception as e:
        logging.error(f"Error optimizing query with Gemini: {str(e)}")
        # On error, extract keywords manually
        return extract_keywords(text)

def extract_keywords(text):
    """Extract important keywords from text without using AI"""
    # Remove common words and punctuation
    common_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'when', 'where', 'how', 'why', 'is', 'am', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'done',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'their', 'this', 'that',
        'these', 'those', 'for', 'to', 'in', 'on', 'at', 'by', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'from', 'up', 'down', 'of', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'can', 'will', 'just', 'should', 'now', 'discuss', 'discusses', 'discussed',
        'speaker', 'talks', 'talking', 'mentioned', 'says', 'said', 'tell', 'told'
    }
    
    # Clean the text
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    words = text.split()
    
    # Filter out common words and short words
    filtered_words = [word for word in words if word not in common_words and len(word) > 3]
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(filtered_words)
    
    # Get the most common words (excluding very common words)
    top_words = [word for word, count in word_counts.most_common(10)]
    
    # Convert to a search query
    query = " ".join(top_words[:7])  # Use top 7 words
    
    logging.info(f"Extracted keywords: {query}")
    return query

async def search_scholarly_resources(query):
    """Search for scholarly resources using targeted searches for different aspects of the content"""
    try:
        # Parse the optimized query into individual keywords
        keywords = [k.strip() for k in query.split(',')]
        logging.info(f"Working with keywords: {keywords}")
        
        # If we have few keywords, add some general ones
        if len(keywords) < 3:
            keywords.extend(["research", "study", "analysis"])
        
        # Create combinations of keywords focusing on different aspects
        keyword_groups = []
        
        # If we have enough keywords, create meaningful combinations
        if len(keywords) >= 3:
            # Theme 1: First two keywords (main focus)
            keyword_groups.append({
                "theme": f"{keywords[0]} and {keywords[1]}",
                "keywords": f"{keywords[0]} {keywords[1]}"
            })
            
            # Theme 2: First keyword with third keyword
            keyword_groups.append({
                "theme": f"{keywords[0]} related to {keywords[2]}",
                "keywords": f"{keywords[0]} {keywords[2]}"
            })
            
            # Theme 3: First three keywords together
            keyword_groups.append({
                "theme": f"{keywords[0]}, {keywords[1]} and {keywords[2]}",
                "keywords": f"{keywords[0]} {keywords[1]} {keywords[2]}"
            })
            
            # Theme 4: Keywords related to future (if applicable) or another combination
            future_keywords = [k for k in keywords if k.lower() in ["future", "innovation", "transformation", "development"]]
            if future_keywords:
                future_kw = future_keywords[0]
                keyword_groups.append({
                    "theme": f"Future of {keywords[0]}",
                    "keywords": f"{keywords[0]} {future_kw} trends"
                })
            else:
                # Alternative: Use a different combination
                mid_point = len(keywords) // 2
                keyword_groups.append({
                    "theme": f"{keywords[mid_point]} research",
                    "keywords": f"{keywords[mid_point]} {keywords[mid_point-1]} research"
                })
            
            # Theme 5: Practical applications or implications
            applications_keywords = [k for k in keywords if k.lower() in ["application", "practice", "implementation", "industry", "education", "collaboration"]]
            if applications_keywords:
                app_kw = applications_keywords[0]
                keyword_groups.append({
                    "theme": f"Applications of {keywords[0]} in {app_kw}",
                    "keywords": f"{keywords[0]} {app_kw} practical examples"
                })
            else:
                # Alternative: Educational perspective
                keyword_groups.append({
                    "theme": f"Teaching and Learning about {keywords[0]}",
                    "keywords": f"{keywords[0]} education teaching learning"
                })
        else:
            # If we have too few keywords, use the whole query and add some variations
            keyword_groups = [
                {"theme": f"Research on {query}", "keywords": f"{query} research"},
                {"theme": f"Applications of {query}", "keywords": f"{query} applications"},
                {"theme": f"Future of {query}", "keywords": f"{query} future trends"},
                {"theme": f"{query} in Education", "keywords": f"{query} education teaching"},
                {"theme": f"{query} Case Studies", "keywords": f"{query} case studies examples"}
            ]
        
        # Create results with carefully constructed search URLs
        results = []
        
        # Add a Google Scholar search for each keyword group
        for i, group in enumerate(keyword_groups):
            if i >= 5:  # Limit to 5 results
                break
                
            # Clean up the keywords for URL
            search_terms = group["keywords"].replace(" ", "+")
            
            # Select different academic sources based on index
            if i == 0:
                # First result: Google Scholar (recent papers)
                results.append({
                    "title": f"Recent Research: {group['theme']}",
                    "authors": "Google Scholar",
                    "year": "Recent papers",
                    "venue": "Academic Search",
                    "url": f"https://scholar.google.com/scholar?as_ylo=2020&q={search_terms}",
                    "abstract": f"Recent academic papers on {group['theme']} published since 2020."
                })
            elif i == 1:
                # Second result: arXiv (for technical/scientific papers)
                results.append({
                    "title": f"Scientific Papers: {group['theme']}",
                    "authors": "arXiv.org",
                    "year": "Preprints and papers",
                    "venue": "Repository",
                    "url": f"https://arxiv.org/search/?query={search_terms}&searchtype=all",
                    "abstract": f"Scientific and technical research papers about {group['theme']} from the arXiv repository."
                })
            elif i == 2:
                # Third result: Open access journals
                results.append({
                    "title": f"Open Access Articles: {group['theme']}",
                    "authors": "Directory of Open Access Journals",
                    "year": "Open access",
                    "venue": "DOAJ",
                    "url": f"https://doaj.org/search/articles?source=%7B%22query%22%3A%7B%22query_string%22%3A%7B%22query%22%3A%22{search_terms}%22%2C%22default_operator%22%3A%22AND%22%7D%7D%7D",
                    "abstract": f"Freely accessible academic articles on {group['theme']} from open access journals."
                })
            elif i == 3:
                # Fourth result: Educational resources
                results.append({
                    "title": f"Educational Resources: {group['theme']}",
                    "authors": "BASE (Bielefeld Academic Search Engine)",
                    "year": "Various",
                    "venue": "Academic Search",
                    "url": f"https://www.base-search.net/Search/Results?lookfor={search_terms}&name=&oaboost=1&newsearch=1&refid=dcbasen",
                    "abstract": f"Educational and academic resources about {group['theme']} from open institutional repositories."
                })
            elif i == 4:
                # Fifth result: Research Gate (for collaboration papers)
                results.append({
                    "title": f"Research Community: {group['theme']}",
                    "authors": "ResearchGate",
                    "year": "Community research",
                    "venue": "Research Network",
                    "url": f"https://www.researchgate.net/search/publication?q={search_terms}",
                    "abstract": f"Research papers and discussions about {group['theme']} from the academic research community."
                })
        
        return results
        
    except Exception as e:
        logging.error(f"Error creating scholarly resource links: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Simple fallback with direct Google Scholar link
        scholar_query = query.replace(" ", "+")
        return [{
            "title": "Academic Research on Your Topic",
            "authors": "Google Scholar",
            "year": "Various",
            "venue": "Search Engine",
            "url": f"https://scholar.google.com/scholar?q={scholar_query}",
            "abstract": f"Find academic papers and research related to: {query}"
        }]

async def process_content_for_scholarly_search():
    """Process the latest content analysis to find scholarly resources"""
    try:
        # Check if content analysis file exists
        if not CONTENT_ANALYSIS_FILE.exists():
            logging.info("No content analysis file found")
            return False
        
        # Load content analysis
        with open(CONTENT_ANALYSIS_FILE, "r") as f:
            content_data = json.load(f)
        
        # Check if we have transcription data
        if not content_data.get("has_audio", False) or "transcription" not in content_data:
            logging.info("No transcription data available")
            return False
        
        # Extract summary or text from transcription
        query_text = ""
        
        # Try to get the summary first
        if "results" in content_data["transcription"] and "summary" in content_data["transcription"]["results"]:
            query_text = content_data["transcription"]["results"]["summary"].get("short", "")
        
        # If no summary, try to use paragraphs
        if not query_text and "results" in content_data["transcription"] and "channels" in content_data["transcription"]["results"]:
            for channel in content_data["transcription"]["results"]["channels"]:
                if "alternatives" in channel and channel["alternatives"]:
                    if "paragraphs" in channel["alternatives"][0]:
                        paragraphs = channel["alternatives"][0]["paragraphs"].get("paragraphs", [])
                        if paragraphs:
                            # Concatenate a few paragraphs
                            text_parts = []
                            for p in paragraphs[:3]:  # Use up to 3 paragraphs
                                sentences = p.get("sentences", [])
                                if sentences:
                                    text_parts.append(" ".join([s.get("text", "") for s in sentences]))
                            query_text = " ".join(text_parts)
                            break
        
        if not query_text:
            logging.info("No suitable text found for search query")
            return False
        
        logging.info(f"Using content text for search: {query_text[:100]}...")
        
        # Optimize the query using Gemini if available
        optimized_query = await optimize_search_query(query_text)
        
        # Search for scholarly resources
        search_results = await search_scholarly_resources(optimized_query)
        
        if not search_results:
            logging.error("No search results returned")
            return False
        
        # Save search results
        scholarly_data = {
            "timestamp": time.time(),
            "original_query": query_text,
            "optimized_query": optimized_query,
            "results": search_results
        }
        
        with open(SCHOLARLY_RESULTS_FILE, "w") as f:
            json.dump(scholarly_data, f, indent=2)
        
        logging.info(f"Scholarly search completed: {SCHOLARLY_RESULTS_FILE}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing content for scholarly search: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

async def main():
    """Main function to periodically check for new content and perform scholarly search"""
    logging.info("Scholarly Search Service started")
    
    # Track the last processed content analysis file
    last_modified = None
    
    try:
        while True:
            # Check if content analysis file exists and has been modified
            if CONTENT_ANALYSIS_FILE.exists():
                current_modified = CONTENT_ANALYSIS_FILE.stat().st_mtime
                
                # If file is new or modified
                if last_modified is None or current_modified > last_modified:
                    logging.info("New or updated content analysis detected")
                    
                    # Wait a moment to ensure the file is fully written
                    await asyncio.sleep(3)
                    
                    # Process the content
                    success = await process_content_for_scholarly_search()
                    if success:
                        last_modified = current_modified
            
            # Sleep before checking again (every 30 seconds)
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        logging.info("Service stopped by user")
    except Exception as e:
        logging.error(f"Service error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())