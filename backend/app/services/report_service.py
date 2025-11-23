import os
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import time
import logging
from app.middleware.privacy import sanitize_text

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Structure for agent responses"""
    agent_name: str
    content: str
    confidence: float
    processing_time: float

class HealthReportAnalyzer:
    """Enhanced health report analysis system with specialized agents"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized medical analysis agents"""
        self.agents = {
            'positive_analyzer': self._create_agent("""
                You are a positive health findings specialist.
                Identify and explain all positive health indicators in the report.
                
                IMPORTANT FORMATTING INSTRUCTIONS:
                1. Each finding MUST start on a new line with a checkmark symbol (✓)
                2. After each finding value and range, add a new line with two spaces of indentation for the significance
                3. Add a blank line between each complete finding
                
                Format each finding exactly like this:
                
                ✓ [Test Name]: [Value] [Unit] (normal range: [range])
                  Significance: [Brief explanation of why this is positive]
                
                [blank line here]
                ✓ [Next Test Name]: [Value] [Unit] (normal range: [range])
                  Significance: [Brief explanation of why this is positive]
            """),
            
            'negative_analyzer': self._create_agent("""
                You are a health risk assessment specialist.
                Identify concerning findings and potential health risks.
                Format findings as bullet points starting with "⚠".
                Each finding must be on a new line.
                Include severity levels and recommended actions.
            """),
            
            'summary_agent': self._create_agent("""
                You are a medical report summarizer.
                Create a comprehensive yet concise summary of all findings.
                Include key metrics, trends, and important observations.
                Use clear, patient-friendly language.
                Format with clear sections and bullet points.
            """),
            
            'recommendation_agent': self._create_agent("""
                You are a healthcare recommendations specialist.
                Provide actionable advice based on the report findings.
                Include lifestyle, diet, and exercise recommendations.
                Prioritize suggestions by importance and urgency.
                Format each recommendation on a new line with clear categorization.
            """),
            
            'diet_planner': self._create_agent("""
                You are a specialized medical nutritionist who creates personalized diet plans.
                
                INSTRUCTIONS:
                1. Analyze the abnormal and low conditions in the medical report
                2. For each identified condition, provide specific dietary recommendations
                3. Create a complete 7-day meal plan addressing all health concerns
                4. Include specific foods to eat and avoid for each condition
                5. Prioritize evidence-based nutritional recommendations
                
                FORMAT YOUR RESPONSE:
                
                ## CONDITIONS REQUIRING DIETARY INTERVENTION
                - [List each condition with brief explanation]
                
                ## DIETARY RECOMMENDATIONS BY CONDITION
                ### [Condition 1]
                - Foods to include: [list with benefits]
                - Foods to avoid: [list with explanation]
                
                ### [Condition 2]
                - Foods to include: [list with benefits]
                - Foods to avoid: [list with explanation]
                
                ## 7-DAY OPTIMAL MEAL PLAN
                ### Day 1
                - Breakfast: [specific meal with ingredients]
                - Lunch: [specific meal with ingredients]
                - Dinner: [specific meal with ingredients]
                - Snacks: [options]
                
                [Continue for all 7 days]
                
                ## NUTRITIONAL SUPPLEMENTS
                - [List recommended supplements if needed]
                
                ## HYDRATION RECOMMENDATIONS
                - [Specific recommendations]
            """)
        }

    def _create_agent(self, system_prompt: str):
        """Create an agent with specific system prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        return prompt | self.llm | StrOutputParser()

    def _format_findings(self, response: str) -> str:
        """Format the findings to ensure proper line breaks and spacing"""
        if not isinstance(response, str):
            return str(response)

        findings = [f.strip() for f in response.split('✓') if f.strip()]
        
        formatted_findings = []
        for finding in findings:
            lines = [line.strip() for line in finding.split('\n') if line.strip()]
            
            if lines:
                main_finding = lines[0]
                formatted_finding = [f"✓ {main_finding}"]
                
                for line in lines[1:]:
                    if line.startswith('Significance:'):
                        formatted_finding.append(f"  {line}")
                    else:
                        formatted_finding.append(f"  {line}")
                
                formatted_findings.append('\n'.join(formatted_finding))
        
        return '\n\n'.join(formatted_findings)

    async def analyze_report(self, report_text: str) -> Dict[str, AgentResponse]:
        """Analyze report using multiple agents with direct context"""
        results = {}
        
        try:
            # Sanitize PII before processing
            sanitized_text = sanitize_text(report_text)
            logger.info("PII sanitized from report before analysis")
            
            # Direct context approach - pass full text to agents
            # Llama 3.1 has 128k context, sufficient for most reports
            
            agents_list = list(self.agents.items())
            
            for agent_name, agent in agents_list:
                start_time = time.time()
                
                try:
                    # Pass the full sanitized text directly
                    response = await agent.ainvoke({"input": sanitized_text})
                    
                    if agent_name == 'positive_analyzer':
                        response = self._format_findings(response)
                    
                    # Sanitize PII from response before showing to user
                    response = sanitize_text(response)
                    
                    processing_time = time.time() - start_time
                    
                    results[agent_name] = AgentResponse(
                        agent_name=agent_name,
                        content=response,
                        confidence=0.9,
                        processing_time=processing_time
                    )
                    
                except Exception as e:
                    logger.error(f"Error in agent {agent_name}: {str(e)}")
                    results[agent_name] = AgentResponse(
                        agent_name=agent_name,
                        content=f"Error: {str(e)}",
                        confidence=0.0,
                        processing_time=0.0
                    )
            
            return results
        except Exception as e:
            logger.error(f"Error in analyze_report: {str(e)}")
            raise

    async def web_search_diet_info(self, abnormal_conditions: List[str]) -> str:
        """Search the web for diet recommendations based on abnormal conditions"""
        try:
            search_results = []
            
            for condition in abnormal_conditions:
                # Simulate web search results (as in original code)
                search_result = f"### Diet Information for {condition}\n"
                search_result += "Based on recent medical research:\n"
                search_result += "- Recommended foods: [would be populated from actual search]\n"
                search_result += "- Foods to avoid: [would be populated from actual search]\n"
                search_result += "- Recent studies suggest: [would be populated from actual search]\n\n"
                
                search_results.append(search_result)
            
            return "\n".join(search_results)
        except Exception as e:
            return f"Error searching for diet information: {str(e)}"

    async def extract_abnormal_conditions(self, report_text: str) -> List[str]:
        """Extract abnormal conditions from the report text"""
        try:
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical condition extractor.
                    Extract all abnormal test results and conditions from the provided medical report.
                    Return ONLY a list of specific conditions, one per line.
                    DO NOT include normal results.
                    Example output:
                    Low Vitamin D
                    Elevated LDL cholesterol
                    Hypothyroidism"""),
                ("human", "{report}")
            ])
            
            chain = extract_prompt | self.llm | StrOutputParser()
            
            conditions_text = await chain.ainvoke({"report": report_text})
            
            conditions = [
                cond.strip() for cond in conditions_text.split('\n')
                if cond.strip() and not cond.startswith("Normal")
            ]
            
            return conditions
        except Exception as e:
            logger.error(f"Error extracting conditions: {str(e)}")
            return []

    async def generate_diet_plan(self, report_text: str) -> str:
        """Generate comprehensive diet plan based on report findings"""
        try:
            conditions = await self.extract_abnormal_conditions(report_text)
            
            diet_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a specialized medical nutritionist.
                    Create a comprehensive diet plan addressing the specific abnormal conditions listed.
                    Include scientific rationale for each recommendation.
                    Format as:
                    1. Analysis of each condition and its nutritional implications
                    2. Specific foods to eat and avoid for each condition
                    3. A detailed 7-day meal plan with recipes
                    4. Supplement recommendations if needed"""),
                ("human", "Create a personalized diet plan for these conditions: {conditions}")
            ])
            
            chain = diet_prompt | self.llm | StrOutputParser()
            
            diet_plan = await chain.ainvoke({"conditions": "\n".join(conditions)})
            
            web_results = await self.web_search_diet_info(conditions)
            
            combined_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a medical nutritionist creating the optimal diet plan.
                    Combine the AI-generated diet plan with web research to create the most 
                    comprehensive and evidence-based recommendations.
                    Keep formatting clear with headers, bullet points, and 7-day meal plan."""),
                ("human", """
                AI Diet Plan:
                {diet_plan}
                
                Web Research:
                {web_results}
                
                Create an optimized diet plan combining this information.
                """)
            ])
            
            chain = combined_prompt | self.llm | StrOutputParser()
            
            final_diet_plan = await chain.ainvoke({
                "diet_plan": diet_plan,
                "web_results": web_results
            })
            
            return final_diet_plan
        except Exception as e:
            logger.error(f"Error creating diet plan: {str(e)}")
            return f"Error creating diet plan: {str(e)}"
