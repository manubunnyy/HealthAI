import os
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from groq import Groq
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
    """Enhanced health report analysis system with specialized agents using native Groq client"""
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized medical analysis agents system prompts"""
        self.agent_prompts = {
            'positive_analyzer': """
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
            """,
            
            'negative_analyzer': """
                You are a health risk assessment specialist.
                Identify concerning findings and potential health risks.
                Format findings as bullet points starting with "⚠".
                Each finding must be on a new line.
                Include severity levels and recommended actions.
            """,
            
            'summary_agent': """
                You are a medical report summarizer.
                Create a comprehensive yet concise summary of all findings.
                Include key metrics, trends, and important observations.
                Use clear, patient-friendly language.
                Format with clear sections and bullet points.
            """,
            
            'recommendation_agent': """
                You are a healthcare recommendations specialist.
                Provide actionable advice based on the report findings.
                Include lifestyle, diet, and exercise recommendations.
                Prioritize suggestions by importance and urgency.
                Format each recommendation on a new line with clear categorization.
            """,
            
            'diet_planner': """
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
            """
        }

    async def _run_agent(self, system_prompt: str, user_input: str) -> str:
        """Run a single agent using Groq API"""
        try:
            # Run in a thread pool since Groq client is synchronous (or use AsyncGroq if available, 
            # but standard Groq client is sync. We can wrap it.)
            # For simplicity and compatibility, we'll use the sync client in a thread.
            
            def _call_api():
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.3,
                    max_tokens=2048
                )
                return completion.choices[0].message.content

            return await asyncio.to_thread(_call_api)
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            raise

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

    def _chunk_text(self, text: str, chunk_size: int = 40000) -> List[str]:
        """Split text into chunks to avoid OOM"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    async def analyze_report(self, report_text: str) -> Dict[str, AgentResponse]:
        """Analyze report using Map-Reduce strategy for large files"""
        results = {}
        CHUNK_SIZE = 40000
        
        try:
            # Skip local PII sanitization (user request)
            sanitized_text = report_text
            
            # Check if we need to chunk
            if len(sanitized_text) > CHUNK_SIZE:
                logger.info(f"Large report detected ({len(sanitized_text)} chars). Using Map-Reduce chunking.")
                chunks = self._chunk_text(sanitized_text, CHUNK_SIZE)
                
                # Phase 1: Map (Extraction)
                # Run extractors on each chunk
                all_positive_findings = []
                all_negative_findings = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    # Run Positive Analyzer
                    pos_prompt = self.agent_prompts['positive_analyzer'] + "\n\nIMPORTANT PRIVACY INSTRUCTION: Anonymize all output."
                    pos_res = await self._run_agent(pos_prompt, chunk)
                    all_positive_findings.append(pos_res)
                    import gc; gc.collect()
                    
                    # Run Negative Analyzer
                    neg_prompt = self.agent_prompts['negative_analyzer'] + "\n\nIMPORTANT PRIVACY INSTRUCTION: Anonymize all output."
                    neg_res = await self._run_agent(neg_prompt, chunk)
                    all_negative_findings.append(neg_res)
                    import gc; gc.collect()
                
                # Combine findings
                combined_positive = "\n".join(all_positive_findings)
                combined_negative = "\n".join(all_negative_findings)
                
                # Format positive findings
                formatted_positive = self._format_findings(combined_positive)
                
                # Store extraction results
                results['positive_analyzer'] = AgentResponse(
                    agent_name='positive_analyzer',
                    content=sanitize_text(formatted_positive),
                    confidence=0.9,
                    processing_time=0.0
                )
                results['negative_analyzer'] = AgentResponse(
                    agent_name='negative_analyzer',
                    content=sanitize_text(combined_negative),
                    confidence=0.9,
                    processing_time=0.0
                )
                
                # Phase 2: Reduce (Synthesis)
                # Run summary and recommendations on the EXTRACTED findings, not the raw text
                # This ensures the context is small and focused
                synthesis_context = f"""
                Based on the following extracted findings from a medical report:
                
                POSITIVE FINDINGS:
                {formatted_positive}
                
                NEGATIVE FINDINGS:
                {combined_negative}
                """
                
                # Run Summary Agent
                sum_prompt = self.agent_prompts['summary_agent'] + "\n\nIMPORTANT PRIVACY INSTRUCTION: Anonymize all output."
                sum_res = await self._run_agent(sum_prompt, synthesis_context)
                results['summary_agent'] = AgentResponse(
                    agent_name='summary_agent',
                    content=sanitize_text(sum_res),
                    confidence=0.9,
                    processing_time=0.0
                )
                import gc; gc.collect()
                
                # Run Recommendation Agent
                rec_prompt = self.agent_prompts['recommendation_agent'] + "\n\nIMPORTANT PRIVACY INSTRUCTION: Anonymize all output."
                rec_res = await self._run_agent(rec_prompt, synthesis_context)
                results['recommendation_agent'] = AgentResponse(
                    agent_name='recommendation_agent',
                    content=sanitize_text(rec_res),
                    confidence=0.9,
                    processing_time=0.0
                )
                import gc; gc.collect()
                
            else:
                # Standard processing for small files
                for agent_name, system_prompt in self.agent_prompts.items():
                    privacy_instruction = "\n\nIMPORTANT PRIVACY INSTRUCTION: You are processing a medical report. STRICTLY DO NOT output the patient's name, ID, date of birth, or any other personally identifiable information (PII). Refer to the patient as 'The Patient'. Anonymize all output."
                    full_prompt = system_prompt + privacy_instruction
                    start_time = time.time()
                    
                    try:
                        # Pass the full text directly with privacy-enhanced prompt
                        response = await self._run_agent(full_prompt, sanitized_text)
                        
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
                    
                    # Force garbage collection after each agent
                    import gc
                    gc.collect()
            
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
            system_prompt = """You are a medical condition extractor.
                Extract all abnormal test results and conditions from the provided medical report.
                Return ONLY a list of specific conditions, one per line.
                DO NOT include normal results.
                Example output:
                Low Vitamin D
                Elevated LDL cholesterol
                Hypothyroidism"""
            
            conditions_text = await self._run_agent(system_prompt, report_text)
            
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
            
            diet_system_prompt = """You are a specialized medical nutritionist.
                Create a comprehensive diet plan addressing the specific abnormal conditions listed.
                Include scientific rationale for each recommendation.
                Format as:
                1. Analysis of each condition and its nutritional implications
                2. Specific foods to eat and avoid for each condition
                3. A detailed 7-day meal plan with recipes
                4. Supplement recommendations if needed"""
            
            diet_plan = await self._run_agent(diet_system_prompt, f"Create a personalized diet plan for these conditions: {' '.join(conditions)}")
            
            web_results = await self.web_search_diet_info(conditions)
            
            combined_system_prompt = """You are a medical nutritionist creating the optimal diet plan.
                Combine the AI-generated diet plan with web research to create the most 
                comprehensive and evidence-based recommendations.
                Keep formatting clear with headers, bullet points, and 7-day meal plan."""
            
            combined_input = f"""
            AI Diet Plan:
            {diet_plan}
            
            Web Research:
            {web_results}
            
            Create an optimized diet plan combining this information.
            """
            
            final_diet_plan = await self._run_agent(combined_system_prompt, combined_input)
            
            return final_diet_plan
        except Exception as e:
            logger.error(f"Error creating diet plan: {str(e)}")
            return f"Error creating diet plan: {str(e)}"
