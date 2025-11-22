import os
import io
import asyncio
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import json
import logging

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from app.middleware.privacy import sanitize_text
except ImportError:
    # Fallback if privacy module not available
    def sanitize_text(text):
        return text

@dataclass
class ProcessedDocument:
    """Structure for processed document information"""
    filename: str
    content: str
    chunks: List[str]
    total_chars: int
    doc_type: str
    summary: str = ""

@dataclass
class AgentResponse:
    """Structure for storing agent responses"""
    agent_name: str
    content: str
    confidence: float
    metadata: Dict = None
    processing_time: float = 0.0

@dataclass
class DietPlan:
    """Structure for storing diet plan information"""
    breakfast: str
    lunch: str
    dinner: str
    snacks: str
    notes: str

class DocumentProcessor:
    """Enhanced document processing with better error handling"""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.processed_documents: List[ProcessedDocument] = []
        self._initialize_embeddings()
        self.vector_store = None

    def _initialize_embeddings(self):
        """Initialize Google AI embeddings"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
                
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key,
                credentials=None
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            self.embeddings = None

    async def process_file(self, file_name: str, file_content: bytes, file_type: str, progress_callback: Callable = None) -> ProcessedDocument:
        """Process a single file"""
        try:
            if progress_callback: progress_callback(0.2, f"Processing {file_name}")
            
            if file_type == "application/pdf":
                content = await self.process_pdf(file_content)
                doc_type = "PDF"
            elif file_type.startswith("image/"):
                content = await self.process_image(file_content)
                doc_type = "Image"
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            if progress_callback: progress_callback(0.4, "Splitting content into chunks")
            chunks = self.text_splitter.split_text(content)
            
            if progress_callback: progress_callback(0.6, "Generating document summary")
            summary = await self._generate_summary(content[:1000])
            
            if progress_callback: progress_callback(0.8, "Finalizing document processing")
            
            return ProcessedDocument(
                filename=file_name,
                content=content,
                chunks=chunks,
                total_chars=len(content),
                doc_type=doc_type,
                summary=summary
            )
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            return None

    async def process_pdf(self, file_content: bytes) -> str:
        """Process PDF file"""
        text = ""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            for page_num, page in enumerate(pdf_reader.pages):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += f"Page {page_num + 1}:\n{extracted_text}\n\n"
            # Sanitize PII before returning
            text = sanitize_text(text.strip())
            logger.info("PII sanitized from PDF document")
            return text
        except Exception as e:
            raise Exception(f"PDF processing error: {str(e)}")

    async def process_image(self, file_content: bytes) -> str:
        """Process image with OCR"""
        try:
            image = Image.open(io.BytesIO(file_content))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise Exception(f"Image processing error: {str(e)}")

    async def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of the document content"""
        return f"{text[:200]}..."

    async def update_vector_store(self, documents: List[ProcessedDocument], progress_callback: Callable = None):
        """Update vector store with new documents"""
        try:
            if self.embeddings is None:
                if progress_callback: progress_callback(0.5, "ERROR: Embeddings not initialized")
                return False
                
            all_chunks = []
            metadata_list = []
            
            for idx, doc in enumerate(documents):
                if progress_callback:
                    progress_callback(0.2 + (0.6 * (idx / len(documents))), f"Indexing {doc.filename}")
                
                for chunk_idx, chunk in enumerate(doc.chunks):
                    all_chunks.append(chunk)
                    metadata_list.append({
                        "source": doc.filename,
                        "chunk_index": chunk_idx,
                        "doc_type": doc.doc_type
                    })

            if all_chunks:
                if progress_callback: progress_callback(0.8, "Creating vector store")
                self.vector_store = FAISS.from_texts(
                    all_chunks,
                    self.embeddings,
                    metadatas=metadata_list
                )
                
                if progress_callback: progress_callback(0.9, "Saving vector store")
                self.vector_store.save_local("faiss_index")
                
                return True
                
        except Exception as e:
            logger.error(f"Vector store update error: {str(e)}")
            return False

class DietAgent:
    """Agent for generating personalized diet plans"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self._initialize_prompt()
        self.specialized_diets = {
            "kidney_stone": {
                "veg": DietPlan(
                    breakfast="Low-oxalate cereal with milk",
                    lunch="Tofu with white rice",
                    dinner="Vegetable soup with bread",
                    snacks="Yogurt or apple",
                    notes="Limit salt, drink plenty water"
                ),
                "non_veg": DietPlan(
                    breakfast="Low-oxalate cereal with milk",
                    lunch="Chicken with white rice",
                    dinner="Fish with steamed vegetables",
                    snacks="Yogurt or apple",
                    notes="Limit salt, drink plenty water"
                )
            },
            "diabetes": {
                "veg": DietPlan(
                    breakfast="Whole grain toast, avocado",
                    lunch="Lentil soup, quinoa",
                    dinner="Chickpea curry, brown rice",
                    snacks="Nuts or berries",
                    notes="Low glycemic index foods"
                ),
                "non_veg": DietPlan(
                    breakfast="Egg whites, whole grain toast",
                    lunch="Grilled chicken, salad",
                    dinner="Baked fish, vegetables",
                    snacks="Greek yogurt, nuts",
                    notes="Low glycemic index foods"
                )
            },
            "hypertension": {
                "veg": DietPlan(
                    breakfast="Oatmeal with berries",
                    lunch="Spinach salad, beans",
                    dinner="Vegetable stir-fry, tofu",
                    snacks="Unsalted nuts, fruit",
                    notes="Low sodium, DASH diet"
                ),
                "non_veg": DietPlan(
                    breakfast="Oatmeal with berries",
                    lunch="Grilled turkey, vegetables",
                    dinner="Baked salmon, vegetables",
                    snacks="Unsalted nuts, fruit",
                    notes="Low sodium, DASH diet"
                )
            },
            "fever": {
                "veg": DietPlan(
                    breakfast="Oatmeal with honey",
                    lunch="Vegetable soup, bread",
                    dinner="Khichdi with vegetables",
                    snacks="Fresh fruit or coconut water",
                    notes="Stay well hydrated"
                ),
                "non_veg": DietPlan(
                    breakfast="Toast with honey",
                    lunch="Chicken soup, crackers",
                    dinner="Boiled rice with fish",
                    snacks="Fresh fruit or coconut water",
                    notes="Stay well hydrated"
                )
            }
        }

    def _initialize_prompt(self):
        """Initialize diet agent prompt"""
        self.prompt = """You are a nutrition specialist. Be concise.
Context: {context}
Query: {query}
Chat History: {chat_history}

First, determine if this query requires a diet plan. Only provide a diet plan if the query is health-related and diet is relevant.
If a diet plan is not needed (for greetings, general questions, etc.), return: {"diet_needed": false}

If a diet plan is needed, determine if the person is vegetarian. Consider any mention of vegetarian, vegan, or plant-based preferences.
Then provide a simple and short diet plan with:
1. Breakfast (1 item)
2. Lunch (1 item)
3. Dinner (1 item)
4. Snacks (1 item)
5. Brief note (1-2 words)
6. Condition (what health condition the diet addresses)
7. Dietary preference (vegetarian or non-vegetarian)

Format as JSON with keys: diet_needed, breakfast, lunch, dinner, snacks, notes, condition, preference.
Keep each suggestion under 5 words. Total response must be under 50 words."""

        self.agent = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            ("human", "{input}")
        ]) | self.llm | StrOutputParser()

    async def generate_diet_plan(self, query: str, context: str, chat_history: str) -> Optional[Dict]:
        """Generate a diet plan based on the conversation"""
        try:
            query_lower = query.lower()
            if query_lower in ["hi", "hello", "hey", "hi there", "hello there"] and len(query_lower) < 10:
                return None
                
            is_vegetarian = "vegetarian" in query_lower or "vegan" in query_lower or "plant-based" in query_lower
            diet_preference = "veg" if is_vegetarian else "non_veg"
            
            condition = None
            plan_title = "Personalized Diet Plan"
            
            if "kidney" in query_lower and ("stone" in query_lower or "stones" in query_lower):
                condition = "kidney_stone"
                plan_title = "Diet Plan for Kidney Stone"
            elif "diabetes" in query_lower or "blood sugar" in query_lower:
                condition = "diabetes"
                plan_title = "Diet Plan for Diabetes"
            elif "hypertension" in query_lower or "high blood pressure" in query_lower:
                condition = "hypertension"
                plan_title = "Diet Plan for Hypertension"
            elif "fever" in query_lower or "temperature" in query_lower or "flu" in query_lower:
                condition = "fever"
                plan_title = "Diet Plan for Fever"
                
            if condition and condition in self.specialized_diets:
                specialized_diet = self.specialized_diets[condition][diet_preference]
                return {
                    "diet_plan": specialized_diet,
                    "condition": condition,
                    "title": plan_title,
                    "preference": diet_preference
                }
                
            if any(term in query_lower for term in [
                "health", "diet", "food", "eat", "meal", "nutrition", 
                "sick", "ill", "unwell", "symptoms", "suffering", "condition", 
                "pain", "ache", "hurt", "doctor", "hospital", "medicine"
            ]):
                response = await self.agent.ainvoke({
                    "input": query,
                    "context": context,
                    "query": query,
                    "chat_history": chat_history
                })
                
                try:
                    diet_data = json.loads(response)
                    
                    if not diet_data.get("diet_needed", True):
                        return None
                        
                    detected_condition = diet_data.get("condition", "general health")
                    detected_preference = diet_data.get("preference", diet_preference)
                    
                    diet_plan = DietPlan(
                        breakfast=diet_data.get("breakfast", "Oatmeal with fruits"),
                        lunch=diet_data.get("lunch", "Grilled chicken salad" if diet_preference == "non_veg" else "Lentil soup with salad"),
                        dinner=diet_data.get("dinner", "Baked fish with vegetables" if diet_preference == "non_veg" else "Vegetable stir fry with tofu"),
                        snacks=diet_data.get("snacks", "Yogurt with nuts"),
                        notes=diet_data.get("notes", "Stay hydrated")
                    )
                    
                    plan_title = f"Diet Plan for {detected_condition.title()}"
                    
                    return {
                        "diet_plan": diet_plan,
                        "condition": detected_condition,
                        "title": plan_title,
                        "preference": detected_preference
                    }
                    
                except json.JSONDecodeError:
                    default_condition = "Fever" if "fever" in query_lower else "General Recovery"
                    default_plan = DietPlan(
                        breakfast="Oatmeal with fruits" if diet_preference == "veg" else "Eggs with whole grain toast",
                        lunch="Lentil soup with salad" if diet_preference == "veg" else "Chicken soup with vegetables",
                        dinner="Vegetable stir fry with tofu" if diet_preference == "veg" else "Baked fish with vegetables",
                        snacks="Yogurt with nuts" if diet_preference == "veg" else "Greek yogurt with nuts",
                        notes="Stay hydrated, get rest"
                    )
                    
                    return {
                        "diet_plan": default_plan,
                        "condition": default_condition,
                        "title": f"Diet Plan for {default_condition}",
                        "preference": diet_preference
                    }
            
            return None
                
        except Exception as e:
            if any(term in query_lower for term in ["fever", "sick", "ill", "symptoms"]):
                condition = "Fever" if "fever" in query_lower else "Recovery"
                fallback_plan = DietPlan(
                    breakfast="Light porridge or oatmeal",
                    lunch="Vegetable soup" if diet_preference == "veg" else "Chicken soup",
                    dinner="Rice with lentils" if diet_preference == "veg" else "Rice with boiled chicken",
                    snacks="Fresh fruits or coconut water",
                    notes="Stay hydrated, rest well"
                )
                return {
                    "diet_plan": fallback_plan,
                    "condition": condition,
                    "title": f"Diet Plan for {condition}",
                    "preference": diet_preference
                }
            return None

class HealthcareAgent:
    """Healthcare agent with concise response generation"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chat_history = []
        self.doc_processor = DocumentProcessor()
        self.diet_agent = DietAgent()
        self._initialize_prompts()
        self.agents = self._initialize_agents()

    def _initialize_prompts(self):
        """Initialize prompts optimized for concise responses"""
        self.prompts = {
            'main_agent': """You are a healthcare coordinator AI. Be direct and concise.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide a brief response with:
1. Key medical concepts (2-3 points)
2. Necessary specialist consultations
3. Quick initial assessment
Limit response to 3-4 sentences.""",

            'diagnosis_agent': """You are a medical diagnosis specialist. Be concise.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide brief:
1. Key symptoms identified
2. Top 2-3 potential conditions
3. Immediate next steps
Limit to 3-4 key points.""",

            'treatment_agent': """You are a treatment specialist. Be direct.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide only:
1. Top 1-2 treatment options
2. Key lifestyle changes
3. Critical warning signs
Keep response under 100 words.""",

            'research_agent': """You are a medical research specialist. Be brief.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide only:
1. Most relevant research finding
2. Key clinical guideline
3. Primary recommendation
Limit to 2-3 sentences.""",

            'synthesis_agent': """You are a medical information synthesizer. Be concise.
Context: {context}
Query: {query}
Chat History: {chat_history}
Agent Responses: {agent_responses}

Provide a clear, concise summary:
1. Main recommendation
2. Key action items
3. Important warnings (if any)

Keep the final response under 150 words and focus on practical next steps.
For simple queries (like greetings), respond in one short sentence."""
        }

    def _initialize_agents(self):
        """Initialize enhanced agent system"""
        return {
            name: ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "{input}")
            ]) | self.llm | StrOutputParser()
            for name, prompt in self.prompts.items()
        }

    def _format_chat_history(self) -> str:
        """Format chat history for context"""
        formatted = []
        for msg in self.chat_history[-5:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def process_documents(self, files: List[tuple], status_callback: Callable = None) -> bool:
        """
        Process documents with detailed status updates
        files: List of tuples (filename, content_bytes, mime_type)
        """
        try:
            processed_docs = []
            
            for idx, (filename, content, mime_type) in enumerate(files):
                doc = await self.doc_processor.process_file(
                    filename,
                    content,
                    mime_type,
                    lambda p, m: status_callback('document_processor', 'working', (idx / len(files)) + (p / len(files)), m) if status_callback else None
                )
                if doc:
                    processed_docs.append(doc)

            if processed_docs:
                success = await self.doc_processor.update_vector_store(
                    processed_docs,
                    lambda p, m: status_callback('document_processor', 'working', 0.8 + (p * 0.2), m) if status_callback else None
                )
                
                if success:
                    if status_callback: status_callback('document_processor', 'completed', 1.0, "Documents processed successfully")
                    return True

            if status_callback: status_callback('document_processor', 'error', 0, "Document processing failed")
            return False
            
        except Exception as e:
            if status_callback: status_callback('document_processor', 'error', 0, str(e))
            return False

    async def get_relevant_context(self, query: str) -> str:
        """Get relevant context from vector store"""
        try:
            if self.doc_processor.vector_store:
                docs = self.doc_processor.vector_store.similarity_search(query, k=3)
                return "\n\n".join(doc.page_content for doc in docs)
            return ""
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""

    async def process_query(self, query: str, status_callback: Callable = None) -> Dict[str, Union[AgentResponse, DietPlan]]:
        """Process query through multi-agent system"""
        responses = {}
        context = await self.get_relevant_context(query)
        chat_history = self._format_chat_history()
        
        try:
            if status_callback: status_callback('main_agent', 'working', 0.2, "Analyzing query")
            main_response = await self._get_agent_response('main_agent', query, context, chat_history)
            responses['main_agent'] = main_response
            if status_callback: status_callback('main_agent', 'completed', 1.0, "Analysis complete")

            if status_callback:
                status_callback('diagnosis_agent', 'working', 0.2, "Analyzing symptoms")
                status_callback('treatment_agent', 'working', 0.2, "Evaluating treatments")
                status_callback('research_agent', 'working', 0.2, "Reviewing research")
                status_callback('diet_agent', 'working', 0.2, "Creating diet plan")

            specialist_tasks = [
                self._get_agent_response('diagnosis_agent', query, context, chat_history),
                self._get_agent_response('treatment_agent', query, context, chat_history),
                self._get_agent_response('research_agent', query, context, chat_history),
                self.diet_agent.generate_diet_plan(query, context, chat_history)
            ]

            specialist_responses = await asyncio.gather(*specialist_tasks)
            
            for agent_name, response in zip(
                ['diagnosis_agent', 'treatment_agent', 'research_agent'],
                specialist_responses[:-1]
            ):
                responses[agent_name] = response
                if status_callback:
                    status_callback(agent_name, 'completed', 1.0, f"{agent_name.split('_')[0].title()} analysis complete")
                
            responses['diet_plan'] = specialist_responses[-1]
            if status_callback: status_callback('diet_agent', 'completed', 1.0, "Diet plan generated")

            if status_callback: status_callback('synthesis_agent', 'working', 0.5, "Synthesizing insights")
            final_response = await self._synthesize_responses(query, context, chat_history, responses)
            responses['synthesis_agent'] = final_response
            if status_callback: status_callback('synthesis_agent', 'completed', 1.0, "Response synthesis complete")

            self.chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=final_response.content)
            ])

            return responses

        except Exception as e:
            if status_callback:
                for agent in self.agents.keys():
                    status_callback(agent, 'error', 0, str(e))
                status_callback('diet_agent', 'error', 0, str(e))
            raise Exception(f"Query processing error: {str(e)}")

    async def _get_agent_response(self, agent_name: str, query: str, context: str, chat_history: str) -> AgentResponse:
        """Get response from specific agent with metadata"""
        start_time = time.time()
        
        try:
            response = await self.agents[agent_name].ainvoke({
                "input": query,
                "context": context,
                "query": query,
                "chat_history": chat_history
            })
            
            # Sanitize PII from response
            response = sanitize_text(response)
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "context_length": len(context),
                "query_length": len(query)
            }
            
            return AgentResponse(
                agent_name=agent_name,
                content=response,
                confidence=0.85, 
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Agent {agent_name} error: {str(e)}")

    async def _synthesize_responses(self, query: str, context: str, chat_history: str, responses: Dict[str, Union[AgentResponse, DietPlan]]) -> AgentResponse:
        """Synthesize final response from all agent responses"""
        try:
            formatted_responses = "\n\n".join([
                f"{name.upper()}:\n{response.content}"
                for name, response in responses.items()
                if name != 'synthesis_agent' and name != 'diet_plan' and hasattr(response, 'content')
            ])

            start_time = time.time()
            
            synthesis_response = await self.agents['synthesis_agent'].ainvoke({
                "input": query,
                "context": context,
                "query": query,
                "chat_history": chat_history,
                "agent_responses": formatted_responses
            })
            
            # Sanitize PII from synthesis response
            synthesis_response = sanitize_text(synthesis_response)
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "source_responses": len(responses),
                "context_used": bool(context)
            }
            
            return AgentResponse(
                agent_name="synthesis_agent",
                content=synthesis_response,
                confidence=0.9,
                metadata=metadata,
                processing_time=processing_time
            )

        except Exception as e:
            raise Exception(f"Synthesis error: {str(e)}")
