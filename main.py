import os
import gradio as gr
import pandas as pd
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from transformers import (
    AutoModelForImageClassification,
    AutoProcessor,
    pipeline
)
from mistralai import Mistral
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from dotenv import load_dotenv
import kagglehub
from datasets import load_dataset
from mistralai import Mistral
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr
import numpy as np


# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

class MistralLLM(LLM):
    model: str = "mistral-large-latest"
    _client: Mistral = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = Mistral(api_key=MISTRAL_API_KEY)

    @property
    def _llm_type(self) -> str:
        return "mistral"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class AgentSystem:
    def __init__(self):
        self.mistral_llm = MistralLLM()
        self.initialize_models()
        self.load_datasets()
        self.setup_agents()
   
    def initialize_models(self):
        """Initialize all required AI models"""
        # Identity verification models
        self.gender_model = AutoModelForImageClassification.from_pretrained(
            "rizvandwiki/gender-classification"
        )
        self.gender_processor = AutoProcessor.from_pretrained(
            "rizvandwiki/gender-classification"
        )
        
        self.age_model = AutoModelForImageClassification.from_pretrained(
            "nateraw/vit-age-classifier"
        )
        self.age_processor = AutoProcessor.from_pretrained(
            "nateraw/vit-age-classifier"
        )
        
        # X-ray analysis model
        self.xray_model = pipeline(
            model="lxyuan/vit-xray-pneumonia-classification"
        )

    def load_datasets(self):
        """Load medical datasets"""
        # Load patient database
        path = kagglehub.dataset_download(
            "uom190346a/disease-symptoms-and-patient-profile-dataset"
        )
        self.patient_data = pd.read_csv(
            path + '/Disease_symptom_and_patient_profile_dataset.csv'
        )
        
        # Load X-ray dataset
        self.xray_dataset = load_dataset(
            "keremberke/chest-xray-classification", 
            name="full"
        )

    def setup_agents(self):
        """Set up all specialized agents"""
        # Front Desk Agent Tools
        self.front_desk_tools = [
            Tool(
                name="verify_identity",
                func=self.verify_identity,
                description="Verify patient identity using photo ID"
            )
        ]

        # Physician Agent Tools
        self.physician_tools = [
            Tool(
                name="examine_patient",
                func=self.examine_patient,
                description="Examine patient symptoms and vital signs"
            ),
            Tool(
                name="medical_history",
                func=self.check_medical_history,
                description="Check patient's medical history"
            )
        ]

        # Radiologist Agent Tools
        self.radiologist_tools = [
            Tool(
                name="analyze_xray",
                func=self.analyze_xray,
                description="Analyze chest X-ray for pneumonia"
            )
        ]
        
        # Initialize agents
        self.front_desk_agent = initialize_agent(
            self.front_desk_tools,
            self.mistral_llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        self.physician_agent = initialize_agent(
            self.physician_tools,
            self.mistral_llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        self.radiologist_agent = initialize_agent(
            self.radiologist_tools,
            self.mistral_llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def verify_identity(self, image_path: str) -> Dict[str, Any]:
        """Front desk agent: Verify patient identity"""
        # Convert numpy array to PIL Image if needed
        if isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = Image.open(image_path)
        
        # Rest of the function remains the same
        gender_inputs = self.gender_processor(images=image, return_tensors="pt")
        gender_outputs = self.gender_model(**gender_inputs)
        predicted_gender = self.gender_model.config.id2label[
            gender_outputs.logits.argmax(-1).item()
        ]
        
        age_inputs = self.age_processor(images=image, return_tensors="pt")
        age_outputs = self.age_model(**age_inputs)
        predicted_age = self.age_model.config.id2label[
            age_outputs.logits.argmax(-1).item()
        ]
        
        return {
            "gender": predicted_gender,
            "age_range": predicted_age,
            "verification_status": "VERIFIED"
        }

    def examine_patient(self, patient_data) -> Dict[str, Any]:
        """Physician agent: Examine patient"""
        if isinstance(patient_data, str):
            # If CSV file path, read it
            patient_data = pd.read_csv(patient_data).iloc[0].to_dict()
        elif not isinstance(patient_data, dict):
            patient_data = {}
            
        symptoms = []
        for symptom in ["Cough", "Fever", "Fatigue", "Difficulty Breathing"]:
            if str(patient_data.get(symptom, "No")).lower() == "yes":
                symptoms.append(symptom)
        
        vitals = {
            "blood_pressure": str(patient_data.get("Blood Pressure", "Not recorded")),
            "temperature": str(patient_data.get("Temperature", "Not recorded")),
            "heart_rate": str(patient_data.get("Heart Rate", "Not recorded"))
        }
        
        return {
            "symptoms": symptoms,
            "vitals": vitals,
            "recommendation": "Requires X-ray" if len(symptoms) >= 2 else "Regular checkup"
        }

    def check_medical_history(self, patient_id: str) -> Dict[str, Any]:
        """Physician agent: Check medical history"""
        patient_record = self.patient_data[
            self.patient_data["Patient_ID"] == patient_id
        ]
        if patient_record.empty:
            return {"status": "No records found"}
        
        return {
            "previous_conditions": patient_record["Disease"].values[0],
            "last_visit": "2024-01-15",  # Example date
            "medications": ["Med A", "Med B"]  # Example medications
        }

    def analyze_xray(self, image_data: Any) -> Dict[str, Any]:
        try:
            # Handle numpy array from gradio
            if isinstance(image_data, np.ndarray):
                image = Image.fromarray(np.uint8(image_data))
            else:
                image = Image.open(image_data)
            
            results = self.xray_model(image)
            highest_result = max(results, key=lambda x: x['score'])
            
            return {
                "condition": highest_result['label'],
                "confidence": f"{highest_result['score']*100:.1f}%",
                "recommendation": "Urgent care" if highest_result['label'] == "PNEUMONIA" else "All clear"
            }
        except Exception as e:
            print(f"Error in analyze_xray: {str(e)}")
            raise
    
    def chat_with_mistral(self, message: str) -> str:
        """Interact with Mistral chat model"""
        chat_response = self.mistral_client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(role="user", content=message)
            ]
        )
        return chat_response.choices[0].message.content

def create_gradio_interface():
    system = AgentSystem()
    
    def process_patient(id_photo, medical_record, xray_image=None):
        if id_photo is None:
            return "Error: No ID photo provided"
            
        # Convert image to numpy array if needed
        if isinstance(id_photo, dict) and 'image' in id_photo:
            id_photo = id_photo['image']
            
        medical_data = {
            "Blood Pressure": "120/80",
            "Temperature": "98.6",
            "Heart Rate": "72",
            "Cough": "Yes",
            "Fever": "Yes",
            "Fatigue": "Yes",
            "Difficulty Breathing": "No"
        }
        
        output = "🏥 Medical AI System Report\n\n"
        
        try:
            verification = system.verify_identity(id_photo)
            output += f"1️⃣ Front Desk Verification:\n"
            output += f"Gender: {verification['gender']}\n"
            output += f"Age Range: {verification['age_range']}\n"
            output += f"Status: {verification['verification_status']}\n\n"
        except Exception as e:
            output += f"Verification failed: {str(e)}\n\n"
            return output

        try:
            examination = system.examine_patient(medical_data)
            output += "2️⃣ Physician Examination:\n"
            output += f"Symptoms: {', '.join(examination['symptoms'])}\n"
            output += "Vitals:\n"
            for key, value in examination['vitals'].items():
                output += f"- {key}: {value}\n"
            output += f"Recommendation: {examination['recommendation']}\n\n"
        except Exception as e:
            output += f"Examination failed: {str(e)}\n\n"
            return output

        if xray_image is not None and examination['recommendation'] == "Requires X-ray":
            if isinstance(xray_image, dict) and 'image' in xray_image:
                xray_image = xray_image['image']
            try:
                xray_results = system.analyze_xray(xray_image)
                output += "3️⃣ Radiologist Analysis:\n"
                output += f"Condition: {xray_results['condition']}\n"
                output += f"Confidence: {xray_results['confidence']}\n"
                output += f"Recommendation: {xray_results['recommendation']}\n"
            except Exception as e:
                output += f"X-ray analysis failed: {str(e)}\n"

        return output

    # Create Gradio interface
    with gr.Blocks(title="Medical Multi-Agent System") as demo:
        gr.Markdown("# 🏥 Medical Multi-Agent System")
        
        with gr.Tab("Patient Processing"):
            with gr.Row():
                with gr.Column():
                    id_photo = gr.Image(label="Patient ID Photo")
                    medical_file = gr.File(label="Medical Records (CSV)")
                    xray_image = gr.Image(label="Chest X-Ray (if needed)")
                with gr.Column():
                    output = gr.Textbox(label="System Report", lines=15)
            
            process_btn = gr.Button("Process Patient")
            process_btn.click(
                fn=process_patient,
                inputs=[id_photo, medical_file, xray_image],
                outputs=[output]
            )
        
        with gr.Tab("System Information"):
            gr.Markdown("""
            ### System Components
            
            1. **Front Desk Agent** 🪪
               - Identity verification
               - Age and gender detection
               - Document processing
            
            2. **Physician Agent** 👨‍⚕️
               - Symptom analysis
               - Medical history review
               - Treatment recommendations
            
            3. **Radiologist Agent** 🩻
               - X-ray analysis
               - Pneumonia detection
               - Urgent care assessment
            """)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()