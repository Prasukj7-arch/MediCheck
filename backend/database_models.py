from datetime import datetime
from typing import Dict, List, Optional
from supabase import Client
from backend.supabase_config import get_supabase_client

class DatabaseModels:
    def __init__(self):
        self.client: Client = get_supabase_client()
    
    def create_user(self, email: str, password_hash: str, name: str) -> Dict:
        """Create a new user in the database"""
        try:
            response = self.client.table('users').insert({
                'email': email,
                'password_hash': password_hash,
                'name': name,
                'created_at': datetime.now().isoformat()
            }).execute()
            
            if response.data:
                return response.data[0]
            else:
                raise Exception("Failed to create user")
        except Exception as e:
            raise Exception(f"Error creating user: {str(e)}")
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        try:
            response = self.client.table('users').select('*').eq('email', email).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            raise Exception(f"Error getting user: {str(e)}")
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        try:
            response = self.client.table('users').select('*').eq('id', user_id).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            raise Exception(f"Error getting user: {str(e)}")
    
    def create_appointment(self, user_id: int, doctor_name: str, specialty: str, 
                          appointment_date: str, appointment_time: str, 
                          reason: str, status: str = 'scheduled') -> Dict:
        """Create a new appointment"""
        try:
            response = self.client.table('appointments').insert({
                'user_id': user_id,
                'doctor_name': doctor_name,
                'specialty': specialty,
                'appointment_date': appointment_date,
                'appointment_time': appointment_time,
                'reason': reason,
                'status': status,
                'created_at': datetime.now().isoformat()
            }).execute()
            
            if response.data:
                return response.data[0]
            else:
                raise Exception("Failed to create appointment")
        except Exception as e:
            raise Exception(f"Error creating appointment: {str(e)}")
    
    def get_user_appointments(self, user_id: int) -> List[Dict]:
        """Get all appointments for a user"""
        try:
            response = self.client.table('appointments').select('*').eq('user_id', user_id).order('appointment_date', desc=True).execute()
            return response.data or []
        except Exception as e:
            raise Exception(f"Error getting appointments: {str(e)}")
    
    def update_appointment(self, appointment_id: int, updates: Dict) -> Dict:
        """Update an appointment"""
        try:
            response = self.client.table('appointments').update(updates).eq('id', appointment_id).execute()
            if response.data:
                return response.data[0]
            else:
                raise Exception("Failed to update appointment")
        except Exception as e:
            raise Exception(f"Error updating appointment: {str(e)}")
    
    def delete_appointment(self, appointment_id: int) -> bool:
        """Delete an appointment"""
        try:
            response = self.client.table('appointments').delete().eq('id', appointment_id).execute()
            return True
        except Exception as e:
            raise Exception(f"Error deleting appointment: {str(e)}")
    
    def get_available_slots(self, date: str, specialty: str = None) -> List[Dict]:
        """Get available appointment slots for a date"""
        try:
            query = self.client.table('appointments').select('appointment_time').eq('appointment_date', date)
            if specialty:
                query = query.eq('specialty', specialty)
            
            response = query.execute()
            booked_times = [apt['appointment_time'] for apt in response.data or []]
            
            # Generate available slots (9 AM to 5 PM, 1-hour slots)
            all_slots = [f"{hour:02d}:00" for hour in range(9, 17)]
            available_slots = [slot for slot in all_slots if slot not in booked_times]
            
            return [{'time': slot} for slot in available_slots]
        except Exception as e:
            raise Exception(f"Error getting available slots: {str(e)}")
    
    def get_specialties(self) -> List[str]:
        """Get list of available medical specialties"""
        return [
            "Cardiology",
            "Dermatology",
            "Endocrinology",
            "Gastroenterology",
            "General Medicine",
            "Neurology",
            "Oncology",
            "Orthopedics",
            "Pediatrics",
            "Psychiatry",
            "Radiology",
            "Surgery",
            "Urology"
        ]

# Global instance
db_models = DatabaseModels()
