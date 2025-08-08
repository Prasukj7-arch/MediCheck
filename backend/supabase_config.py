import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseConfig:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
    
    def get_client(self) -> Client:
        return self.client

# Global instance
supabase_config = None

def get_supabase_client() -> Client:
    global supabase_config
    if supabase_config is None:
        supabase_config = SupabaseConfig()
    return supabase_config.get_client()
