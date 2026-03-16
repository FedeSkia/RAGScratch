import os
import sys
from dotenv import load_dotenv

load_dotenv()

print(f"Python version: {sys.version}")
print(f"API Key set: {'ANTHROPIC_API_KEY' in os.environ}")

try:
    from anthropic import Anthropic

    print("✓ Anthropic imported successfully")

    # Prova senza argomenti
    client = Anthropic()
    print("✓ Client created without arguments")

except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()