import pinecone
import os
import sys

print(f"Pinecone file: {pinecone.__file__}")
try:
    print(f"Pinecone version: {pinecone.__version__}")
except:
    print("Pinecone version: unknown")

print("Dir pinecone:")
print(dir(pinecone))

try:
    from pinecone import Pinecone
    print("✅ Successfully imported Pinecone class")
except ImportError as e:
    print(f"❌ Failed to import Pinecone class: {e}")
