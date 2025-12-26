
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def encrypt(self, data: bytes) -> bytes:
        encrypted = bytearray()
        key_length = len(self.key)
        for i, byte in enumerate(data):
            encrypted.append(byte ^ self.key[i % key_length])
        return bytes(encrypted)
    
    def decrypt(self, data: bytes) -> bytes:
        return self.encrypt(data)

def process_file(input_path: str, output_path: str, key: str, mode: str):
    cipher = XORCipher(key)
    
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        if mode == 'encrypt':
            processed_data = cipher.encrypt(data)
            action = "Encrypted"
        elif mode == 'decrypt':
            processed_data = cipher.decrypt(data)
            action = "Decrypted"
        else:
            raise ValueError("Mode must be 'encrypt' or 'decrypt'")
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"{action} file saved to: {output_path}")
        print(f"Original size: {len(data)} bytes")
        print(f"Processed size: {len(processed_data)} bytes")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key> <encrypt|decrypt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    mode = sys.argv[4].lower()
    
    if mode not in ['encrypt', 'decrypt']:
        print("Error: Mode must be 'encrypt' or 'decrypt'")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    process_file(input_file, output_file, key, mode)

if __name__ == "__main__":
    main()