import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _xor_operation(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt(self, plaintext: str) -> bytes:
        return self._xor_operation(plaintext.encode('utf-8'))
    
    def decrypt(self, ciphertext: bytes) -> str:
        return self._xor_operation(ciphertext).decode('utf-8')
    
    def encrypt_file(self, input_path: str, output_path: str):
        with open(input_path, 'rb') as f:
            data = f.read()
        
        encrypted = self._xor_operation(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
    
    def decrypt_file(self, input_path: str, output_path: str):
        self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file>")
        print("Set environment variable XOR_KEY for encryption key")
        sys.exit(1)
    
    action = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    key = os.environ.get('XOR_KEY', 'default_secret_key')
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    if action == 'encrypt':
        cipher.encrypt_file(input_file, output_file)
        print(f"File encrypted successfully: {output_file}")
    elif action == 'decrypt':
        cipher.decrypt_file(input_file, output_file)
        print(f"File decrypted successfully: {output_file}")
    else:
        print("Error: Action must be 'encrypt' or 'decrypt'")
        sys.exit(1)

if __name__ == "__main__":
    main()