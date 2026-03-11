
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _process(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str):
        self._process_file(input_path, output_path)
    
    def decrypt_file(self, input_path: str, output_path: str):
        self._process_file(input_path, output_path)
    
    def _process_file(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        
        processed_data = self._process(data)
        
        with open(output_path, 'wb') as f_out:
            f_out.write(processed_data)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <key> <input_file> <output_file>")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    key = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    if operation not in ['encrypt', 'decrypt']:
        print("Error: Operation must be 'encrypt' or 'decrypt'")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    try:
        if operation == 'encrypt':
            cipher.encrypt_file(input_file, output_file)
            print(f"File encrypted successfully: {output_file}")
        else:
            cipher.decrypt_file(input_file, output_file)
            print(f"File decrypted successfully: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _xor_operation(self, data: bytes) -> bytes:
        key_bytes = self.key
        key_length = len(key_bytes)
        return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            
            ciphertext = self._xor_operation(plaintext)
            
            with open(output_path, 'wb') as f:
                f.write(ciphertext)
            
            return True
        except Exception as e:
            print(f"Encryption error: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str):
        return self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file>")
        print("Note: The key is read from environment variable XOR_KEY")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    key = os.environ.get('XOR_KEY')
    if not key:
        print("Error: XOR_KEY environment variable not set")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    if action == 'encrypt':
        if cipher.encrypt_file(input_file, output_file):
            print(f"File encrypted successfully: {output_file}")
        else:
            print("Encryption failed")
    elif action == 'decrypt':
        if cipher.decrypt_file(input_file, output_file):
            print(f"File decrypted successfully: {output_file}")
        else:
            print("Decryption failed")
    else:
        print("Invalid action. Use 'encrypt' or 'decrypt'")

if __name__ == "__main__":
    main()